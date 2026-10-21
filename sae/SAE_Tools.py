import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import einops
from functools import partial
from typing import List, Tuple, Dict, Any, Union, Callable, cast
from tqdm import tqdm
from collections import OrderedDict
from transformer_lens import (
    HookedTransformer,
    ActivationCache,
)
import math
from .autoencoder import TopKSAE, SAEOut, ReluSAE, SAE_Template, TopKMatryoshka, BatchTopKSAE, BatchTopKMatryoshka
from transformer_lens.hook_points import HookedRootModule, HookPoint
from sae_lens import SAE as SAE_LENS

from torch.utils.data import DataLoader, TensorDataset

import plotly.express as px
from jaxtyping import Float, Int
from rich import print as rprint
from rich.table import Table
from tabulate import tabulate

def load_sae_model(file_path: str, model: HookedTransformer, device: str | t.device = "cpu", **kwargs) -> SAE_Template:
    '''
    Load TopKSAE from path
    '''
    tmp = t.load(file_path, map_location=device)
    
    sae_state_dict = {k: v for k, v in tmp["state_dict"].items() if "autoencoder" in k}
    sae_state_dict = {k.replace("autoencoder.", ""): v for k, v in sae_state_dict.items()}
      
    hyperparameters = tmp["hyper_parameters"]
    model_cfg = model.cfg
    dead_steps_threshold = hyperparameters["dead_steps_threshold"] or hyperparameters["dead_tokens_threshold"] // (
        hyperparameters["batch_size"] * hyperparameters["max_length"] * hyperparameters["accumulate_grad_batches"]
    )
    
    sae_name = hyperparameters.get("sae_name", "topk")
    
    if sae_name == "relusae":
        class_sae = ReluSAE
    elif sae_name == "topk" or sae_name == "topk_custom":
        class_sae = TopKSAE
    elif sae_name == "topk_matryoshka":
        class_sae = TopKMatryoshka
    elif sae_name == "batchtopk_matryoshka":
        class_sae = BatchTopKMatryoshka
    elif sae_name == "batchtopk":
        class_sae = BatchTopKSAE
    else:
        raise ValueError(f"Unknown SAE name: {hyperparameters['sae_name']}")
        
    if hyperparameters.get("list_k", None): # if we have list k
        sae = class_sae(
            d_in=model_cfg.d_model,
            d_sae=int(hyperparameters["expansion_factor"] * model_cfg.d_model),
            hook_names=hyperparameters["hook_names"],
            dead_steps_threshold=dead_steps_threshold,
            dead_threshold=hyperparameters["dead_threshold"],
            auxk=hyperparameters["auxk"],
            standardize=hyperparameters["standardize"],
            sizes=hyperparameters.get("sizes", [1]),
            list_k=hyperparameters["list_k"],
            skip_affine=hyperparameters.get("skip_affine", False),
            **kwargs,
        ).to(device)
    else:
        sae = class_sae(
            d_in=model_cfg.d_model,
            d_sae=int(hyperparameters["expansion_factor"] * model_cfg.d_model),
            hook_names=hyperparameters["hook_names"],
            k=hyperparameters["k"],
            dead_steps_threshold=dead_steps_threshold,
            dead_threshold=hyperparameters["dead_threshold"],
            auxk=hyperparameters["auxk"],
            standardize=hyperparameters["standardize"],
            sizes=hyperparameters.get("sizes", [1]),
            **kwargs,
        ).to(device)
    sae.load_state_dict(sae_state_dict, strict=False)
    return sae

def load_multi_sae_model(file_path: str, model: HookedTransformer, device: str | t.device = "cpu", **kwargs) -> List[SAE_Template]:
    '''
    Load TopKSAE from path
    '''
    tmp = t.load(file_path, map_location=device)
    
    multisae_state_dict = {k: v for k, v in tmp["state_dict"].items() if "autoencoders" in k}
      
    hyperparameters = tmp["hyper_parameters"]
    model_cfg = model.cfg
    dead_steps_threshold = hyperparameters["dead_steps_threshold"] or hyperparameters["dead_tokens_threshold"] // (
        hyperparameters["batch_size"] * hyperparameters["max_length"] * hyperparameters["accumulate_grad_batches"]
    )
    
    sae_name = hyperparameters.get("sae_name", "topk")
    
    if sae_name == "relusae":
        class_sae = ReluSAE
    elif sae_name == "topk" or sae_name == "topk_custom":
        class_sae = TopKSAE
    elif sae_name == "topk_matryoshka":
        class_sae = TopKMatryoshka
    elif sae_name == "batchtopk_matryoshka":
        class_sae = BatchTopKMatryoshka
    elif sae_name == "batchtopk":
        class_sae = BatchTopKSAE
    else:
        raise ValueError(f"Unknown SAE name: {hyperparameters['sae_name']}")
        
    list_saes = []
    for i in range(len(hyperparameters["hook_names"])):
        sae_state_dict = {k.replace(f"autoencoders.{i}.", ""): v for k, v in multisae_state_dict.items()}
        if hyperparameters.get("list_k", None): # if we have list k
            sae = class_sae(
                d_in=model_cfg.d_model,
                d_sae=int(hyperparameters["expansion_factor"] * model_cfg.d_model),
                hook_names=[hyperparameters["hook_names"][i]],
                dead_steps_threshold=dead_steps_threshold,
                dead_threshold=hyperparameters["dead_threshold"],
                auxk=hyperparameters["auxk"],
                standardize=hyperparameters["standardize"],
                sizes=hyperparameters.get("sizes", [1]),
                list_k=hyperparameters["list_k"],
                skip_affine=hyperparameters.get("skip_affine", False),
                **kwargs,
            ).to(device)
        else:
            sae = class_sae(
                d_in=model_cfg.d_model,
                d_sae=int(hyperparameters["expansion_factor"] * model_cfg.d_model),
                hook_names=[hyperparameters["hook_names"][i]],
                k=hyperparameters["k"],
                dead_steps_threshold=dead_steps_threshold,
                dead_threshold=hyperparameters["dead_threshold"],
                auxk=hyperparameters["auxk"],
                standardize=hyperparameters["standardize"],
                sizes=hyperparameters.get("sizes", [1]),
                **kwargs,
            ).to(device)
        sae.load_state_dict(sae_state_dict, strict=False)
        list_saes.append(sae)
    return list_saes

def normalize_activations(
    activations: Tensor,
    eps: float = 1e-6,
) ->  Tensor:
    '''
    Normalize activations
    '''
    exclude_first_dim = [i for i in range(1, activations.ndim)]
    mean = activations.mean(dim=exclude_first_dim, keepdim=True)
    std = activations.std(dim=exclude_first_dim, keepdim=True)
    activations = (activations - mean) / (std + eps)
    return activations

def get_sae_activations(
    sae: HookedRootModule,
    dataloader: DataLoader,
    device: str | t.device,
    normalize: bool = False,
    no_tqdm: bool = False,
) -> Tuple[Tensor, Tensor]:
    '''
    Return the indices and values of HookedRootModule given the cached activations dataset
    '''
    with t.no_grad():
        sae.eval()
        sae.to(device)
        
        latents_list: List[Tensor] = []
        max_k = [0] # to change outside the function
        def hook_fn(inputs: Tensor, hook: HookPoint, max_k = max_k) -> None:
            # (b, s, d)
            inputs = inputs
            mask = inputs > 0
            max_k[0] = max(max_k[0], mask.sum(-1).max().item()) # update the max_k
            
            # convert to sparse tensor to save memory
            indices = mask.nonzero(as_tuple=False).t() 
            values = inputs[mask] 
            sparse_tensor = t.sparse_coo_tensor(indices, values, inputs.shape).coalesce().detach().cpu()
            latents_list.append(sparse_tensor)

        for act in tqdm(dataloader, disable=no_tqdm):
            act = act[0]
            act = normalize_activations(act.to(device)) if normalize else act.to(device)
            sae.run_with_hooks(
                act, 
                fwd_hooks=[(lambda name: "acts_post" in name, hook_fn)],
            )

        indices_list = []
        values_list = []
        # transform into values and indices
        for sparse_latent in latents_list:
            latents = sparse_latent.to_dense() # (b, s, d)
            values, indices = t.topk(
                latents, k=max_k[0], sorted=False,
            )
            indices_list.append(indices)
            values_list.append(values)
            
        indices = t.cat(indices_list)
        values = t.cat(values_list)
    
    return indices, values


def show_activation_histogram(
    sae: SAE_Template | SAE_LENS,
    values: Tensor,
    indices: Tensor,
    threshold: float = 1e-3,
) -> None:
    '''
    Show activation histogram given the values and indices of SAE_Template or SAE from sae_lens
    '''
    batch_size, pos, k = values.shape
    
    activation_counts = t.zeros(sae.cfg.d_sae, dtype=t.float32)
    activated = values > threshold
    activated_indices = indices[activated]

    # accumulate activation counts
    activation_counts.scatter_add_(0, activated_indices.flatten(), t.ones_like(activated_indices, dtype=t.float32).flatten())

    # Compute activation fraction
    activation_fraction = activation_counts / (batch_size * pos)
    

    px.histogram(
        activation_fraction.cpu().numpy(),
        nbins=50,
        title=f"ACTIVATIONS DENSITY",
        labels={"value": "Activation"},
        width=800,
        template="ggplot2",
        color_discrete_sequence=["darkorange"],
    ).update_layout(bargap=0.02, showlegend=False).show()


def show_feature_act_histogram(
    sae: SAE_Template | SAE_LENS,
    feat_idx: int,
    values: Tensor, 
    indices: Tensor, 
    threshold=1e-3,
) -> None:
    '''
    Show feature activation histogram given the values and indices of SAE_Template or SAE in sae_lens
    '''
    batch_size, pos, k = values.shape
    
    activation_counts = t.zeros(sae.cfg.d_sae, dtype=t.float32)
    activated = values > threshold
    # filter out non-activated indices
    activated_indices = t.where(activated, indices, t.zeros_like(indices))

    # filter out feature activations
    activation_feat = values[activated_indices.long() == feat_idx] 
    
    px.histogram(
        activation_feat.cpu().numpy(),
        nbins=50,
        title=f"ACTIVATIONS DENSITY",
        labels={"value": "Activation"},
        width=800,
        template="ggplot2",
        color_discrete_sequence=["darkorange"],
    ).update_layout(bargap=0.02, showlegend=False).show()
    
def show_feature_act_on_component_histogram(
    sae: SAE_Template | SAE_LENS,
    feat_idx: int,
    values: Tensor, 
    indices: Tensor, 
    toks: Tensor,
    threshold=1e-3,
    figsize: Tuple[int, int] = (7,5),
    labelfont: int = 14,
    titlefont: int = 16,
    xtickfont: int = 8,
    ytickfont: int = 12,
    tickrotation: int = 45,
) -> None:
    '''
    Show feature activation histogram on model components determined by sae.cfg.hook_names
    The component in hook_names should match the order of the stacked activation in values / indices
    '''
    if not hasattr(sae.cfg, "hook_names") or len(sae.cfg.hook_names) == 1: # type: ignore
        print("This is a single layers SAE.")
        return
    
    num_examples, seq_len = toks.shape
    num_components = len(sae.cfg.hook_names) # type: ignore
    assert len(indices) == num_examples * num_components
    assert len(indices) == len(values)
    
    activated = ((values > threshold) & (indices == feat_idx)).any(dim=-1) # (n_comp * n_examples, seq_len)
    activated_bins = activated.view(num_components, num_examples, seq_len).sum(dim=(1,2))
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=figsize)  # Set the figure size
    plt.bar(sae.cfg.hook_names, activated_bins.numpy(), color='skyblue', edgecolor='black')  # type: ignore

    # Add labels and title
    plt.xlabel("Components", fontsize=labelfont, fontweight='bold', style='italic')
    plt.ylabel("Activation Count", fontsize=labelfont, fontweight='bold', style='italic')
    plt.title("Feature Activation on Components", fontsize=titlefont, fontweight='bold')

    # Customize tick labels
    plt.xticks(fontsize=xtickfont, rotation=tickrotation, ha='right')  # Rotate x-axis labels for better readability
    plt.yticks(fontsize=ytickfont)

    # Add grid lines for better readability
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Show the plot
    plt.tight_layout()  # Adjust layout to prevent overlap
    plt.show()
    
def find_parent_and_child(allocate_parents: Tensor, child_start_id: int, parent_to_child: bool = True) -> Dict[int, List[int]]:
    '''
    allocate_parents: Tensor
        Parameter (n_child), where each element is the parent id.
    child_start_id: int
        The id of the first child feature, if not provided, return the relative child id.
    '''
    parent_child_dict = {}
    for child_idx, parent_idx in enumerate(allocate_parents):
        parent_idx = parent_idx.item() 
        if parent_idx == child_start_id: # root node
            parent_idx = -1
        if parent_to_child:
            if parent_idx not in parent_child_dict:
                parent_child_dict[parent_idx] = []
            parent_child_dict[parent_idx].append(child_idx + child_start_id)
        else:
            if (child_idx + child_start_id) not in parent_child_dict:
                parent_child_dict[child_idx + child_start_id] = []
            parent_child_dict[child_idx + child_start_id].append(parent_idx)
    
    return parent_child_dict

@t.no_grad() 
def cache_activation(
    transformer: HookedTransformer,
    input_toks: Tensor,
    hook_names: List[str],
    batch_size: int = 256,
    return_toks: bool = False,
    save_path: str | None = None,
    no_verbose: bool = False,
) -> Dict[str, Tensor] | Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    '''
    Caches the activations of the transformer for the given input tokens.
    '''
    if save_path is not None:
        try:
            all_cache = t.load(save_path)
            if not no_verbose:
                print("Activations loaded from cache.")
            return all_cache
        except:
            if not no_verbose:
                print("No cache found.")
            pass
    
    toks_loader = get_loader(input_toks, batch_size)
    
    if not no_verbose:
        print("Caching activations...")
        
    assert hook_names is not None, "hook_names must not None."
    all_cache = OrderedDict({name: [] for name in hook_names})
    all_toks = OrderedDict({name: [] for name in hook_names})
    names_filter = lambda x: x in hook_names
    
    for batch in tqdm(toks_loader, disable=no_verbose):
        batch_toks = batch[0]
        _, tmp_cache = transformer.run_with_cache(batch_toks, names_filter=names_filter)
        
        for name, cache in tmp_cache.items():
            if name.endswith("result"):
                cache = einops.rearrange(cache, "b s h d -> (b h) s d")
            all_cache[name].append(cache.cpu())
            if return_toks:
                all_toks[name].append(batch_toks.clone().cpu())
    
    for tens in all_cache.values():
        assert len(tens) > 0, "No activations cached."
    
    all_cache = {name: t.cat(cache) for name, cache in all_cache.items()}
    if return_toks:
        all_toks = {name: t.cat(toks) for name, toks in all_toks.items()}
        return all_cache, all_toks
    if save_path is not None:
        t.save(all_cache, save_path)
    return all_cache    

@t.no_grad()
def get_cache_with_toks_dataset(
    model: HookedTransformer,
    toks: Tensor, 
    hook_names: List[str] | None,
    batch_size: int,
    no_verbose: bool = False
) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
    '''
    Cache the model activations given the toks and the list of component to cache.
    The return is a dictionary of activations and toks
    '''
    
    if toks.ndim == 1:
        # single input string
        toks = toks.unsqueeze(0)
    # shape must be [batch, ...]
    activation_cache, toks_cache = cache_activation(
        model,
        toks,
        hook_names, # type: ignore
        batch_size,
        return_toks=True,
        no_verbose=no_verbose,
    )
    
    return activation_cache, toks_cache # type: ignore

@t.no_grad()
def get_sae_activation_from_tok(
    sae_dict: Dict[str, SAE_Template | SAE_LENS | Any],
    transformer: HookedTransformer,
    input_toks: Tensor,
    batch_size: int = 256,
    feature_dict: Dict[str, List[int]] | None = None,
    no_verbose: bool = False,
) -> Dict[str, Tuple[Tensor, Tensor]]:
    '''
    Caches the activations of the transformer for the given input tokens.
    '''
    saes = list(sae_dict.values())
    for sae in saes:
        sae.reset_hooks()
        
    toks_loader = get_loader(input_toks, batch_size)
    
    if not no_verbose:
        print("Caching activations...")
        
    hook_names = list(sae_dict.keys())
    
    latents_list = {hook_name: [] for hook_name in hook_names}
    max_k = {hook_name: 0 for hook_name in hook_names}
    def sae_hook_fn(tensor: Tensor, hook: HookPoint, sae_name: str, max_k = max_k) -> None:
        # (b, s, d)
        if feature_dict is not None:
            mask = t.zeros_like(tensor, dtype=t.bool)
            mask[..., feature_dict[sae_name]] = 1
            mask = mask * (tensor > 1e-3)
        else:
            mask = tensor > 1e-3
        
        # update the max_k
        max_k[sae_name] = max(max_k[sae_name], mask.sum(-1).max().item())
        
        # convert to sparse tensor to save memory
        indices = mask.nonzero(as_tuple=False).t() 
        values = tensor[mask] 
        sparse_tensor = t.sparse_coo_tensor(indices, values, tensor.shape).coalesce().detach().cpu()
        latents_list[sae_name].append(sparse_tensor)
        
    
    def hook_fn(tens: Tensor, hook: HookPoint):
        sae = sae_dict[hook.name] # type: ignore
        
        original_layer = None
        if hasattr(sae, "layer"):
            original_layer = sae.layer
            sae.layer = hook.layer() # type: ignore
            
        sae(tens)
        if hasattr(sae, "layer"):
            sae.layer = original_layer # type: ignore
            
    for sae in saes:
        sae.add_hook(
            lambda name: "acts_post" in name, partial(sae_hook_fn, sae_name=sae.cfg.hook_name), dir="fwd"
        )
        
    with transformer.hooks(
        fwd_hooks=[
            (lambda name: name in hook_names, hook_fn)
        ]
    ):
        for batch in tqdm(toks_loader, disable=no_verbose):
            batch_toks = batch[0]
            transformer(batch_toks)
            
    for sae in saes:
        sae.reset_hooks()
    
    dict_acts = {}
    
    for sae in saes:
        indices_list = []
        values_list = []
        # transform into values and indices
        for sparse_latent in latents_list[sae.cfg.hook_name]:
            latents = sparse_latent.to_dense() # (b, s, d)
            values, indices = t.topk(
                latents, k=max_k[sae.cfg.hook_name], sorted=False,
            )
            indices_list.append(indices)
            values_list.append(values)
            
        indices = t.cat(indices_list)
        values = t.cat(values_list)
        dict_acts[sae.cfg.hook_name] = (values, indices)
    
    return dict_acts    

def get_loader_from_dict(list_dict: List[Dict[str, Tensor]], batch_size: int) -> DataLoader:
    '''
    Get loader from a dictionary of tensors
    '''
    list_tensor = []
    for d in list_dict:
        list_tensor.append(t.cat(list(d.values()), dim=0))
    
    dataset = TensorDataset(*list_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False) # must be false, don't shuffle

def get_loader(toks: Tensor, batch_size: int) -> DataLoader:
    '''
    Get loader from toks
    '''
    if toks.ndim == 1:
        toks = toks.unsqueeze(0)
    dataset = TensorDataset(toks)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False) # must be false, don't shuffle

@t.no_grad()
def get_cache_with_toks_dataset_on_target_components(
    sae: SAE_Template | SAE_LENS,
    model: HookedTransformer,
    toks: Tensor,
    hook_names: List[str],
    batch_size: int,
    device: str | t.device = "cpu",
    no_verbose: bool = False,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
    '''
    Get the activation cache and toks cache for the list components
    '''
    activation_cache, toks_cache = get_cache_with_toks_dataset(
        model, 
        toks, 
        hook_names,
        batch_size,
        no_verbose,
    )

    indices, values = get_sae_activations(
        sae, 
        get_loader_from_dict([activation_cache], batch_size), 
        device, 
        False, 
        no_verbose,
    )
    return indices, values, toks_cache, activation_cache

@t.no_grad()
def get_cache_with_toks_dataset_on_target_hooks(
    sae: SAE_Template | SAE_LENS,
    model: HookedTransformer,
    toks: Tensor,
    hook_names: List[str],
    batch_size: int,
    device: str | t.device = "cpu",
    no_verbose: bool = False,
) -> Tuple[Tensor, Tensor, Dict[str, Tensor], Dict[str, Tensor]]:
    '''
    Get the activation cache and toks cache for the list hook_names
    '''
    activation_cache, toks_cache = get_cache_with_toks_dataset(
        model, 
        toks, 
        hook_names,
        batch_size,
        no_verbose,
    )

    indices, values = get_sae_activations(
        sae, 
        get_loader_from_dict([activation_cache], batch_size), 
        device, 
        False, 
        no_verbose,
    )
    return indices, values, toks_cache, activation_cache

def rank_top_act(
    sae: SAE_Template | SAE_LENS, 
    values: Tensor, 
    indices: Tensor,
    k: int,
    activation_threshold: float = 1e-3,
) -> Tensor:
    '''
    Rank the features activate the most frequent, given the values and indices
    '''
    count = t.bincount(indices.flatten(), weights=(values > activation_threshold).flatten())
    return t.topk(count, k).indices
  
@t.no_grad()  
def top_act_on_target_components(
    sae: SAE_Template | SAE_LENS,
    model: HookedTransformer,
    toks: Tensor,
    hook_names: List[str],
    k : int,
    batch_size: int,
    device: str,
    activation_threshold: float = 1e-3, 
) -> Tensor:
    '''
    Return the top activation features on the target components
    '''
    indices, values, _, _  = get_cache_with_toks_dataset_on_target_components(
        sae,
        model, 
        toks, 
        hook_names,
        batch_size,
        device,
    )

    return rank_top_act(sae, values, indices, k=k, activation_threshold=activation_threshold)

@t.no_grad()
def find_top_activating_components(
    sae: SAE_Template | SAE_LENS,
    activation_cache: ActivationCache | Dict[str, Tensor],
    batch_size: int,
    device: str,
    filter_threshold: float = 1e-3,
    filter_resid: bool = True,
) -> Dict[int, List[Tuple[str, float]]]:
    '''
    Get the top activating components for all feature id.
    Return: Dict[feature_id, List[(name, score)]], the list is sorted by score
    '''
    num_features = sae.cfg.d_sae
    activated_components = {feat_id : {} for feat_id in range(num_features)}
    for name, data in tqdm(activation_cache.items()):
        if "resid" in name and filter_resid:
            continue
        indices_comp, values_comp = get_sae_activations(
            sae, 
            get_loader(data, batch_size), 
            device, 
            False, 
            no_tqdm=True
        )
        for feat_id in range(num_features):
            score = t.where(
                indices_comp == feat_id, (values_comp > filter_threshold), t.zeros_like(values_comp)
            ).sum().item()
            if score > 0:
                activated_components[feat_id][name] = score
    return {
        feat_id: sorted(activated_components[feat_id].items(), key=lambda x: x[1], reverse=True)
        for feat_id in range(num_features)
    }

def get_k_pos_indices(
    x: Float[Tensor, "batch seq"],
    k: int,
    buffer: int = 0,
    no_overlap: bool = True,
    mode: str = "max",
) -> Int[Tensor, "k 2"]:
    """
    The indices of the top k elements (if mode == "max") or k random non-negative elements (if mode == "rand") in the input tensor, 
    i.e. output[i, :] is the (batch, seqpos) value of the i-th largest element in x.

    Args:
        buffer:     We won't choose any elements within `buffer` from the start or end of their seq (this helps if we
                    want more context around the chosen tokens).
        no_overlap: If True, this ensures that no 2 top-activating tokens are in the same seq and within `buffer` of
                    each other.
    """
    assert buffer * 2 < x.size(1), "Buffer is too large for the sequence length"
    assert not no_overlap or k <= x.size(0), "Not enough sequences to have a different token in each sequence"

    if buffer > 0:
        x = x[:, buffer:-buffer]

    if mode == "max":
        indices = x.flatten().argsort(-1, descending=True)
        rows = indices // x.size(1)
        cols = indices % x.size(1) + buffer
    elif mode == "rand":
         # Get all non-negative indices
        nonneg_mask = x > 0
        indices = nonneg_mask.nonzero(as_tuple=False)
        rows, cols = indices[:, 0], indices[:, 1] + buffer

        # Randomly shuffle the indices
        perm = t.randperm(indices.size(0), device=x.device)
        rows, cols = rows[perm], cols[perm]
    else:
        raise ValueError(f"Unsupported mode: {mode}. Expect \'max\', \'rand\'.")

    if no_overlap:
        unique_indices = t.empty((0, 2), device=x.device).long()
        while len(unique_indices) < k and len(rows) > 0:
            unique_indices = t.cat((unique_indices, t.tensor([[rows[0], cols[0]]], device=x.device)))
            is_overlapping_mask = (rows == rows[0]) & ((cols - cols[0]).abs() <= buffer)
            rows = rows[~is_overlapping_mask]
            cols = cols[~is_overlapping_mask]
        return unique_indices[:k]

    return t.stack((rows, cols), dim=1)[:k]


def display_top_seqs(data: list[tuple[float, list[str], list[str], int, int]]):
    """
    Given a list of (activation: float, str_toks: list[str], seq_pos: int), displays a table of these sequences, with
    the relevant token highlighted.

    We also turn newlines into "\\n", and remove unknown tokens � (usually weird quotation marks) for readability.
    """
    table = Table("Act", "Sequence", "Prompt", title="Max Activating Examples", show_lines=True)
    for act, str_toks, input_toks, seq_pos, input_pos in data:
        formatted_seq = (
            "".join([f"[b u green]{str_tok}[/]" if i == seq_pos else str_tok for i, str_tok in enumerate(str_toks)])
            .replace("�", "")
            .replace("\n", "↵")
        )
        formatted_input = (
            "".join([f"[b u green]{tok}[/]" if i == input_pos else tok for i, tok in enumerate(input_toks)])
            .replace("�", "")
            .replace("\n", "↵")
        )
        table.add_row(f"{act:.3f}", repr(formatted_seq), repr(formatted_input))
    rprint(table)
    
    
def fetch_max_activating_examples(
    sae: SAE_Template | SAE_LENS,
    model: HookedTransformer,
    feat_idx: int,
    values: Tensor,
    indices: Tensor,
    toks: Tensor,
    k: int = 10,
    buffer: int = 10,
    show_input: bool = True,
) -> list[tuple[float, list[str], list[str], int, int]]:
    """
    Returns the max activating examples across a number of batches from the activations store.
    """
    # filter the activations features
    max_acts_per_row = t.max(t.where(indices == feat_idx, values, t.zeros_like(values)), dim=-1).values
    
    # Get largest indices, get the corresponding max acts, and get the surrounding indices
    k_largest_indices = get_k_pos_indices(max_acts_per_row, k=k, buffer=buffer)
    tokens_with_buffer = index_with_buffer(toks, k_largest_indices, buffer=buffer)
    str_toks = [model.to_str_tokens(tok) for tok in tokens_with_buffer]
    str_input = [model.to_str_tokens(toks[row]) if show_input else "" for row, _ in k_largest_indices]
    top_acts = index_with_buffer(max_acts_per_row, k_largest_indices).tolist()
    data = list(zip(top_acts, str_toks, str_input, [buffer] * len(str_toks), [col for _, col in k_largest_indices]))

    return sorted(data, key=lambda x: x[0], reverse=True)[:k] # type: ignore

def sparsity(
    feat_idx: int,
    values: Tensor,
    indices: Tensor,
    threshold: float = 1e-3,
) -> float:
    """
    Returns the sparsity of the activations for a particular feature.
    """
    num_act = ((indices == feat_idx) & (values > threshold)).sum().item()
    batch, seq = indices.shape[:-1]
    return num_act / (batch * seq)

from jaxtyping import Float, Int
import html

# --- Helper Functions ---

def get_k_largest_indices(
    x: t.Tensor, 
    k: int, 
    buffer: int = 0, 
    no_overlap: bool = True
) -> t.Tensor:
    """
    Extracts the top k indices from tensor x.
    
    Improvements:
    1. Does NOT filter out edges (start/end of sequence).
    2. Safely handles overlap masking at edges.
    3. Stops early if values are 0 (prevents picking low/zero activation padding).
    """
    # Work on a copy so we can mask out neighbors without affecting original
    temp_x = x.clone()
    
    indices = []
    batch_size, seq_len = x.shape
    
    for _ in range(k):
        # 1. Find the global maximum in the remaining tensor
        # We flatten to use argmax, then convert back to (row, col)
        flat_idx = temp_x.argmax()
        max_val = temp_x.flatten()[flat_idx].item()
        
        # 2. Stop if the maximum value is 0 (or negative)
        # This prevents returning indices that are just empty padding
        if max_val <= 0:
            break
            
        row = flat_idx // seq_len
        col = flat_idx % seq_len
        
        indices.append([row, col])
        
        # 3. Mask out the selected token and its neighbors
        if no_overlap:
            # Dynamic clamping to prevent index out of bounds errors
            # This allows us to pick index 0 even if buffer is 5
            start = max(0, col - buffer)
            end = min(seq_len, col + buffer + 1)
            temp_x[row, start:end] = 0.0
        else:
            temp_x[row, col] = 0.0
            
    if len(indices) == 0:
        return t.empty((0, 2), dtype=t.long)
        
    return t.tensor(indices, dtype=t.long)


def index_with_buffer(
    x: Float[Tensor, "batch seq"], indices: Int[Tensor, "k 2"], buffer: int | None = None
) -> Float[Tensor, "k *buffer_x2_plus1"]:
    if indices.shape[0] == 0:
        return t.tensor([], device=x.device)

    rows, cols = indices.unbind(dim=-1)
    if buffer is not None:
        cols = cols.clone()
        cols = t.clamp(cols, min=buffer, max=x.size(1) - buffer - 1)
        offsets = t.arange(-buffer, buffer + 1, device=x.device)
        rows_expanded = einops.repeat(rows, "k -> k w", w=len(offsets))
        cols_expanded = einops.repeat(cols, "k -> k w", w=len(offsets)) + offsets
        return x[rows_expanded, cols_expanded]
        
    return x[rows, cols]

# --- Updated Data Extraction & Visualization ---

def fetch_feature_activation_intervals(
    model: Any, 
    feat_idx: int,
    values: t.Tensor, 
    indices: t.Tensor, 
    toks: t.Tensor, 
    n_intervals: int = 5,
    k_per_interval: int = 5,
    buffer: int = 5, 
) -> Dict[str, Any]:
    """
    Extracts examples for specific feature activation intervals.
    """
    
    # 1. Reconstruct dense activations
    feat_acts = t.zeros_like(toks, dtype=t.float)
    mask = (indices == feat_idx)
    total_tokens = feat_acts.numel()
    total_density = mask.sum().item() / total_tokens
    feat_acts = (values * mask.float()).sum(dim=-1)
    
    max_act = feat_acts.max().item()
    
    # Debug output
    if max_act <= 0:
        return {"feature_id": feat_idx, "intervals": [], "max_act": 0, "density": total_density}

    boundaries = t.linspace(1e-3, max_act, n_intervals + 1)
    interval_data = []
    seq_len = toks.shape[1]

    for i in range(n_intervals, 0, -1):
        lower_bound = boundaries[i-1].item()
        upper_bound = boundaries[i].item()
        
        # Inclusive upper bound for the top interval
        if i == n_intervals:
            interval_mask = (feat_acts >= lower_bound) & (feat_acts <= upper_bound + 1e-4)
        else:
            interval_mask = (feat_acts >= lower_bound) & (feat_acts < upper_bound)
            
        count = interval_mask.sum().item()
        density = count / total_tokens
        
        if count == 0:
            continue

        # Apply mask to activations so get_k_largest only sees valid interval data
        masked_acts = feat_acts * interval_mask.float()
        
        # Call the FIXED selector
        # We pass the buffer here so it knows how wide to mask neighbors,
        # but the function now safely handles edges.
        selected_indices = get_k_largest_indices(masked_acts, k=k_per_interval, buffer=buffer, no_overlap=True)
        
        if selected_indices.shape[0] == 0:
            continue
        
        examples = []
        for j in range(selected_indices.shape[0]):
            row_idx = selected_indices[j, 0].item()
            col_idx = selected_indices[j, 1].item()
            
            actual_val = feat_acts[row_idx, col_idx].item() # type: ignore

            # --- Dynamic Window Slicing ---
            # Calculate start/end based on buffer, clamped to sequence limits.
            # This removes the need for padding tokens.
            start_idx = max(0, col_idx - buffer)
            end_idx = min(seq_len, col_idx + buffer + 1)
            
            # 1. Short View (Windowed)
            raw_ids_short = toks[row_idx, start_idx:end_idx].tolist() # type: ignore
            str_tokens_short = model.to_str_tokens(t.tensor(raw_ids_short))
            acts_short = feat_acts[row_idx, start_idx:end_idx].tolist() # type: ignore
            
            # Calculate target index relative to the short slice
            # e.g., if slice starts at 5 and target is at 7, relative is 2
            relative_target_idx = col_idx - start_idx
            
            # 2. Full View
            raw_ids_full = toks[row_idx].tolist() # type: ignore
            acts_full = feat_acts[row_idx].tolist() # type: ignore
            str_tokens_full = model.to_str_tokens(t.tensor(raw_ids_full))
            
            examples.append({
                "short": {
                    "tokens": str_tokens_short,
                    "token_ids": raw_ids_short,
                    "activations": acts_short,
                    "target_idx": relative_target_idx
                },
                "full": {
                    "tokens": str_tokens_full,
                    "token_ids": raw_ids_full,
                    "activations": acts_full,
                    "target_idx": col_idx
                },
                "target_act": actual_val,
            })

        if len(examples) > 0:
            interval_data.append({
                "min": lower_bound,
                "max": upper_bound,
                "density": density,
                "examples": examples
            })

    return {
        "feature_id": feat_idx,
        "max_act": max_act,
        "intervals": interval_data,
        "density": total_density
    }

def _render_token_html(tokens: List[str], acts: List[float], target_idx: int, max_global: float) -> str:
    """
    Helper to generate the HTML string for a sequence of tokens with green highlighting.
    """
    html_str = ""
    for i, (tok, act) in enumerate(zip(tokens, acts)):
        # 1. Calculate opacity relative to global max for consistent coloring across examples
        # Using a slight non-linear scaling (sqrt) can sometimes help see lower values, 
        # but linear is standard. We'll stick to linear here.
        ratio = act / max_global if max_global > 0 else 0
        alpha = max(0.0, min(1.0, ratio))
        
        # 2. Use RGBA for the Green highlight (Matches the original style)
        # rgba(0, 200, 83) is a vibrant green. 
        # We add a base opacity of 0.1 for non-zero items so they are visible, 
        # or just pure alpha. Let's use pure alpha for accuracy.
        bg_color = f"rgba(0, 200, 83, {alpha})" if act > 0 else "transparent"
        
        # Escape HTML characters (like <, >) and replace newlines with a visual symbol
        safe_tok = html.escape(tok).replace('\n', '↵')
        
        # 3. Target highlighting (The specific token the SAE looked at)
        is_target = (i == target_idx)
        target_class = "target-token" if is_target else ""
        
        # 4. Construct the span with a custom tooltip
        # REMOVE THE NEWLINE in the f-string - it's causing each token to be on a new line
        html_str += f'<span class="token {target_class}" style="background-color: {bg_color};">{safe_tok}<span class="tooltip">{act:.4f}</span></span>'
        
    return html_str

def generate_interactive_html(data: Dict[str, Any]) -> str:
    css = """
    <style>
        body { 
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
            background-color: #f8f9fa; 
            color: #333; 
            padding: 20px; 
        }
        .container { 
            max-width: 1100px; 
            margin: 0 auto; 
            background: white; 
            border-radius: 8px; 
            box-shadow: 0 2px 10px rgba(0,0,0,0.05); 
            overflow: hidden; 
        }
        .header { 
            background: #fff; 
            padding: 15px 20px; 
            border-bottom: 1px solid #eee; 
        }
        .header h2 { margin: 0; font-size: 1.1rem; color: #444; }
        
        .interval-block { border-bottom: 1px solid #eee; }
        .interval-header { 
            background: #f1f3f5; 
            padding: 5px 20px; 
            font-size: 0.75rem; 
            font-weight: bold; 
            color: #555; 
            text-transform: uppercase; 
            letter-spacing: 0.5px;
            display: flex;
            justify-content: space-between;
        }
        
        /* Compact Row Layout */
        .example-row { 
            display: flex; 
            align-items: flex-start; 
            padding: 6px 15px; /* Reduced whitespace */
            border-bottom: 1px solid #f9f9f9; 
        }
        .example-row:last-child { border-bottom: none; }
        
        .meta-col { 
            width: 60px; 
            flex-shrink: 0; 
            display: flex; 
            flex-direction: column; 
            align-items: center; 
            margin-right: 15px;
            padding-top: 2px;
        }
        .act-badge { 
            background: #e9ecef; 
            color: #495057; 
            padding: 1px 5px; 
            border-radius: 3px; 
            font-size: 0.65rem; 
            font-weight: bold;
            margin-bottom: 2px;
            text-transform: uppercase;
        }
        .act-val { color: #28a745; font-weight: bold; font-size: 0.85rem; }
        
        .seq-col { 
            flex-grow: 1; 
            font-family: 'Consolas', 'Monaco', monospace; 
            font-size: 13px;
            line-height: 1.1;
            white-space: pre-wrap; 
            cursor: pointer; 
        }
        
        /* Token Styling */
        .token { 
            display: inline; 
            padding: 0 1px;      
            margin: 0;
            border-radius: 2px;
            cursor: help; 
            position: relative; /* Needed for tooltip positioning */
            transition: outline 0.1s;
        }
        
        /* Hover Effect: Outline + Tooltip Visibility */
        .token:hover {
            outline: 1px solid #444;
            z-index: 10;
        }
        
        .target-token { 
            border-bottom: 2px solid #333; 
            font-weight: bold;
        }
        
        /* Custom CSS Tooltip */
        .token .tooltip {
            visibility: hidden;
            background-color: #333;
            color: #fff;
            text-align: center;
            padding: 4px 8px;
            border-radius: 4px;
            position: absolute;
            z-index: 100;
            bottom: 120%; /* Position above the token */
            left: 50%;
            transform: translateX(-50%);
            font-family: sans-serif;
            font-size: 11px;
            font-weight: normal;
            white-space: nowrap;
            pointer-events: none;
            opacity: 0;
            transition: opacity 0.2s;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        
        /* Tiny arrow for the tooltip */
        .token .tooltip::after {
            content: "";
            position: absolute;
            top: 100%;
            left: 50%;
            margin-left: -5px;
            border-width: 5px;
            border-style: solid;
            border-color: #333 transparent transparent transparent;
        }

        .token:hover .tooltip {
            visibility: visible;
            opacity: 1;
        }
        
        .seq-col:hover {
            background-color: #fafafa; /* Subtle hint that row is clickable */
        }

    </style>
    <script>
        function toggleSeq(element) {
            const shortView = element.querySelector('.short-view');
            const fullView = element.querySelector('.full-view');
            
            if (shortView.style.display !== 'none') {
                shortView.style.display = 'none';
                fullView.style.display = 'block';
            } else {
                shortView.style.display = 'block';
                fullView.style.display = 'none';
            }
        }
    </script>
    """
    
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>SAE Feature {data['feature_id']}</title>
        {css}
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h2>Feature {data['feature_id']} (Max Act: {data['max_act']:.4f})</h2>
            </div>
    """
    
    max_global = data['max_act']
    
    for interval in data['intervals']:
        density_pct = interval['density'] * 100
        
        html_content += f"""
            <div class="interval-block">
                <div class="interval-header">
                    <span>Interval {interval['min']:.3f} - {interval['max']:.3f}</span>
                    <span>Density: {density_pct:.4f}%</span>
                </div>
        """
        
        for ex in interval['examples']:
            short_html = _render_token_html(
                ex['short']['tokens'], 
                ex['short']['activations'], 
                ex['short']['target_idx'], 
                max_global
            )
            
            full_html = _render_token_html(
                ex['full']['tokens'], 
                ex['full']['activations'], 
                ex['full']['target_idx'], 
                max_global
            )
            
            html_content += f"""
                <div class="example-row">
                    <div class="meta-col">
                        <span class="act-badge">Max</span>
                        <span class="act-val">{ex['target_act']:.2f}</span>
                    </div>
                    <div class="seq-col" onclick="toggleSeq(this)">
                        <div class="short-view">{short_html}</div>
                        <div class="full-view" style="display:none;">{full_html}</div>
                    </div>
                </div>
            """
            
        html_content += "</div>"
        
    html_content += """
        </div>
    </body>
    </html>
    """
    
    return html_content


def extract_data(
    dataloader: DataLoader[tuple[t.Tensor]], 
    num_batches: int = 200,
) -> Tensor:
    '''
    Get the tokens from dataset up to given number of batches
    '''
    data = []
    for i, batch in enumerate(dataloader):
        if i >= num_batches:
            break
        data.append(batch[0])
    return t.cat(data, dim=0)


def show_top_logits(
    model: HookedTransformer,
    sae: SAE_Template | SAE_LENS,
    latent_idx: int,
    k: int = 10,
) -> None:
    """
    Displays the top & bottom logits for a particular latent.
    """
    if isinstance(sae, SAE_LENS):
        logits = model.ln_final(sae.W_dec[latent_idx]) @ model.W_U
    else:
        logits = model.ln_final(sae.decoder.weight.data.T[latent_idx]) @ model.W_U

    pos_logits, pos_token_ids = logits.topk(k)
    pos_tokens = model.to_str_tokens(pos_token_ids)
    neg_logits, neg_token_ids = logits.topk(k, largest=False)
    neg_tokens = model.to_str_tokens(neg_token_ids)

    print(
        tabulate(
            zip(map(repr, neg_tokens), neg_logits, map(repr, pos_tokens), pos_logits),
            headers=["Bottom tokens", "Value", "Top tokens", "Value"],
            tablefmt="simple_outline",
            stralign="right",
            numalign="left",
            floatfmt="+.3f",
        )
    )
    
def create_prompt(
    sae: SAE_Template | SAE_LENS,
    model: HookedTransformer,
    latent_idx: int,
    values: Tensor,
    indices: Tensor,
    toks: Tensor,
    k: int = 15,
    buffer: int = 10,
) -> dict[str, str]:
    """
    Returns the system, user & assistant prompts for autointerp.
    """
    data = fetch_max_activating_examples(sae, model, latent_idx, values, indices, toks, k, buffer)
    str_formatted_examples = "\n".join(
        f"{i+1}. {''.join(f'<<{tok}>>' if j == buffer else tok for j, tok in enumerate(seq[1]))}"
        for i, seq in enumerate(data)
    )

    return {
        "system": """We're studying neurons in a neural network. Each neuron activates on some particular word or concept in a short document. The activating words in each document are indicated with << ... >>. Look at the parts of the document the neuron activates for and summarize in a single sentence what the neuron is activating on. Try to be specific in your explanations, although don't be so specific that you exclude some of the examples from matching your explanation. Pay attention to things like the capitalization and punctuation of the activating words or concepts, if that seems relevant. Keep the explanation as short and simple as possible, limited to 20 words or less. Omit punctuation and formatting. You should avoid giving long lists of words.""",
        "user": f"""The activating documents are given below:\n\n{str_formatted_examples}""",
        "assistant": "this neuron fires on",
    }
    

from openai import OpenAI
from dotenv import load_dotenv
import os

def get_autointerp_explanation(
    sae: SAE_Template | SAE_LENS,
    model: HookedTransformer,
    latent_idx: int,
    values: Tensor,
    indices: Tensor,
    toks: Tensor,
    k: int = 15,
    buffer: int = 10,
    n_completions: int = 1,
) -> list[str]:
    """
    Queries OpenAI's API using prompts returned from `create_prompt`, and returns
    a list of the completions.
    """
    client = OpenAI(api_key=API_KEY)

    prompts = create_prompt(sae, model, latent_idx, values, indices, toks, k, buffer)

    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": prompts["system"]},
            {"role": "user", "content": prompts["user"]},
            {"role": "assistant", "content": prompts["assistant"]},
        ],
        n=n_completions,
        max_tokens=50,
        stream=False,
    )
    return [choice.message.content for choice in result.choices] # type: ignore

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY", None)