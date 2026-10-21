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
from .autoencoder import TopKSAE, SAEOut, SAE_Template
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
    
    if sae_name not in ("topk", "topk_custom"):
        raise ValueError(f"Unknown SAE name: {hyperparameters['sae_name']}")

    sae = TopKSAE(
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


from openai import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("OPENAI_API_KEY", None)