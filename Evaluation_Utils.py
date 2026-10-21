from tabnanny import check
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Dict, Any, Union, Callable, Literal
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint
from transformer_lens import (
    utils,
)
from Utils import Pruner
from sae.SAE_Tools import *
import random as rd
from contextlib import contextmanager
import sys

from huggingface_hub import login
import os
from dotenv import load_dotenv

from graph.Feature_Graph import Feature_Graph, ConnectionNode, ConnectionIndex, sae_hook_name, error_term_name, output_hook_name, input_hook_name, FeatureIndex, ErrorIndex, return_idx, FeatureErrorIndex
from graph.Feature_Graph_Trans import *
from graph.Feature_Graph_Cross import *
from sae.autoencoder.SAE_Wrapper import HookedSAE
from sae_lens import HookedSAETransformer, SAE as SAE_LENS
from sae.autoencoder import SAE_Template
from random import randint
from collections.abc import Generator, Sequence
from typing import TypeVar, overload
import copy

from graph_visualizer import SAEGraphVisualizer

load_dotenv()   
hf_key = os.getenv('HUGGING_FACE_API_KEY')
login(hf_key)

def load_graph(
    model: HookedSAETransformer,
    dict_saes: Dict[int, List[Tuple[str, Any]]],
    method_name: Literal["fc", "trans", "cross", "ml"],
    use_error_term: bool,
) -> Feature_Graph:
    if method_name == "fc":
        fg = Feature_Graph(
            model, 
            dict_saes, 
            use_error_term=use_error_term,
        )
    else:
        raise
    return fg

def load_saes_(
    model_name: str,
    releases: List[str | Literal["local"]],
    sae_ids: List[str | Literal["hook_name"]],
    hook_pos: List[str],
    device: str | t.device = "cpu",
    layer_range: List[int] | None = None,
    dtype: t.dtype | str = t.float32,
) -> Tuple[
    HookedSAETransformer,
    Dict[int, List[Tuple[str, HookedSAE]]],
    List[HookedSAE],
]:
    model: HookedSAETransformer = HookedSAETransformer.from_pretrained_no_processing(
        model_name, device=device, dtype=dtype # type: ignore
    )

    if layer_range == None:
        layer_range = [layer for layer in range(model.cfg.n_layers)]
        
    dict_saes = {}
    list_saes = []
    
    for release, sae_id in zip(releases, sae_ids):
        if release == "local_multi":
            saes = load_multi_sae_model(
                sae_id,
                model,
                device=str(device),
            )
            list_saes += [sae.to(model.cfg.dtype) for sae in saes]
        
    for layer in tqdm(layer_range):
        
        layer_saes = []
        for id, release, sae_id in zip(range(len(releases)), releases, sae_ids):
            hook_name = utils.get_act_name(hook_pos[id], layer)
            
            if release == "local":
                hooked_sae = load_sae_model(
                    sae_id.format(layer=layer),
                    model,
                    device=str(device),
                    training=False,
                ).to(model.cfg.dtype)
                layer_saes.append((hooked_sae.cfg.hook_name, hooked_sae))
                list_saes.append(hooked_sae)
                
            elif release == "local_multi":
                target_sae = list_saes[0] # placeholder
                for sae in list_saes:
                    if hook_name == sae.cfg.hook_name:
                        target_sae = sae
                        break
                    
                # remove redundant saes with the same hook_name
                has_hook = False
                for hook, sae in layer_saes:
                    if sae.cfg.hook_name == target_sae.cfg.hook_name:
                        has_hook = True
                        break
                if has_hook:
                    continue
                    
                layer_saes.append((hook_name, target_sae))
                    
            else:
                hooked_sae, _, _ = HookedSAE.from_pretrained(
                    release=release,
                    sae_id=hook_name if sae_id == "hook_name" else sae_id.format(layer=layer),
                    device=str(device),
                )
                hooked_sae = hooked_sae.to(model.cfg.dtype) # type: ignore
                layer_saes.append((hook_name, hooked_sae))
                list_saes.append(hooked_sae)
                
        dict_saes[layer] = layer_saes
        
    return model, dict_saes, list_saes

def load_saes(
    model_name: str,
    sae_types: List[str],
    device: str | t.device = "cpu",
    dtype: t.dtype | str = t.float32,
    layer_range: List[int] | None = None,
) -> Tuple[
    HookedSAETransformer,
    Dict[int, List[Tuple[str, HookedSAE]]],
    List[HookedSAE],
]:
    if model_name == "meta-llama/Llama-3.1-8B":
        releases = []
        hook_pos = []
        sae_ids = []
        for sae_type in sae_types:
            if sae_type == "post":
                release = "llama_scope_lxr_8x"
                pos = "resid_post"
                sae_id = "l{layer}r_8x"
                
            elif sae_type == "attn":
                release = "llama_scope_lxa_8x"
                pos = "attn_out"
                sae_id = "l{layer}a_8x"
                
            elif sae_type == "mlp":
                release = "llama_scope_lxm_8x"
                pos = "mlp_out"
                sae_id = "l{layer}m_8x"
                
            else:
                raise ValueError(f"Unknown SAE type: {sae_type}")
            
            releases.append(release)
            hook_pos.append(pos)
            sae_ids.append(sae_id)
            
    if model_name == "meta-llama/Llama-3.2-1B":
        releases = []
        hook_pos = []
        sae_ids = []
        for sae_type in sae_types:
            if sae_type == "post":
                release = "local"
                pos = "resid_post"
                sae_id = "data/checkpoints/topk_16.0_32_blocks.{layer}.hook_resid_post_meta-llama_Llama-3.2-1b_0.0001_1024_0.03125_42.ckpt"
                
            else:
                raise ValueError(f"Unknown SAE type: {sae_type}")
            
            releases.append(release)
            hook_pos.append(pos)
            sae_ids.append(sae_id)
            
    elif model_name == "qwen3-1.7b":
        releases = []
        hook_pos = []
        sae_ids = []
        for sae_type in sae_types:
            if sae_type == "post":
                release = "local"
                pos = "resid_post"
                sae_id = ""
                
            elif sae_type == "mlp":
                release = "local_multi"
                pos = "mlp_out"
                sae_id_1 = "data/checkpoints/topk_32.0_128_0_1_2_3_4_5_6_7_8_9_10_11_12_13_hook_mlp_out_qwen3-1.7b_0.0001_1024_0.03125_42.ckpt"
                sae_id_2 = "data/checkpoints/topk_32.0_128_14_15_16_17_18_19_20_21_22_23_24_25_26_27_hook_mlp_out_qwen3-1.7b_0.0001_1024_0.03125_42.ckpt"
                releases.extend([release, release])
                hook_pos.extend([pos, pos])
                sae_ids.extend([sae_id_1, sae_id_2])
                continue
            
            else:
                raise ValueError(f"Unknown SAE type: {sae_type}")
            
            releases.append(release)
            hook_pos.append(pos)
            sae_ids.append(sae_id)
            
    elif model_name == "gpt2-small":
        releases = []
        hook_pos = []
        sae_ids = []
        for sae_type in sae_types:
            if sae_type == "ds":
                # release = "gpt2-small-res-jb"
                release = "local"
                pos = "resid_pre"
                # sae_id = "blocks.{layer}.hook_resid_pre"
                # Our TopK SAEs (Appendix D.4). Override the location with GPT2_SAE_TEMPLATE.
                sae_id = os.getenv("GPT2_SAE_TEMPLATE", "data/topk_custom_32.0_32_0.1_blocks.{layer}.hook_resid_pre_0.0001_256_0.03125_42.ckpt")
                
            elif sae_type == "mlp":
                # release = "gpt2-small-mlp-out-v5-32k"
                release = "gpt2-small-mlp-tm"
                pos = "mlp_out"
                sae_id = "blocks.{layer}.hook_mlp_out"
                
            elif sae_type == "trans_mlp":
                release = "local"
                pos = "mlp_out"
                sae_id = "checkpoints/topk_transcoder_32.0_32_blocks.{layer}.hook_mlp_out_0.0001_256_0.03125_42.ckpt"
                
            else:
                raise
            
            releases.append(release)
            hook_pos.append(pos)
            sae_ids.append(sae_id)
            
    elif model_name == "google/gemma-2-9b":
        releases = []
        hook_pos = []
        sae_ids = []
        for sae_type in sae_types:
            if sae_type == "ds":
                release = "gemma-scope-9b-pt-res-canonical"
                pos = "resid_post"
                sae_id = "layer_{layer}/width_16k/canonical"
                
            elif sae_type == "mlp":
                release = "gemma-scope-9b-pt-mlp-canonical"
                pos = "mlp_out"
                sae_id = "layer_{layer}/width_16k/canonical"
                
            else:
                raise
            
            releases.append(release)
            hook_pos.append(pos)
            sae_ids.append(sae_id)
            
    elif model_name == "google/gemma-2-2b":
        releases = []
        hook_pos = []
        sae_ids = []
        for sae_type in sae_types:
            if sae_type == "ds":
                release = "gemma-scope-2b-pt-res-canonical"
                pos = "resid_post"
                sae_id = "layer_{layer}/width_16k/canonical"
                
            elif sae_type == "mlp":
                release = "gemma-scope-2b-pt-mlp-canonical"
                pos = "mlp_out"
                sae_id = "layer_{layer}/width_16k/canonical"
                
            else:
                raise
            
            releases.append(release)
            hook_pos.append(pos)
            sae_ids.append(sae_id)
    else:
        raise ValueError(f"Unknown model name: {model_name}")
    
    model, dict_saes, list_saes= load_saes_(
        model_name=model_name,
        releases=releases,
        sae_ids=sae_ids,
        hook_pos=hook_pos,
        device=device,
        layer_range=layer_range,
        dtype=dtype,
    )
    return model, dict_saes, list_saes 


def active_nodes(
    graph: Feature_Graph,
    interpretable_node: bool = False,
) -> Tuple[int, int]:
    num_active = 0
    num_inactive = 0
    def active_nodes(sae_name, node_idx, reverse) -> int:
        if graph.token_wise:
            node = graph.nodes[(sae_name, node_idx)] # act: (seq, d_sae), resc: (seq) 
        else:
            node = graph.nodes[(sae_name, node_idx)].mean(dim=0) # act: (d_sae), resc: () - COLAPSED TENSOR
            
        if reverse:
            active_node = (node.act == 0).sum().item()
            active_error = 0
            if node.resc is not None and not interpretable_node:
                active_error += (node.resc == 0).sum().item()
            if node.res is not None and not interpretable_node:
                active_error += (node.res == 0).sum().item()
        else:
            active_node = (node.act > 0).sum().item()
            active_error = 0
            if node.resc is not None and not interpretable_node:
                active_error += (node.resc > 0).sum().item()
            if node.res is not None and not interpretable_node:
                active_error += (node.res > 0).sum().item() 
            
        return active_node + active_error # type: ignore
    
    for sae_name, node_idx in graph.nodes.keys():
        num_active += active_nodes(sae_name, node_idx, False)
        num_inactive += active_nodes(sae_name, node_idx, True)
    
    return num_active, num_inactive


def percen_active_nodes(graph: Feature_Graph, d_model: int) -> float:
    num_active, num_inactive = active_nodes(graph)
    return max(num_active, 1) * 100 / (num_active + num_inactive)


def kl_div_from_logits(logits: t.Tensor, target: t.Tensor, reduction: str | None = "mean") -> t.Tensor:
    """
    Calculate the KL divergence between two distributions represented by logits.
    
    Args:
        logits: The predicted logits
        target: The target logits
        reduction: How to reduce the result, either "mean", "sum", or None for no reduction
        
    Returns:
        The KL divergence tensor
    """
    # Convert logits to probabilities using softmax
    probs1 = t.nn.functional.softmax(target, dim=-1)
    probs2 = t.nn.functional.softmax(logits[:, -1], dim=-1)
    kl_div = probs1 * (t.log(probs1 + 1e-10) - t.log(probs2 + 1e-10))
    kl_div = kl_div.sum(-1)
    if reduction == "mean":
        return kl_div.mean()
    elif reduction == "sum":
        return kl_div.sum()
    return kl_div

def revise_index(index: Index) -> Index:
    if isinstance(index, FeatureIndex):
        return FeatureIndex([index.list_index[-1]]) # type: ignore
    elif isinstance(index, ErrorIndex):
        return ErrorIndex([index.list_index[-1]]) # type: ignore
    elif isinstance(index, FeatureErrorIndex):
        return FeatureErrorIndex([index.list_index[-1]]) # type: ignore
    else:
        raise ValueError(f"Unknown index type: {type(index)}")

def cache_activation_on_general_dataset(
    graph: Feature_Graph,
    dict_all_list: Dict[Tuple[Node, Index], List[Tuple[Node, Index, float]]],
    tok_name: str,
    num_batches: int,
    batch_size: int,
    device: str | t.device = "cpu",
    max_length: int = 128,
    data_path: str = "monology/pile-uncopyrighted",
):
    from sae.SAE_Trainer import DataConfig
    from sae.Load_Data import load_data
    
    data_config = DataConfig(
        data_path=data_path,
        batch_size=1,
        max_length=max_length,
    )
    
    max_retries = 10
    for attempt in range(max_retries):
        try:
            _, data_loader = load_data(tok_name, data_config)
            break
        except Exception as e:
            pass
        
    data_toks = extract_data(data_loader, num_batches=num_batches) # type: ignore
    
    loader = get_loader(
        data_toks,
        batch_size=batch_size,
    )
    
    def revise_dict_all_list(dict_all_list):
        new_dict = {}
        for (end_node, end_index), values in dict_all_list.items():
            end_index = revise_index(end_index)
            if (end_node, end_index) not in new_dict:
                new_dict[(end_node, end_index)] = []
            for (start_node, start_index, value) in values:
                start_index = revise_index(start_index)
                if (start_node, start_index) not in new_dict:
                    new_dict[(start_node, start_index)] = []
        return new_dict
    
    new_dict_all_list = revise_dict_all_list(dict_all_list)
    
    all_cache = {}
    for batch in tqdm(loader):
        batch = batch[0].to(device)
        with t.no_grad():
            _, cache = graph.run_model(batch, use_error_term=True)
            
            # for name, tens in cache.items():
            #     if name not in all_cache:
            #         all_cache[name] = []
            #     all_cache[name].append(tens.detach().cpu())
        
        for node, index in new_dict_all_list.keys():
            if isinstance(index, FeatureIndex):
                if (node, index) not in all_cache:
                    all_cache[(node, index)] = []
                    
                all_cache[(node, index)].append(
                    cache[sae_hook_name(node.name)][:, :, index.list_index[-1]].flatten().detach().cpu()
                )
            elif isinstance(index, FeatureErrorIndex):
                if (node, index) not in all_cache:
                    all_cache[(node, index)] = []
                    
                all_cache[(node, index)].append(
                    cache[sae_hook_name(error_term_name(node.name))][:, :, index.list_index[-1]].flatten().detach().cpu()
                )
    
    for key in all_cache.keys():
        all_cache[key] = t.cat(all_cache[key], dim=0)
        
    return all_cache

def to_binary(x):
    return (x > 0).int()

T = TypeVar("T")
K = TypeVar("K")

@overload
def batchify(
    data: Sequence[T], batch_size: int, show_progress: bool = False
) -> Generator[Sequence[T], None, None]: ...


@overload
def batchify(
    data: t.Tensor, batch_size: int, show_progress: bool = False
) -> Generator[t.Tensor, None, None]: ...


def batchify(
    data: Sequence[T] | t.Tensor, batch_size: int, show_progress: bool = False
) -> Generator[Sequence[T] | t.Tensor, None, None]:
    """Generate batches from data. If show_progress is True, display a progress bar."""

    for i in tqdm(
        range(0, len(data), batch_size),
        total=(len(data) // batch_size + (len(data) % batch_size != 0)),
        disable=not show_progress,
    ):
        yield data[i : i + batch_size]
        
@contextmanager
def suppress_print(suppress: bool):
    with open(os.devnull, "w") as fnull:
        original_stdout = sys.stdout
        if suppress:
            sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = original_stdout