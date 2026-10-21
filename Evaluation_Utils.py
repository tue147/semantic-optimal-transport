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
    method_name: Literal["fc"],
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
    releases: List[str],
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
    # GPT2: our TopK SAEs on the residual stream (Appendix D.4).
    # Gemma-2-2B: Gemma Scope JumpReLU SAEs on the MLP outputs.
    releases = []
    hook_pos = []
    sae_ids = []
    if model_name == "gpt2-small" and sae_types == ["ds"]:
        releases.append("local")
        hook_pos.append("resid_pre")
        # Override the checkpoint location with GPT2_SAE_TEMPLATE.
        sae_ids.append(os.getenv("GPT2_SAE_TEMPLATE", "data/checkpoints/topk_32.0_32_blocks.{layer}.hook_resid_pre_gpt2-small_0.0001_256_0.03125_42.ckpt"))
    elif model_name == "google/gemma-2-2b" and sae_types == ["mlp"]:
        releases.append("gemma-scope-2b-pt-mlp-canonical")
        hook_pos.append("mlp_out")
        sae_ids.append("layer_{layer}/width_16k/canonical")
    else:
        raise ValueError(f"Unsupported model / SAE type: {model_name} {sae_types}")
    
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