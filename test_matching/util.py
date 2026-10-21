import torch as t
from typing import Dict, List, Tuple, Any
from functools import partial
from contextlib import contextmanager
from sae.SAE_Tools import *
from sae.autoencoder.SAE_Wrapper import HookedSAE
from graph.Feature_Graph import *

def extract_pair(key: int, final_alignments: Dict[int, Dict[str, Any] | List[Any]], top: int = 1):
    assert key in final_alignments
    info = final_alignments[key]
    feat_pairs = []
    
    if isinstance(info, List):
        for dict in info[:min(len(info), top)]:
            feat_pairs.append((key, dict["target_feature"]))
    elif isinstance(info, Dict):
        if "top_k_targets" in info:
            for dict in info["top_k_targets"][:min(len(info["top_k_targets"]), top)]:
                feat_pairs.append((key, dict["target_feature"]))
        elif "target_feature" in info:
            feat_pairs.append((key, info["target_feature"]))
        
    return feat_pairs

@contextmanager
def setup_forward_model_hook(
    model: HookedSAETransformer, list_saes: List[HookedSAE], use_error_term: bool
):
    input_hooks_trans = []
    output_hooks_trans = []
    transcoders = []
    input_hooks_cross = []
    output_hooks_cross = []
    crosscoders = []
    non_cross_or_trans = []
    for sae in list_saes:
        if getattr(sae, "input_hook", False) and getattr(sae, "output_hook", False):
            input_hooks_trans.append(sae.input_hook)
            output_hooks_trans.append(sae.output_hook)
            transcoders.append(sae)
        elif getattr(sae, 'input_hook', False) and getattr(sae, 'output_hooks', False):
            input_hooks_cross.append(sae.input_hook)
            output_hooks_cross.append(sae.output_hooks[0]) # type: ignore
            crosscoders.append(sae)
        else:
            non_cross_or_trans.append(sae)
            
    # Transcoder
    trans_cache = {}
    # Hook function at transcoder input: caches the activations before transcoder
    def hook_transcoder_input(activations: Tensor, hook: HookPoint, transcoder_idx: int):
        trans_cache[transcoder_idx] = activations.clone()

    # Hook function at transcoder output: replaces activations with transcoder output
    def hook_transcoder_output(activations: Tensor, hook: HookPoint, transcoder_idx: int):
        trans_input = trans_cache[transcoder_idx]
        return transcoders[transcoder_idx]((trans_input, activations))
    
    # Crosscoder
    cross_cache = {}
    # Hook function at transcoder input: caches the activations before transcoder
    def hook_crosscoder_input(activations: Tensor, hook: HookPoint, crosscoder_idx: int):
        cross_cache[crosscoder_idx] = activations.clone()
    
    all_cross_recons = [0.0 for _ in range(len(crosscoders))]
    def hook_crosscoder_output(activations: Tensor, hook: HookPoint, crosscoder_idx: int):
        recons, cross_recons = crosscoders[crosscoder_idx].forward_crosscoder(
            (cross_cache[crosscoder_idx], activations, all_cross_recons[crosscoder_idx])
        )
        for j in range(crosscoder_idx+1, model.cfg.n_layers):
            all_cross_recons[j] += cross_recons[j-crosscoder_idx-1]
        
        return recons
    
    fwd_hooks = []
    for i in range(len(transcoders)):
        fwd_hooks.append((input_hooks_trans[i], partial(hook_transcoder_input, transcoder_idx=i)))
        fwd_hooks.append((output_hooks_trans[i], partial(hook_transcoder_output, transcoder_idx=i)))
    for i in range(len(crosscoders)):
        fwd_hooks.append((input_hooks_cross[i], partial(hook_crosscoder_input, crosscoder_idx=i)))
        fwd_hooks.append((output_hooks_cross[i], partial(hook_crosscoder_output, crosscoder_idx=i)))
        
    try: 
        for sae in non_cross_or_trans:
            model.add_sae(sae, use_error_term)
        for hook, func in fwd_hooks:
            model.add_hook(hook, func, dir="fwd")
        
        yield
    finally:
        model.reset_saes()
        model.reset_hooks()