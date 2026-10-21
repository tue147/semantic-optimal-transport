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
    try:
        for sae in list_saes:
            model.add_sae(sae, use_error_term)
        yield
    finally:
        model.reset_saes()
        model.reset_hooks()