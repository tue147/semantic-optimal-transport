import torch as t
from typing import Any, Dict, List

@t.no_grad()
def match_intra_sae_features_mech_permute(
    sae: Any,
    feature_pool: t.Tensor,
    k: int = 5
) -> Dict[int, List[Dict[str, float]]]:
    """Matches features based on the L2 distance of their decoder weights."""
    
    W_dec = sae.W_dec[feature_pool] # (N, d_model)
    
    # Compute pairwise L2 distance matrix
    dist_matrix = t.cdist(W_dec, W_dec, p=2.0)
    
    results = {}
    for i, feat_idx in enumerate(feature_pool.tolist()):
        dists = dist_matrix[i]
        
        # lowest distance first (largest=False)
        topk_vals, topk_inds = t.topk(dists, k + 1, largest=False)
        
        feat_results = []
        for val, ind in zip(topk_vals, topk_inds):
            if ind.item() == i: 
                continue
            feat_results.append({
                'feature': feature_pool[ind].item(),
                'l2_distance': val.item()
            })
            
        results[feat_idx] = feat_results[:k]
        
    return results