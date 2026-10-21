import torch as t
from typing import Any, Dict, List

@t.no_grad()
def match_intra_sae_features_feat_flow(
    sae: Any,
    feature_pool: t.Tensor,
    k: int = 5
) -> Dict[int, List[Dict[str, float]]]:
    """Matches features based on the Cosine Similarity of their decoder weights."""
    
    # Extract and normalize decoder weights
    W_dec = sae.W_dec[feature_pool] # (N, d_model)
    W_dec_norm = W_dec / (W_dec.norm(dim=-1, keepdim=True) + 1e-8)
    
    # Compute similarity matrix: (N, N)
    sim_matrix = W_dec_norm @ W_dec_norm.T 
    
    results = {}
    for i, feat_idx in enumerate(feature_pool.tolist()):
        sims = sim_matrix[i]
        
        # highest similarity first
        topk_vals, topk_inds = t.topk(sims, k + 1, largest=True)
        
        feat_results = []
        for val, ind in zip(topk_vals, topk_inds):
            if ind.item() == i:  # Skip self-match
                continue
            feat_results.append({
                'feature': feature_pool[ind].item(),
                'similarity': val.item()
            })
            
        results[feat_idx] = feat_results[:k]
        
    return results