import torch as t
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from typing import Any, Dict, List
from sae_lens import HookedSAETransformer

@t.no_grad()
def match_intra_sae_features_naive(
    data_toks: Tensor,
    model: HookedSAETransformer,
    sae: Any,
    feature_pool: t.Tensor,
    k: int = 5,
    caching_hook: str = "ln_final.hook_normalized",
    max_points_per_feature: int = 500,
    activation_threshold: float = 1e-3,
    batch_size: int = 8,
) -> Dict[int, List[Dict[str, float]]]:
    
    feat_list = feature_pool.tolist()
    feature_activations = {
        feat_idx: {'toks': [], 'weights': []} for feat_idx in feat_list
    }
    
    latents_list = []
    global_tok_offset = 0
    sae.reset_hooks()
    
    def hook_sae_fn(inputs: Tensor, hook: Any) -> None:
        acts = inputs.detach().view(-1, inputs.shape[-1])
        pool_acts = acts[:, feature_pool]
        
        for i, feat_idx in enumerate(feat_list):
            f_acts = pool_acts[:, i]
            active_mask = f_acts > activation_threshold
            active_local_toks = active_mask.nonzero(as_tuple=True)[0]
            
            if len(active_local_toks) > 0:
                feature_activations[feat_idx]['toks'].append(active_local_toks.cpu() + global_tok_offset)
                feature_activations[feat_idx]['weights'].append(f_acts[active_local_toks].cpu())

    sae.add_hook(lambda n: "acts_post" in n, hook_sae_fn, dir="fwd")
    
    dataloader = DataLoader(TensorDataset(data_toks), batch_size=batch_size)
    
    def trigger_sae(tens: Tensor, hook: Any):
        _ = sae(tens)
        return tens
    
    for (batch_toks,) in tqdm(dataloader, desc="Extracting Latents and Activations"):
        toks = batch_toks.to(model.cfg.device)
        flat_bs = toks.shape[0] * toks.shape[1]
        
        batch_cache = {}
        def cache_hook_fn(tens: Tensor, hook: Any):
            batch_cache['acts'] = tens.detach().cpu().half().view(flat_bs, -1)
            
        with model.hooks(fwd_hooks=[
            (caching_hook, cache_hook_fn),
            (sae.cfg.hook_name, trigger_sae)
        ]):
            model(toks)
            
        latents_list.append(batch_cache['acts'])
        global_tok_offset += flat_bs
        
    sae.reset_hooks()
    
    # Consolidate all CPU latents
    all_latents = t.cat(latents_list, dim=0).to(t.float32) # Convert back to fp32 for math
    
    centroids = []
    valid_features = []
    
    for feat_idx in feat_list:
        if len(feature_activations[feat_idx]['weights']) == 0:
            # Fallback for dead features
            centroids.append(t.zeros(model.cfg.d_model, device=model.cfg.device))
            valid_features.append(feat_idx)
            continue
            
        f_toks = t.cat(feature_activations[feat_idx]['toks'])
        f_weights = t.cat(feature_activations[feat_idx]['weights'])
        
        if len(f_weights) > max_points_per_feature:
            _, top_indices = t.topk(f_weights, max_points_per_feature)
            top_global_toks = f_toks[top_indices]
        else:
            top_global_toks = f_toks
            
        # Extract corresponding hook vectors from the global tensor
        valid_vecs = all_latents[top_global_toks].to(model.cfg.device)
        centroids.append(valid_vecs.mean(dim=0))
        valid_features.append(feat_idx)
        
    centroids_tensor = t.stack(centroids) # (N, d_model)
    
    # Compute Pairwise L2
    dist_matrix = t.cdist(centroids_tensor, centroids_tensor, p=2.0)
    
    results = {}
    for i, feat_idx in enumerate(valid_features):
        dists = dist_matrix[i]
        topk_vals, topk_inds = t.topk(dists, k + 1, largest=False)
        
        feat_results = []
        for val, ind in zip(topk_vals, topk_inds):
            if ind.item() == i: 
                continue
            feat_results.append({
                'feature': valid_features[ind.item()], # type: ignore
                'l2_distance': val.item()
            })
            
        results[feat_idx] = feat_results[:k]
        
    return results