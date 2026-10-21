import torch as t
from torch import Tensor
import numpy as np
import ot
from scipy.spatial.distance import cdist
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from functools import partial
from typing import Any, Dict, List
from sae_lens import HookedSAETransformer

@t.no_grad()
def match_intra_sae_features_ot(
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
    """
    Finds the closest semantic matches for each feature within a specified pool 
    from the SAME SAE, using Optimal Transport on their activation point clouds.
    """
    feat_list = feature_pool.tolist()
    n_features = len(feat_list)
    
    if n_features < 2:
        print("Feature pool must contain at least 2 features to find matches.")
        return {}
        
    print(f"Executing Intra-SAE OT Match for {n_features} features...")
    
    # =====================================================================
    # STEP 1: Surgical Extraction
    # =====================================================================
    feature_activations = {f: {'toks': [], 'weights': []} for f in feat_list}
    latents_list = []
    global_tok_offset = 0
    
    sae.reset_hooks()
    
    def hook_sae_fn(inputs: Tensor, hook: Any) -> None:
        # Expected shape: (batch * seq, d_sae)
        acts = inputs.detach().view(-1, inputs.shape[-1])
        
        # Slice OUT only the features we care about to save VRAM
        pool_acts = acts[:, feature_pool]
        
        for i, feat_idx in enumerate(feat_list):
            f_acts = pool_acts[:, i]
            active_mask = f_acts > activation_threshold
            active_local_toks = active_mask.nonzero(as_tuple=True)[0]
            
            if len(active_local_toks) > 0:
                # Store CPU tensors immediately
                feature_activations[feat_idx]['toks'].append(active_local_toks.cpu() + global_tok_offset)
                feature_activations[feat_idx]['weights'].append(f_acts[active_local_toks].cpu())

    sae.add_hook(lambda n: "acts_post" in n, hook_sae_fn, dir="fwd")
    
    dataloader = DataLoader(TensorDataset(data_toks), batch_size=batch_size)
    
    def trigger_sae(tens: Tensor, hook: Any):
        _ = sae(tens)
        return tens
    
    for batch_toks in tqdm(dataloader, desc="Extracting Pool Activations"):
        toks = batch_toks[0].to(model.cfg.device)
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

    # =====================================================================
    # STEP 2: Process Sets for OT (Top-K Filter)
    # =====================================================================
    print("Formatting latents and applying global max_points filter...")
    all_latents = t.cat(latents_list, dim=0)
    
    processed_features = []
    valid_feat_list = []
    
    for f_id in feat_list:
        toks_list = feature_activations[f_id]['toks']
        weights_list = feature_activations[f_id]['weights']
        
        if not toks_list:
            continue
            
        T = t.cat(toks_list, dim=0)
        W = t.cat(weights_list, dim=0)
        
        if len(W) > max_points_per_feature:
            W, top_indices = t.topk(W, max_points_per_feature)
            T = T[top_indices]
            
        X = all_latents[T].float().numpy()
        
        W_np = W.float().numpy().astype(np.float64)
        W_np = W_np / W_np.sum()
        W_np[-1] = 1.0 - np.sum(W_np[:-1]) # Guarantee sum=1.0 for POT
        
        processed_features.append({'X': X, 'W': W_np})
        valid_feat_list.append(f_id)
        
    n_valid = len(valid_feat_list)
    if n_valid < 2:
        print("Not enough active features in the pool to compute matches.")
        return {}

    # =====================================================================
    # STEP 3: Compute SYMMETRIC OT Matrix
    # =====================================================================
    print(f"Computing Symmetric OT Matrix for {n_valid} valid features...")
    D_sem = np.full((n_valid, n_valid), np.inf) # Initialize with infinity
    
    # Only compute upper triangle
    for i in tqdm(range(n_valid), desc="OT Distance Loop"):
        X_i, w_i = processed_features[i]['X'], processed_features[i]['W']
        D_sem[i, i] = 0.0 # Distance to self is 0
        
        for j in range(i + 1, n_valid):
            X_j, w_j = processed_features[j]['X'], processed_features[j]['W']
            
            cost_matrix = cdist(X_i, X_j, metric='euclidean') # type: ignore
            ot_dist = float(ot.emd2(w_i, w_j, cost_matrix)) # type: ignore
            
            # Mirror the distance
            D_sem[i, j] = ot_dist
            D_sem[j, i] = ot_dist

    # =====================================================================
    # STEP 4: Rank and Extract Top-K Matches
    # =====================================================================
    print("Ranking matches...")
    final_alignments = {}
    
    # We want to force the diagonal (self-match) to infinity so it isn't returned
    np.fill_diagonal(D_sem, np.inf)
    
    for i, source_f_id in enumerate(valid_feat_list):
        distances = D_sem[i]
        
        # Sort indices by smallest OT distance
        best_indices = np.argsort(distances)[:k]
        
        matches = []
        for target_idx in best_indices:
            dist = distances[target_idx]
            if not np.isinf(dist): # Ensure we don't accidentally grab invalid comparisons
                matches.append({
                    "feature": valid_feat_list[target_idx],
                    "ot_distance": float(dist)
                })
                
        final_alignments[source_f_id] = matches

    print("Intra-SAE Matching Complete.")
    return final_alignments