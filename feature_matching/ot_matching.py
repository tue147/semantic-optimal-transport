import torch as t
from torch import Tensor
import numpy as np
import ot
from scipy.spatial.distance import cdist
from geomloss import SamplesLoss
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from functools import partial
from typing import Dict, List, Any
from transformer_lens import HookedTransformer
from .utils import *

@t.no_grad()
def align_saes_retrieve_and_rerank(
    data_toks: t.Tensor, 
    sae_a: Any, 
    sae_b: Any, 
    model: HookedTransformer,
    caching_hook: str | None = "ln_final.hook_normalized", 
    M: int = 500, 
    k: int = 10, 
    max_density: float = 0.1,
    activation_threshold: float = 1e-3,
    max_points_per_feature: int = 100,
    batch_size: int = 8,
    device: str = "cuda",
    chunk_size: int = 2000,
    stage1_metric: str = "cosine",
    ot_method: str = "sinkhorn",
    sinkhorn_impl: str = "old",
) -> Dict[int, List[Dict[str, float]]]:
    
    d_sae_a = sae_a.cfg.d_sae
    d_sae_b = sae_b.cfg.d_sae
    
    # "old" = compute_sinkhorn_ot_gpu_old, the helper behind the paper's numbers; "fixed" = the
    # corrected true-W_2 helper. See feature_matching/utils.py.
    sinkhorn_fn = {"old": compute_sinkhorn_ot_gpu_old, "fixed": compute_sinkhorn_ot_gpu}[sinkhorn_impl]
    print("Aligning SAE A to SAE B. Base Metric: L2 EUCLIDEAN")
    print(f"Stage 1: {stage1_metric.upper()}, Stage 2: {ot_method.upper()}")
    
    # --- 1. Fast Extraction Phase (Top-K per token) ---
    total_tokens = 0
    global_counts_a = t.zeros(d_sae_a, device=device)
    global_counts_b = t.zeros(d_sae_b, device=device)
    
    latents_list, idx_a_list, vals_a_list, idx_b_list, vals_b_list = [], [], [], [], []

    dataloader = DataLoader(TensorDataset(data_toks), batch_size=batch_size)
    
    for batch_toks in tqdm(dataloader, desc="Extracting Activations"):
        toks = batch_toks[0].to(device)
        flat_bs = toks.shape[0] * toks.shape[1]
        total_tokens += flat_bs
        
        batch_cache, batch_sae = {}, {}
        
        def cache_hook_fn(tens: Tensor, hook: Any):
            # Store latents directly on CPU in float16 to survive millions of tokens
            latents_list.append(tens.detach().cpu().half().view(flat_bs, -1))
            
        def hook_sae_fn(tens: Tensor, hook: Any, sae_key: str, counts: Tensor):
            acts = tens.detach().view(flat_bs, -1)
            mask = acts > activation_threshold
            counts += mask.sum(dim=0) # Update exact global density
            
            # Extract top features for this specific batch
            max_k = max(1, mask.sum(dim=-1).max().item())
            vals, idx = acts.topk(max_k, dim=-1) # type: ignore
            
            batch_sae[f'{sae_key}_vals'] = vals.cpu()
            batch_sae[f'{sae_key}_idx'] = idx.cpu()

        sae_a.reset_hooks()
        sae_b.reset_hooks()
        
        sae_a.add_hook(lambda n: "acts_post" in n, partial(hook_sae_fn, sae_key='A', counts=global_counts_a), dir="fwd")
        sae_b.add_hook(lambda n: "acts_post" in n, partial(hook_sae_fn, sae_key='B', counts=global_counts_b), dir="fwd")

        def sae_hook(tens: t.Tensor, hook: Any, sae: Any):
            sae(tens)
            return tens

        with model.hooks(fwd_hooks=[
            (caching_hook if caching_hook is not None else sae_a.cfg.hook_name, cache_hook_fn), 
            (sae_a.cfg.hook_name, partial(sae_hook, sae=sae_a)),
            (sae_b.cfg.hook_name, partial(sae_hook, sae=sae_b))
        ]):
            model(toks)

        idx_a_list.append(batch_sae['A_idx'])
        vals_a_list.append(batch_sae['A_vals'])
        idx_b_list.append(batch_sae['B_idx'])
        vals_b_list.append(batch_sae['B_vals'])

    sae_a.reset_hooks()
    sae_b.reset_hooks()

    # --- 2. Dynamic Padding & Concatenation ---
    print("Formatting and applying global sort...")
    all_latents = t.cat(latents_list, dim=0) # Shape: (total_toks, d_model)
    
    def pad_and_cat(idx_l, val_l):
        max_k = max(x.shape[1] for x in idx_l)
        p_idx, p_val = [], []
        for i, v in zip(idx_l, val_l):
            pad_len = max_k - i.shape[1]
            p_idx.append(t.nn.functional.pad(i, (0, pad_len), value=0) if pad_len > 0 else i)
            p_val.append(t.nn.functional.pad(v, (0, pad_len), value=0.0) if pad_len > 0 else v)
        return t.cat(p_idx, dim=0), t.cat(p_val, dim=0)

    all_idx_a, all_vals_a = pad_and_cat(idx_a_list, vals_a_list)
    all_idx_b, all_vals_b = pad_and_cat(idx_b_list, vals_b_list)

    # --- 3. Density Filter & Global Top-P Aggregation ---
    density_a = (global_counts_a / total_tokens).cpu().numpy()
    density_b = (global_counts_b / total_tokens).cpu().numpy()
    valid_a = set(np.where((density_a > 0) & (density_a <= max_density))[0])
    valid_b = set(np.where((density_b > 0) & (density_b <= max_density))[0])

    def get_global_top_points(idx_tensor, val_tensor, valid_set):
        flat_idx = idx_tensor.flatten().numpy()
        flat_vals = val_tensor.flatten().numpy()
        flat_toks = np.repeat(np.arange(total_tokens), idx_tensor.shape[1])
        
        mask = flat_vals > activation_threshold
        flat_idx, flat_vals, flat_toks = flat_idx[mask], flat_vals[mask], flat_toks[mask]
        
        sort_order = np.argsort(-flat_vals)
        flat_idx, flat_vals, flat_toks = flat_idx[sort_order], flat_vals[sort_order], flat_toks[sort_order]
        
        acts_data = {f: {'toks': [], 'weights': []} for f in valid_set}
        for f_id, val, tok in zip(flat_idx, flat_vals, flat_toks):
            if f_id in valid_set and len(acts_data[f_id]['weights']) < max_points_per_feature:
                acts_data[f_id]['toks'].append(tok)
                acts_data[f_id]['weights'].append(val)
        return acts_data

    acts_data_a = get_global_top_points(all_idx_a, all_vals_a, valid_a)
    acts_data_b = get_global_top_points(all_idx_b, all_vals_b, valid_b)

    # --- 4. Build OT Sets and Centroids (MODIFIED) ---
    print("Building Point Clouds and Centroids...")
    d_model = model.cfg.d_model
    valid_a_list = sorted(list(valid_a))
    valid_b_list = sorted(list(valid_b))
    
    centroids_a = t.zeros((len(valid_a_list), d_model), device=device)
    centroids_b = t.zeros((len(valid_b_list), d_model), device=device)
    
    processed_a, processed_b = {}, {}
    
    for i, (f_list, acts_data, centroids, processed) in enumerate([
        (valid_a_list, acts_data_a, centroids_a, processed_a),
        (valid_b_list, acts_data_b, centroids_b, processed_b)
    ]):
        desc = "SAE A" if i == 0 else "SAE B"
        for idx, f_id in tqdm(enumerate(f_list), total=len(f_list), desc=f"Transforming {desc}"):
            if not acts_data[f_id]['toks']: continue
            
            # Get raw latents
            X_raw = all_latents[acts_data[f_id]['toks']].float()
            
            V = X_raw
                
            W = t.tensor(acts_data[f_id]['weights'], dtype=t.float32)
            
            # Centroid = activation-weighted mean of the feature's hidden states
            centroids[idx] = (V.to(device) * W.to(device).unsqueeze(1)).sum(dim=0) / W.sum().to(device)
            
            W_np = (W / W.sum()).numpy().astype(np.float64)
            W_np[-1] = 1.0 - np.sum(W_np[:-1])
            processed[f_id] = {'X': V, 'W': W_np, 'W_t': W/W.sum()}

    # --- 5. Stage 1: Coarse Search (Top M) ---
    print(f"Executing Stage 1 Search (M={M}, Metric={stage1_metric})...")
    
    # EXACT METRIC NORMALIZATION
    if stage1_metric == "cosine":
        centroids_a = t.nn.functional.normalize(centroids_a, p=2, dim=1)
        centroids_b = t.nn.functional.normalize(centroids_b, p=2, dim=1)
    elif stage1_metric == "euclidean":
        # Global Max-Norm scaling for Stage 1 Euclidean centroids
        # Preserves ranking order exactly, but prevents massive float magnitudes
        max_norm_a = t.norm(centroids_a, p=2, dim=1).max()
        max_norm_b = t.norm(centroids_b, p=2, dim=1).max()
        global_max_norm = t.clamp(t.max(max_norm_a, max_norm_b), min=1e-9)
        
        centroids_a = centroids_a / global_max_norm
        centroids_b = centroids_b / global_max_norm

    top_M_indices_b = []
    for i in range(0, len(valid_a_list), chunk_size):
        chunk_a = centroids_a[i:i+chunk_size]
        if stage1_metric == "cosine":
            scores = t.matmul(chunk_a, centroids_b.T) 
            _, top_m_idx = t.topk(scores, M, dim=1, largest=True)
        elif stage1_metric == "euclidean":
            distances = t.cdist(chunk_a, centroids_b, p=2)
            _, top_m_idx = t.topk(distances, M, dim=1, largest=False)
        top_M_indices_b.append(top_m_idx.cpu())
        
    top_M_indices_b = t.cat(top_M_indices_b, dim=0).numpy()

    # --- 6. Stage 2: OT Reranking ---
    print("Executing Stage 2 Reranking...")
    final_alignments = {}
    
    if ot_method == "sinkhorn":
        gpu_solver = SamplesLoss(loss="sinkhorn", p=2, blur=0.01, scaling=0.9)

    for i, f_id_a in tqdm(enumerate(valid_a_list), total=len(valid_a_list)):
        if f_id_a not in processed_a: continue
        
        candidates_b_local_idx = top_M_indices_b[i]
        ot_distances = []
        
        X_a = processed_a[f_id_a]['X']
        
        for local_idx in candidates_b_local_idx:
            f_id_b = valid_b_list[local_idx]
            if f_id_b not in processed_b: continue
            
            X_b = processed_b[f_id_b]['X']
            
            if ot_method == "exact":
                cost = cdist(X_a.numpy(), X_b.numpy(), metric='euclidean') # type: ignore
                dist = float(ot.emd2(processed_a[f_id_a]['W'], processed_b[f_id_b]['W'], cost)) # type: ignore
            else:
                dist = sinkhorn_fn(
                    X_a.to(device), processed_a[f_id_a]['W_t'].to(device), 
                    X_b.to(device), processed_b[f_id_b]['W_t'].to(device), 
                    gpu_solver
                )
                
            ot_distances.append((f_id_b, dist))
            
        ot_distances.sort(key=lambda x: x[1])
        final_alignments[int(f_id_a)] = [
            {"target_feature": int(f_b), "ot_distance": d} for f_b, d in ot_distances[:k]
        ]

    return final_alignments