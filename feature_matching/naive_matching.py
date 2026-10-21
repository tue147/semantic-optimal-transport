import torch as t
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from functools import partial
from typing import Any, List, Dict
from transformer_lens import HookedTransformer

@t.no_grad()
def align_saes_centroid_match(
    data_toks: t.Tensor,
    sae_a: Any, 
    sae_b: Any, 
    model: HookedTransformer,
    caching_hook: str | None = "ln_final.hook_normalized", 
    k: int = 10, 
    max_density: float = 0.1,
    activation_threshold: float = 1e-3,
    batch_size: int = 8,
    device: str = "cuda",
    chunk_size: int = 2048,
    metric: str = "cosine"  # 'cosine' or 'euclidean'
) -> Dict[int, List[Dict[str, float]]]:
    """
    Matches features by computing the naive weighted centroid of their 
    model activations, replacing Optimal Transport with simple distance metrics.
    """
    d_sae_a = sae_a.cfg.d_sae
    d_sae_b = sae_b.cfg.d_sae
    d_model = model.cfg.d_model
    
    print(f"Executing Naive Centroid Match ({metric.upper()})...")

    # --- 1. Global Trackers for Online Centroids ---
    total_tokens = 0
    global_counts_a = t.zeros(d_sae_a, device=device)
    global_counts_b = t.zeros(d_sae_b, device=device)
    
    # Numerator: Sum of (activation_weight * latent_vector)
    # Denominator: Sum of activation_weights
    centroid_num_a = t.zeros((d_sae_a, d_model), device=device)
    centroid_den_a = t.zeros(d_sae_a, device=device)
    centroid_num_b = t.zeros((d_sae_b, d_model), device=device)
    centroid_den_b = t.zeros(d_sae_b, device=device)

    dataloader = DataLoader(TensorDataset(data_toks), batch_size=batch_size)
    sae_a.reset_hooks()
    sae_b.reset_hooks()

    for batch_toks in tqdm(dataloader, desc="Extracting Online Centroids"):
        toks = batch_toks[0].to(device)
        flat_bs = toks.shape[0] * toks.shape[1]
        total_tokens += flat_bs
        
        batch_cache, batch_sae = {}, {}
        
        def cache_hook_fn(tens: t.Tensor, hook: Any): 
            batch_cache['acts'] = tens.detach().view(flat_bs, -1)
            
        def hook_sae_a_fn(tens: t.Tensor, hook: Any): batch_sae['A'] = tens.detach().view(flat_bs, -1)
        def hook_sae_b_fn(tens: t.Tensor, hook: Any): batch_sae['B'] = tens.detach().view(flat_bs, -1)

        sae_a.add_hook(lambda n: "acts_post" in n, hook_sae_a_fn, dir="fwd")
        sae_b.add_hook(lambda n: "acts_post" in n, hook_sae_b_fn, dir="fwd")
        
        def sae_hook(tens: t.Tensor, hook: Any, sae: Any):
            sae(tens)
            return tens

        with model.hooks(fwd_hooks=[
            (caching_hook if caching_hook is not None else sae_a.cfg.hook_name, cache_hook_fn), 
            (sae_a.cfg.hook_name, partial(sae_hook, sae=sae_a)),
            (sae_b.cfg.hook_name, partial(sae_hook, sae=sae_b))
        ]):
            model(toks)
            
        X = batch_cache['acts'].float()

        # Update SAE A
        acts_a = batch_sae['A']
        mask_a = acts_a > activation_threshold
        global_counts_a += mask_a.sum(dim=0)
        
        rows_a, cols_a = mask_a.nonzero(as_tuple=True)
        for f_idx in cols_a.unique():
            r_idx = rows_a[cols_a == f_idx]
            w = acts_a[r_idx, f_idx]
            centroid_num_a[f_idx] += (X[r_idx] * w.unsqueeze(1)).sum(dim=0)
            centroid_den_a[f_idx] += w.sum()
            
        # Update SAE B
        acts_b = batch_sae['B']
        mask_b = acts_b > activation_threshold
        global_counts_b += mask_b.sum(dim=0)
        
        rows_b, cols_b = mask_b.nonzero(as_tuple=True)
        for f_idx in cols_b.unique():
            r_idx = rows_b[cols_b == f_idx]
            w = acts_b[r_idx, f_idx]
            centroid_num_b[f_idx] += (X[r_idx] * w.unsqueeze(1)).sum(dim=0)
            centroid_den_b[f_idx] += w.sum()

    sae_a.reset_hooks()
    sae_b.reset_hooks()

    # --- 2. Filter and Finalize Centroids ---
    print("Finalizing Centroids...")
    density_a = (global_counts_a / total_tokens).cpu().numpy()
    density_b = (global_counts_b / total_tokens).cpu().numpy()
    
    valid_a = t.tensor(np.where((density_a > 0) & (density_a <= max_density))[0], device=device)
    valid_b = t.tensor(np.where((density_b > 0) & (density_b <= max_density))[0], device=device)

    # Divide sum(w * X) by sum(w) to get the true weighted mean
    den_a_safe = t.clamp(centroid_den_a[valid_a], min=1e-9).unsqueeze(1)
    den_b_safe = t.clamp(centroid_den_b[valid_b], min=1e-9).unsqueeze(1)
    
    C_a = centroid_num_a[valid_a] / den_a_safe
    C_b = centroid_num_b[valid_b] / den_b_safe

    if metric == "cosine":
        C_a = t.nn.functional.normalize(C_a, p=2, dim=1)
        C_b = t.nn.functional.normalize(C_b, p=2, dim=1)

    # --- 3. Compute Distances and Rank ---
    print(f"Ranking Top-{k} candidates via {metric.upper()}...")
    final_alignments = {}
    
    for i in range(0, len(valid_a), chunk_size):
        chunk_a = C_a[i:i+chunk_size]
        
        if metric == "cosine":
            # Largest cosine similarity
            sim = t.matmul(chunk_a, C_b.T)
            top_vals, top_idx = t.topk(sim, k, dim=1, largest=True)
        elif metric == "euclidean":
            # Smallest euclidean distance
            dist = t.cdist(chunk_a, C_b, p=2)
            top_vals, top_idx = t.topk(dist, k, dim=1, largest=False)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        top_vals = top_vals.cpu().numpy()
        top_idx = top_idx.cpu().numpy()
        
        for local_idx in range(top_vals.shape[0]):
            global_idx = i + local_idx
            f_id_a = int(valid_a[global_idx].item())
            
            final_alignments[f_id_a] = [
                {
                    "target_feature": int(valid_b[idx].item()),
                    "metric_value": float(val)
                }
                for val, idx in zip(top_vals[local_idx], top_idx[local_idx])
            ]

    return final_alignments