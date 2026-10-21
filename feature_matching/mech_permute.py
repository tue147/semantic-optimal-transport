import torch as t
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from functools import partial
from typing import Any, List, Dict
from scipy.optimize import linear_sum_assignment
from transformer_lens import HookedTransformer

@t.no_grad()
def align_saes_mechanistic_match(
    data_toks: t.Tensor, 
    sae_a: Any, 
    sae_b: Any, 
    model: HookedTransformer,
    k: int = 10, 
    max_density: float = 0.1,
    activation_threshold: float = 1e-3,
    batch_size: int = 8,
    device: str = "cuda",
    chunk_size: int = 2000,
    use_lap: bool = False
) -> Dict[int, List[Dict[str, float]]]:
    """
    Matches features between two SAEs using the "SAE Match" Parameter Folding method.
    Folds the decoder weights by the minimum dataset activation (simulating JumpReLU theta)
    and matches them based on Mean Squared Error (MSE).
    """
    d_sae_a = sae_a.cfg.d_sae
    d_sae_b = sae_b.cfg.d_sae
    d_model = model.cfg.d_model
    
    print(f"Aligning SAE A to SAE B using Folded Parameter MSE (LAP={use_lap})...")
    
    # 1. Initialize Trackers for Global Density and Minimum Activations
    total_tokens = 0
    global_counts_a = t.zeros(d_sae_a, device=device)
    global_counts_b = t.zeros(d_sae_b, device=device)
    
    # Start min_acts at infinity. We will progressively lower this to the true minimum.
    min_acts_a = t.full((d_sae_a,), float('inf'), device=device)
    min_acts_b = t.full((d_sae_b,), float('inf'), device=device)

    # 2. Forward Pass: Collect Global Min Activations
    dataloader = DataLoader(TensorDataset(data_toks), batch_size=batch_size)
    
    for batch_toks in tqdm(dataloader, desc="Extracting Minimum Activations"):
        toks = batch_toks[0].to(device)
        flat_bs = toks.shape[0] * toks.shape[1]
        total_tokens += flat_bs
        
        batch_sae = {}
        
        def hook_sae_a_fn(tens: t.Tensor, hook: Any): batch_sae['A'] = tens.detach().view(flat_bs, -1)
        def hook_sae_b_fn(tens: t.Tensor, hook: Any): batch_sae['B'] = tens.detach().view(flat_bs, -1)

        sae_a.reset_hooks()
        sae_b.reset_hooks()
        sae_a.add_hook(lambda n: "acts_post" in n, hook_sae_a_fn, dir="fwd")
        sae_b.add_hook(lambda n: "acts_post" in n, hook_sae_b_fn, dir="fwd")
        
        def sae_hook(tens: t.Tensor, hook: Any, sae: Any):
            sae(tens)
            return tens

        with model.hooks(fwd_hooks=[
            (sae_a.cfg.hook_name, partial(sae_hook, sae=sae_a)),
            (sae_b.cfg.hook_name, partial(sae_hook, sae=sae_b))
        ]):
            model(toks)

        # Process SAE A
        acts_a = batch_sae['A']
        mask_a = acts_a > activation_threshold
        global_counts_a += mask_a.sum(dim=0)
        
        # Replace non-active values with infinity so they don't break the minimum calculation
        active_only_a = t.where(mask_a, acts_a, float('inf'))
        batch_min_a = active_only_a.amin(dim=0)
        min_acts_a = t.minimum(min_acts_a, batch_min_a)
        
        # Process SAE B
        acts_b = batch_sae['B']
        mask_b = acts_b > activation_threshold
        global_counts_b += mask_b.sum(dim=0)
        
        active_only_b = t.where(mask_b, acts_b, float('inf'))
        batch_min_b = active_only_b.amin(dim=0)
        min_acts_b = t.minimum(min_acts_b, batch_min_b)

    sae_a.reset_hooks()
    sae_b.reset_hooks()

    # 3. Density Filtering
    print("Applying density filter and folding parameters...")
    density_a = (global_counts_a / total_tokens).cpu().numpy()
    density_b = (global_counts_b / total_tokens).cpu().numpy()
    
    valid_a = t.tensor(np.where((density_a > 0) & (density_a <= max_density))[0], device=device)
    valid_b = t.tensor(np.where((density_b > 0) & (density_b <= max_density))[0], device=device)
    
    # 4. Parameter Folding (W_dec * diag(theta))
    # sae.W_dec shape is expected to be (d_sae, d_model)
    W_dec_a = sae_a.W_dec[valid_a]
    W_dec_b = sae_b.W_dec[valid_b]
    
    filtered_min_acts_a = min_acts_a[valid_a].unsqueeze(1) # Shape: (N_valid_A, 1)
    filtered_min_acts_b = min_acts_b[valid_b].unsqueeze(1) # Shape: (N_valid_B, 1)
    
    # Multiply the columns/rows by the activation threshold (Folding)
    W_folded_a = W_dec_a * filtered_min_acts_a
    W_folded_b = W_dec_b * filtered_min_acts_b

    # 5. Compute the Folded MSE Distance Matrix (Chunked)
    print("Computing Folded MSE Matrix...")
    # Math trick for fast MSE: ||A - B||^2 = ||A||^2 + ||B||^2 - 2(A @ B.T)
    sq_norm_a = (W_folded_a ** 2).sum(dim=1, keepdim=True)
    sq_norm_b = (W_folded_b ** 2).sum(dim=1).unsqueeze(0)
    
    N_A = len(valid_a)
    N_B = len(valid_b)
    
    # We will build a dense numpy matrix if LAP is requested, 
    # otherwise we just store the Top-K directly to save memory.
    final_alignments = {}
    
    if use_lap:
        print("Running Exact Linear Assignment Problem (LAP) Solver...")
        if N_A * N_B > 5e8: # ~2GB matrix safeguard
            print("WARNING: Cost matrix is extremely large. LAP solver may freeze.")
            
        cost_matrix = np.zeros((N_A, N_B), dtype=np.float32)
        
        for i in range(0, N_A, chunk_size):
            chunk_a = W_folded_a[i:i+chunk_size]
            chunk_sq_a = sq_norm_a[i:i+chunk_size]
            
            # Compute MSE: (sq_norm_A + sq_norm_B - 2 * dot) / d_model
            dot_product = t.matmul(chunk_a, W_folded_b.T)
            mse_chunk = (chunk_sq_a + sq_norm_b - 2 * dot_product) / d_model
            cost_matrix[i:i+chunk_size] = mse_chunk.cpu().numpy()
            
        # LAP Solver: Returns the optimal 1-to-1 matching to minimize total MSE
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        for r, c in zip(row_ind, col_ind):
            f_id_a = int(valid_a[r].item())
            f_id_b = int(valid_b[c].item())
            mse_dist = float(cost_matrix[r, c])
            
            # LAP only provides a 1-to-1 mapping, so K=1 naturally.
            final_alignments[f_id_a] = [{"target_feature": f_id_b, "mse_distance": mse_dist}]
            
    else:
        print(f"Running Greedy Top-K Search (k={k})...")
        for i in tqdm(range(0, N_A, chunk_size), desc="MSE Search"):
            chunk_a = W_folded_a[i:i+chunk_size]
            chunk_sq_a = sq_norm_a[i:i+chunk_size]
            
            dot_product = t.matmul(chunk_a, W_folded_b.T)
            mse_chunk = (chunk_sq_a + sq_norm_b - 2 * dot_product) / d_model
            
            # We want the SMALLEST Mean Squared Error
            top_distances, top_indices = t.topk(mse_chunk, k, dim=1, largest=False)
            
            top_distances = top_distances.cpu().numpy()
            top_indices = top_indices.cpu().numpy()
            
            for local_r in range(top_distances.shape[0]):
                global_r = i + local_r
                f_id_a = int(valid_a[global_r].item())
                
                final_alignments[f_id_a] = [
                    {
                        "target_feature": int(valid_b[idx].item()), 
                        "mse_distance": float(dist)
                    } 
                    for dist, idx in zip(top_distances[local_r], top_indices[local_r])
                ]

    print("Alignment complete.")
    return final_alignments


@t.no_grad()
def align_mechanistic_permute_multilayer(
    data_toks: t.Tensor,
    sae_list: List[Any], 
    model: HookedTransformer,
    max_density: float = 0.1,
    activation_threshold: float = 1e-3,
    batch_size: int = 8,
    device: str = "cuda",
    chunk_size: int = 2000
) -> Dict[int, Dict[str, Any]]:
    """
    Traces feature evolution across multiple layers using Folded MSE and EXACT LAP.
    Filters density only at the first and last layers, but applies Parameter 
    Folding and the Linear Assignment Problem solver to all intermediate layers.
    """
    L = len(sae_list)
    assert L >= 2, "Must provide at least two SAEs to align."
    
    d_model = model.cfg.d_model
    d_sae = sae_list[0].cfg.d_sae # Assuming uniform SAE sizes across layers
    print(f"Executing Multi-Layer Mechanistic Permute (Exact LAP) across {L} SAEs...")

    # --- 1. Online Extraction (Density & Min Acts) ---
    global_counts_start = t.zeros(d_sae, device=device)
    global_counts_end = t.zeros(sae_list[-1].cfg.d_sae, device=device)
    min_acts = [t.full((sae.cfg.d_sae,), float('inf'), device=device) for sae in sae_list]
    total_tokens = 0

    dataloader = DataLoader(TensorDataset(data_toks), batch_size=batch_size)
    for sae in sae_list: sae.reset_hooks()

    for batch_toks in tqdm(dataloader, desc="Extracting Global Metrics"):
        toks = batch_toks[0].to(device)
        total_tokens += toks.shape[0] * toks.shape[1]
        
        def hook_sae_fn(tens: t.Tensor, hook: Any, idx: int, global_counts_end=global_counts_end, global_counts_start=global_counts_start):
            acts = tens.detach().view(-1, tens.shape[-1])
            mask = acts > activation_threshold
            
            if idx == 0:
                global_counts_start += mask.sum(dim=0)
            elif idx == L - 1:
                global_counts_end += mask.sum(dim=0)
                
            active_only = t.where(mask, acts, float('inf'))
            batch_min = active_only.amin(dim=0)
            min_acts[idx] = t.minimum(min_acts[idx], batch_min)

        for i, sae in enumerate(sae_list):
            sae.add_hook(lambda n: "acts_post" in n, partial(hook_sae_fn, idx=i), dir="fwd")

        fwd_hooks = [(sae.cfg.hook_name, lambda tens, hook, s=sae: s(tens)) for sae in sae_list]
        with model.hooks(fwd_hooks=fwd_hooks):
            model(toks)

    for sae in sae_list: sae.reset_hooks()

    # --- 2. Density Filtering (Start and End Only) ---
    density_start = (global_counts_start / total_tokens).cpu().numpy()
    density_end = (global_counts_end / total_tokens).cpu().numpy()
    
    valid_start = set(np.where((density_start > 0) & (density_start <= max_density))[0])
    valid_end = set(np.where((density_end > 0) & (density_end <= max_density))[0])

    # --- 3. Parameter Folding ---
    print("Folding parameters for all layers...")
    W_folded = []
    for i, sae in enumerate(sae_list):
        # FIX: Replace 'inf' from dead features with 0.0 before folding
        clean_min_acts = t.where(t.isinf(min_acts[i]), t.zeros_like(min_acts[i]), min_acts[i])
        # W_dec * diag(theta)
        W_folded.append(sae.W_dec.to(device) * clean_min_acts.unsqueeze(1))

    # --- 4. LAP Permutation Matrices ---
    print("Computing exact LAP permutations between adjacent layers...")
    layer_mappings = [] # Will store a dict mapping feature -> feature for each layer step
    
    for i in range(L - 1):
        print(f"  Solving LAP for Layer {i} -> {i+1}...")
        W_A = W_folded[i]
        W_B = W_folded[i+1]
        
        N_A, N_B = W_A.shape[0], W_B.shape[0]
        cost_matrix = np.zeros((N_A, N_B), dtype=np.float32)
        
        sq_norm_A = (W_A ** 2).sum(dim=1, keepdim=True)
        sq_norm_B = (W_B ** 2).sum(dim=1).unsqueeze(0)
        
        # Chunked computation of the MSE distance matrix to prevent VRAM/RAM OOM
        for chunk_idx in range(0, N_A, chunk_size):
            chunk_a = W_A[chunk_idx:chunk_idx+chunk_size]
            chunk_sq_a = sq_norm_A[chunk_idx:chunk_idx+chunk_size]
            
            mse = (chunk_sq_a + sq_norm_B - 2 * t.matmul(chunk_a, W_B.T)) / d_model
            
            # FIX: Sanitize the output to prevent floating point drift creating NaNs
            mse = t.nan_to_num(mse, nan=1e4, posinf=1e4, neginf=0.0)
            cost_matrix[chunk_idx:chunk_idx+chunk_size] = mse.cpu().numpy()
            
        # Execute Exact Linear Assignment Problem (O(N^3))
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        
        # Store the 1-to-1 mapping for this step
        step_mapping = {int(r): int(c) for r, c in zip(row_ind, col_ind)}
        layer_mappings.append(step_mapping)

    # --- 5. Composing Permutations (Path Tracing) ---
    print("Composing permutations to trace multi-layer paths...")
    final_alignments = {}
    
    for f_start in valid_start:
        current_f = f_start
        path = [current_f]
        
        # Trace the exact LAP path through all layers
        for step_mapping in layer_mappings:
            current_f = step_mapping[current_f]
            path.append(current_f)
            
        f_end = current_f
        
        # Only retain paths that land on a valid (density-filtered) final feature
        if f_end in valid_end:
            final_alignments[f_start] = {
                "path": path,
                "target_feature": f_end
            }

    print(f"Alignment complete. Found {len(final_alignments)} valid paths.")
    return final_alignments