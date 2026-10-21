import torch as t
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from functools import partial
from typing import Any, List, Dict
from transformer_lens import HookedTransformer
from functools import partial

@t.no_grad()
def align_saes_feature_flow_match(
    data_toks: t.Tensor,
    sae_a: Any, 
    sae_b: Any, 
    model: HookedTransformer,
    k: int = 5, 
    device: str = "cuda",
    chunk_size: int = 2048,
    similarity_threshold: float = 0.0,
    max_density: float = 0.1,
    activation_threshold: float = 1e-3,
    batch_size: int = 8
) -> Dict[int, List[Dict[str, float]]]:
    """
    Matches features using data-free Cosine Similarity of decoder weights,
    but applies a preliminary data pass to filter out features above max_density.
    """
    d_sae_a = sae_a.cfg.d_sae
    d_sae_b = sae_b.cfg.d_sae
    
    print(f"Executing Feature Flow Match (W_dec Cosine) with Density Filter...")

    # --- 1. Quick Forward Pass for Density ---
    total_tokens = 0
    global_counts_a = t.zeros(d_sae_a, device=device)
    global_counts_b = t.zeros(d_sae_b, device=device)

    dataloader = DataLoader(TensorDataset(data_toks), batch_size=batch_size)
    sae_a.reset_hooks()
    sae_b.reset_hooks()

    for batch_toks in tqdm(dataloader, desc="Calculating Densities"):
        toks = batch_toks[0].to(device)
        flat_bs = toks.shape[0] * toks.shape[1]
        total_tokens += flat_bs
        
        batch_sae = {}
        def hook_sae_a_fn(tens: t.Tensor, hook: Any): batch_sae['A'] = tens.detach().view(flat_bs, -1)
        def hook_sae_b_fn(tens: t.Tensor, hook: Any): batch_sae['B'] = tens.detach().view(flat_bs, -1)

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

        acts_a = batch_sae['A']
        global_counts_a += (acts_a > activation_threshold).sum(dim=0)
        
        acts_b = batch_sae['B']
        global_counts_b += (acts_b > activation_threshold).sum(dim=0)

    sae_a.reset_hooks()
    sae_b.reset_hooks()

    # --- 2. Apply Density Filter ---
    density_a = (global_counts_a / total_tokens).cpu().numpy()
    density_b = (global_counts_b / total_tokens).cpu().numpy()
    
    valid_a = t.tensor(np.where((density_a > 0) & (density_a <= max_density))[0], device=device)
    valid_b = t.tensor(np.where((density_b > 0) & (density_b <= max_density))[0], device=device)
    
    if len(valid_a) == 0 or len(valid_b) == 0:
        print("Warning: Filtering removed all features.")
        return {}

    # --- 3. Extract and Normalize Valid Decoder Weights ---
    W_dec_a = sae_a.W_dec[valid_a].float()
    W_dec_b = sae_b.W_dec[valid_b].float()
    
    W_norm_a = t.nn.functional.normalize(W_dec_a, p=2, dim=1)
    W_norm_b = t.nn.functional.normalize(W_dec_b, p=2, dim=1)

    # --- 4. Matrix Search ---
    print("Computing Decoder Cosine Flow...")
    final_alignments = {}
    
    for i in range(0, len(valid_a), chunk_size):
        chunk_a = W_norm_a[i:i+chunk_size]
        sim_matrix = t.matmul(chunk_a, W_norm_b.T)
        
        top_scores, top_indices = t.topk(sim_matrix, k, dim=1, largest=True)
        top_scores = top_scores.cpu().numpy()
        top_indices = top_indices.cpu().numpy()
        
        for local_idx in range(top_scores.shape[0]):
            global_idx = i + local_idx
            f_id_a = int(valid_a[global_idx].item())
            
            matches = []
            for score, idx in zip(top_scores[local_idx], top_indices[local_idx]):
                if score > similarity_threshold:
                    matches.append({
                        "target_feature": int(valid_b[idx].item()),
                        "cosine_similarity": float(score)
                    })
            
            if matches:
                final_alignments[f_id_a] = matches

    return final_alignments

@t.no_grad()
def align_feature_flow_multilayer(
    data_toks: t.Tensor,
    sae_list: List[Any], 
    model: HookedTransformer,
    k: int = 5, 
    max_density: float = 0.1,
    activation_threshold: float = 1e-3,
    batch_size: int = 8,
    device: str = "cuda",
    chunk_size: int = 2048
) -> Dict[int, Dict[str, Any]]:
    """
    Traces feature flow across multiple layers using Data-Free Cosine Similarity.
    Follows a 1-to-1 path until the final layer mapping, where it branches to Top-K.
    """
    L = len(sae_list)
    assert L >= 2, "Must provide at least two SAEs to align."
    print(f"Executing Multi-Layer Feature Flow across {L} SAEs...")

    # --- 1. Online Extraction (Density Only) ---
    global_counts_start = t.zeros(sae_list[0].cfg.d_sae, device=device)
    global_counts_end = t.zeros(sae_list[-1].cfg.d_sae, device=device)
    total_tokens = 0

    dataloader = DataLoader(TensorDataset(data_toks), batch_size=batch_size)
    for sae in (sae_list[0], sae_list[-1]): sae.reset_hooks()

    for batch_toks in tqdm(dataloader, desc="Extracting Density Metrics"):
        toks = batch_toks[0].to(device)
        total_tokens += toks.shape[0] * toks.shape[1]
        
        def hook_sae_start(tens: t.Tensor, hook: Any, global_counts_start=global_counts_start):
            acts = tens.detach().view(-1, tens.shape[-1])
            global_counts_start += (acts > activation_threshold).sum(dim=0)
            
        def hook_sae_end(tens: t.Tensor, hook: Any, global_counts_end=global_counts_end):
            acts = tens.detach().view(-1, tens.shape[-1])
            global_counts_end += (acts > activation_threshold).sum(dim=0)

        sae_list[0].add_hook(lambda n: "acts_post" in n, hook_sae_start, dir="fwd")
        sae_list[-1].add_hook(lambda n: "acts_post" in n, hook_sae_end, dir="fwd")

        fwd_hooks = [(sae.cfg.hook_name, lambda tens, hook, s=sae: s(tens)) for sae in (sae_list[0], sae_list[-1])]
        with model.hooks(fwd_hooks=fwd_hooks):
            model(toks)

    for sae in (sae_list[0], sae_list[-1]): sae.reset_hooks()

    # --- 2. Density Filtering (Start and End Only) ---
    density_start = (global_counts_start / total_tokens).cpu().numpy()
    density_end = (global_counts_end / total_tokens).cpu().numpy()
    
    valid_start = t.tensor(np.where((density_start > 0) & (density_start <= max_density))[0], device=device)
    valid_end = t.tensor(np.where((density_end > 0) & (density_end <= max_density))[0], device=device)

    # --- 3. Normalize Decoder Weights ---
    print("Normalizing Decoder weights...")
    W_norm = [t.nn.functional.normalize(sae.W_dec.to(device).float(), p=2, dim=1) for sae in sae_list]

    # --- 4. Path Tracing ---
    print("Tracing Flow...")
    current_f = valid_start.clone()
    paths = current_f.unsqueeze(1) # Shape: (N_start, 1)

    # Trace intermediate layers (1-to-1 Greedy)
    for i in range(L - 2):
        W_A = W_norm[i][current_f]
        W_B = W_norm[i+1] # Match against ALL features
        
        best_matches = t.zeros(len(valid_start), dtype=t.long, device=device)
        for chunk_idx in range(0, len(valid_start), chunk_size):
            chunk_a = W_A[chunk_idx:chunk_idx+chunk_size]
            sim = t.matmul(chunk_a, W_B.T)
            best_matches[chunk_idx:chunk_idx+chunk_size] = sim.argmax(dim=1)
            
        current_f = best_matches
        paths = t.cat([paths, current_f.unsqueeze(1)], dim=1)

    # Trace final layer (Branch to Top-K in valid_end)
    W_A = W_norm[L-2][current_f]
    W_B = W_norm[L-1][valid_end]
    
    top_k_indices = t.zeros((len(valid_start), k), dtype=t.long, device=device)
    top_k_scores = t.zeros((len(valid_start), k), dtype=t.float, device=device)
    
    for chunk_idx in range(0, len(valid_start), chunk_size):
        chunk_a = W_A[chunk_idx:chunk_idx+chunk_size]
        sim = t.matmul(chunk_a, W_B.T)
        scores, indices = sim.topk(k, dim=1, largest=True)
        top_k_scores[chunk_idx:chunk_idx+chunk_size] = scores
        top_k_indices[chunk_idx:chunk_idx+chunk_size] = indices

    # --- 5. Formatting ---
    final_alignments = {}
    paths_cpu = paths.cpu().numpy()
    scores_cpu = top_k_scores.cpu().numpy()
    indices_cpu = top_k_indices.cpu().numpy()
    
    for i, row in enumerate(paths_cpu):
        f_start = int(row[0])
        
        top_targets = []
        for rank in range(k):
            local_tgt_idx = indices_cpu[i, rank]
            score = scores_cpu[i, rank]
            global_tgt_idx = int(valid_end[local_tgt_idx].item())
            
            top_targets.append({
                "target_feature": global_tgt_idx,
                "cosine_similarity": float(score)
            })
            
        final_alignments[f_start] = {
            "intermediate_path": [int(f) for f in row],
            "top_k_targets": top_targets
        }

    return final_alignments