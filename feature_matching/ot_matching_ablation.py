"""Ablation-friendly split of `ot_matching.align_saes_retrieve_and_rerank`, used by the support-size
(K) ablation in ot_topk_ablation.py (Appendix D).

The stock matcher extracts activations, builds point clouds + centroids, runs the Stage-1 coarse
centroid search, and reranks with Sinkhorn OT -- all in one call, over *every* valid feature. The
K ablation needs two things the stock function does not expose:

  * restrict the (expensive) Stage-2 rerank to the 100 *query* features (side A);
  * rebuild centroids / weights from the top-K activations of each feature (K = the number of
    support points, i.e. `max_points_per_feature`) without re-extracting activations.

The numeric bodies below are copied verbatim from `ot_matching.py` (sections 1-6) so the ablation
reproduces the stock method exactly; only the control flow is refactored.
"""

import time
from functools import partial
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import ot
import torch as t
from geomloss import SamplesLoss
from scipy.spatial.distance import cdist
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from transformer_lens import HookedTransformer

from .utils import compute_sinkhorn_ot_gpu, compute_sinkhorn_ot_gpu_old

# Which Sinkhorn helper Stage 2 uses. "fixed" = the corrected `compute_sinkhorn_ot_gpu` (true W_2);
# "old" = `compute_sinkhorn_ot_gpu_old`, the helper every published "Our" number was produced with
# (ranks by 0.5*W_2^2/s with a per-pair s). Use "old" to stay comparable with the paper's tables.
SINKHORN_IMPLS = {"fixed": compute_sinkhorn_ot_gpu, "old": compute_sinkhorn_ot_gpu_old}


# =============================================================================================
# Section 1-3: activation extraction, density filter, global top-P aggregation.
# Produces `acts_data_{a,b}` = {feature_id: {'toks': [...], 'weights': [...]}} holding, per valid
# feature, up to `max_points_per_feature` (token index, activation) points sorted by activation
# descending. Centroids / point clouds are built later (they depend on K), so this runs once.
# =============================================================================================
@t.no_grad()
def extract_ot_state(
    data_toks: Tensor,
    sae_a: Any,
    sae_b: Any,
    model: HookedTransformer,
    caching_hook: Optional[str] = None,       # None -> sae_a.cfg.hook_name (the paper's R = layer A)
    max_density: float = 0.1,
    activation_threshold: float = 1e-3,
    max_points_per_feature: int = 500,
    batch_size: int = 8,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Sections 1-3 of the stock matcher. Returns a state dict reused across seeds and K settings."""
    t0 = time.perf_counter()
    d_sae_a = sae_a.cfg.d_sae
    d_sae_b = sae_b.cfg.d_sae

    total_tokens = 0
    global_counts_a = t.zeros(d_sae_a, device=device)
    global_counts_b = t.zeros(d_sae_b, device=device)

    latents_list, idx_a_list, vals_a_list, idx_b_list, vals_b_list = [], [], [], [], []

    dataloader = DataLoader(TensorDataset(data_toks), batch_size=batch_size)

    for batch_toks in tqdm(dataloader, desc="Extracting Activations"):
        toks = batch_toks[0].to(device)
        flat_bs = toks.shape[0] * toks.shape[1]
        total_tokens += flat_bs

        batch_sae: Dict[str, Any] = {}

        def cache_hook_fn(tens: Tensor, hook: Any):
            latents_list.append(tens.detach().cpu().half().view(flat_bs, -1))

        def hook_sae_fn(tens: Tensor, hook: Any, sae_key: str, counts: Tensor):
            acts = tens.detach().view(flat_bs, -1)
            mask = acts > activation_threshold
            counts += mask.sum(dim=0)
            max_k = max(1, mask.sum(dim=-1).max().item())
            vals, idx = acts.topk(max_k, dim=-1)  # type: ignore
            batch_sae[f'{sae_key}_vals'] = vals.cpu()
            batch_sae[f'{sae_key}_idx'] = idx.cpu()

        sae_a.reset_hooks()
        sae_b.reset_hooks()
        sae_a.add_hook(lambda n: "acts_post" in n, partial(hook_sae_fn, sae_key='A', counts=global_counts_a), dir="fwd")
        sae_b.add_hook(lambda n: "acts_post" in n, partial(hook_sae_fn, sae_key='B', counts=global_counts_b), dir="fwd")

        def sae_hook(tens: Tensor, hook: Any, sae: Any):
            sae(tens)
            return tens

        with model.hooks(fwd_hooks=[
            (caching_hook if caching_hook is not None else sae_a.cfg.hook_name, cache_hook_fn),
            (sae_a.cfg.hook_name, partial(sae_hook, sae=sae_a)),
            (sae_b.cfg.hook_name, partial(sae_hook, sae=sae_b)),
        ]):
            model(toks)

        idx_a_list.append(batch_sae['A_idx'])
        vals_a_list.append(batch_sae['A_vals'])
        idx_b_list.append(batch_sae['B_idx'])
        vals_b_list.append(batch_sae['B_vals'])

    sae_a.reset_hooks()
    sae_b.reset_hooks()

    # --- Section 2: dynamic padding & concatenation ---
    print("Formatting and applying global sort...")
    all_latents = t.cat(latents_list, dim=0)

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

    # --- Section 3: density filter & global top-P aggregation ---
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

    state = {
        "all_latents": all_latents,          # (total_tokens, d_model) cpu fp16
        "acts_data_a": acts_data_a,
        "acts_data_b": acts_data_b,
        "valid_a_list": sorted(list(valid_a)),
        "valid_b_list": sorted(list(valid_b)),
        "global_counts_a": global_counts_a.cpu(),   # firing counts (side A) -> non-dead eligibility
        "global_counts_b": global_counts_b.cpu(),
        "total_tokens": total_tokens,
        "d_model": model.cfg.d_model,
        "max_points_per_feature": max_points_per_feature,
        "timing": {"extract_s": time.perf_counter() - t0},
    }
    return state


# =============================================================================================
# Section 4: build point clouds + centroids, optionally truncating each feature's support to the
# top-K activations. K=None uses every stored point (the extraction's max_points_per_feature).
# =============================================================================================
def build_clouds(state: Dict[str, Any], device: str = "cuda", K: Optional[int] = None) -> Dict[str, Any]:
    """Build centroids + weighted point clouds. `K` truncates each feature's support to its top-K
    activations (points are stored activation-descending), which is exactly the `max_points_per_feature`
    knob of the stock matcher -- so this is how experiment #2 sweeps K without re-extracting."""
    d_model = state["d_model"]
    valid_a_list = state["valid_a_list"]
    valid_b_list = state["valid_b_list"]
    all_latents = state["all_latents"]

    centroids_a = t.zeros((len(valid_a_list), d_model), device=device)
    centroids_b = t.zeros((len(valid_b_list), d_model), device=device)
    processed_a: Dict[int, Any] = {}
    processed_b: Dict[int, Any] = {}
    # local index of each valid feature within its centroid matrix (Stage-1 returns local indices)
    local_index_a = {int(f): i for i, f in enumerate(valid_a_list)}
    local_index_b = {int(f): i for i, f in enumerate(valid_b_list)}

    for side, (f_list, acts_data, centroids, processed) in enumerate([
        (valid_a_list, state["acts_data_a"], centroids_a, processed_a),
        (valid_b_list, state["acts_data_b"], centroids_b, processed_b),
    ]):
        desc = "SAE A" if side == 0 else "SAE B"
        for idx, f_id in tqdm(enumerate(f_list), total=len(f_list), desc=f"Transforming {desc} (K={K})"):
            toks = acts_data[f_id]['toks']
            weights = acts_data[f_id]['weights']
            if not toks:
                continue
            if K is not None:
                toks = toks[:K]
                weights = weights[:K]

            X_raw = all_latents[toks].float()
            V = X_raw
            W = t.tensor(weights, dtype=t.float32)

            centroids[idx] = (V.to(device) * W.to(device).unsqueeze(1)).sum(dim=0) / W.sum().to(device)

            W_np = (W / W.sum()).numpy().astype(np.float64)
            W_np[-1] = 1.0 - np.sum(W_np[:-1])
            processed[f_id] = {'X': V, 'W': W_np, 'W_t': W / W.sum()}

    return {
        "centroids_a": centroids_a,
        "centroids_b": centroids_b,
        "processed_a": processed_a,
        "processed_b": processed_b,
        "local_index_a": local_index_a,
        "local_index_b": local_index_b,
        "K": K,
    }


# =============================================================================================
# Sections 5-6: Stage-1 coarse centroid search + Stage-2 OT rerank, restricted to `query_features`.
# Returns, per query, the candidate B features in *centroid-rank order* with their OT distances, so
# any M <= M_pool is a prefix and the top-1 match at each M is argmin over that prefix. Wall-clock
# is snapshotted at the requested candidate-count checkpoints from a single pass.
# =============================================================================================
@t.no_grad()
def rerank_queries(
    state: Dict[str, Any],
    clouds: Dict[str, Any],
    query_features: Sequence[int],
    M: int,
    device: str = "cuda",
    chunk_size: int = 2048,
    stage1_metric: str = "euclidean",
    ot_method: str = "sinkhorn",
    time_checkpoints: Optional[Sequence[int]] = None,
    candidate_pool_ids: Optional[Sequence[int]] = None,
    sinkhorn_impl: str = "fixed",
) -> Dict[str, Any]:
    """One rerank pass over the top-`M` centroid neighbours for each query feature.

    `candidate_pool_ids` restricts the Stage-1 centroid universe to a subset of side-B features
    (e.g. a random pool). With it, "top-M" means the top-M centroid neighbours *within that pool*,
    which is exactly the top-50 prefilter path of the random-pool ablation. None = all valid B.

    Output:
      per_query[f_a] = [(f_b, ot_distance), ...]   # in centroid-rank order, length <= M
      stage1_s                                     # centroid search wall-clock (whole valid_a)
      stage2_cum_s[c]                              # cumulative OT wall-clock to process the first c
                                                   #   candidates of every query (c in time_checkpoints)
      n_candidates                                 # min(M, pool size)
    """
    valid_b_list = state["valid_b_list"]
    centroids_a = clouds["centroids_a"].clone()
    centroids_b = clouds["centroids_b"].clone()
    processed_a = clouds["processed_a"]
    processed_b = clouds["processed_b"]
    local_index_a = clouds["local_index_a"]
    local_index_b = clouds["local_index_b"]

    if candidate_pool_ids is not None:
        pool_local = np.asarray([local_index_b[int(f)] for f in candidate_pool_ids], dtype=np.int64)
        n_pool = len(pool_local)
    else:
        pool_local = None
        n_pool = len(valid_b_list)

    M = int(min(M, n_pool))
    if time_checkpoints is None:
        time_checkpoints = [M]
    checkpoints = sorted({int(min(c, M)) for c in time_checkpoints})

    # --- Section 5: Stage-1 coarse search (metric normalisation copied verbatim) ---
    stage1_t0 = time.perf_counter()
    if stage1_metric == "cosine":
        centroids_a = t.nn.functional.normalize(centroids_a, p=2, dim=1)
        centroids_b = t.nn.functional.normalize(centroids_b, p=2, dim=1)
    elif stage1_metric == "euclidean":
        max_norm_a = t.norm(centroids_a, p=2, dim=1).max()
        max_norm_b = t.norm(centroids_b, p=2, dim=1).max()
        global_max_norm = t.clamp(t.max(max_norm_a, max_norm_b), min=1e-9)
        centroids_a = centroids_a / global_max_norm
        centroids_b = centroids_b / global_max_norm

    # Scaling above is uniform, so the centroid ranking is identical whether we search all of B or a
    # pool -- restricting the columns only changes which candidates are visible, not their order.
    centroids_b_search = (centroids_b if pool_local is None
                          else centroids_b[t.from_numpy(pool_local).to(centroids_b.device)])

    # Only the query rows are searched (the stock code searches all of valid_a).
    query_local = [local_index_a[int(f)] for f in query_features]
    query_rows = centroids_a[query_local]

    top_M_indices_b = []
    for i in range(0, len(query_rows), chunk_size):
        chunk_a = query_rows[i:i + chunk_size]
        if stage1_metric == "cosine":
            scores = t.matmul(chunk_a, centroids_b_search.T)
            _, top_m_idx = t.topk(scores, M, dim=1, largest=True)
        elif stage1_metric == "euclidean":
            distances = t.cdist(chunk_a, centroids_b_search, p=2)
            _, top_m_idx = t.topk(distances, M, dim=1, largest=False)
        top_M_indices_b.append(top_m_idx.cpu())
    top_M_indices_b = t.cat(top_M_indices_b, dim=0).numpy()
    if pool_local is not None:
        top_M_indices_b = pool_local[top_M_indices_b]   # pool-space idx -> global valid_b local idx
    if device.startswith("cuda"):
        t.cuda.synchronize()
    stage1_s = time.perf_counter() - stage1_t0

    # --- Section 6: Stage-2 OT rerank (query-restricted; OT body copied verbatim) ---
    gpu_solver = None
    if ot_method == "sinkhorn":
        gpu_solver = SamplesLoss(loss="sinkhorn", p=2, blur=0.01, scaling=0.9)

    per_query: Dict[int, List[Any]] = {}
    stage2_cum_s = {c: 0.0 for c in checkpoints}

    for i, f_id_a in enumerate(tqdm(query_features, desc=f"OT rerank (M={M})")):
        if f_id_a not in processed_a:
            continue
        candidates_b_local_idx = top_M_indices_b[i]
        X_a = processed_a[f_id_a]['X']

        # One entry per centroid position (None placeholder for the rare invalid candidate) so that
        # `ordered[:m]` is exactly the top-m centroid shortlist -- what the stock prefilter reranks.
        ordered: List[Any] = []
        q_start = time.perf_counter()
        next_cp = 0
        for pos, local_idx in enumerate(candidates_b_local_idx):
            f_id_b = valid_b_list[local_idx]
            if f_id_b in processed_b:
                X_b = processed_b[f_id_b]['X']
                if ot_method == "exact":
                    cost = cdist(X_a.numpy(), X_b.numpy(), metric='euclidean')  # type: ignore
                    dist = float(ot.emd2(processed_a[f_id_a]['W'], processed_b[f_id_b]['W'], cost))  # type: ignore
                else:
                    dist = SINKHORN_IMPLS[sinkhorn_impl](
                        X_a.to(device), processed_a[f_id_a]['W_t'].to(device),
                        X_b.to(device), processed_b[f_id_b]['W_t'].to(device),
                        gpu_solver,
                    )
                ordered.append((int(f_id_b), dist))
            else:
                ordered.append((None, float("inf")))
            # snapshot cumulative time whenever we cross a checkpoint boundary (by candidates seen)
            while next_cp < len(checkpoints) and (pos + 1) >= checkpoints[next_cp]:
                if device.startswith("cuda"):
                    t.cuda.synchronize()
                stage2_cum_s[checkpoints[next_cp]] += time.perf_counter() - q_start
                next_cp += 1

        per_query[int(f_id_a)] = ordered

    return {
        "per_query": per_query,
        "stage1_s": stage1_s,
        "stage2_cum_s": stage2_cum_s,
        "n_candidates": M,
    }


# ---------------------------------------------------------------------------------------------
# Helper turning a rerank result into top-1 matches at shortlist size M.
# ---------------------------------------------------------------------------------------------
def _best_pair_within(ordered: List[Any], m: int):
    """(feat_b, ot_distance) of the OT-argmin over the first `m` centroid positions; None if empty."""
    valid = [(fb, d) for fb, d in ordered[:m] if fb is not None]
    if not valid:
        return None
    return min(valid, key=lambda fb_d: fb_d[1])


def assignments_at_M(per_query: Dict[int, List[Any]], m: int) -> Dict[int, List[Dict[str, Any]]]:
    """Stock alignment format {feat_a: [{'target_feature': feat_b}]} using the top-1 match at pool m."""
    out: Dict[int, List[Dict[str, Any]]] = {}
    for f_a, ordered in per_query.items():
        best = _best_pair_within(ordered, m)
        if best is not None:
            out[int(f_a)] = [{"target_feature": int(best[0]), "ot_distance": float(best[1])}]
    return out
