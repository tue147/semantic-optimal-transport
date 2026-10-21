import torch as t
from torch import Tensor
import numpy as np
import ot
from typing import Dict, List, Any, Optional, Union


def compute_sinkhorn_ot_gpu_old(X_a: t.Tensor, W_a: t.Tensor, X_b: t.Tensor, W_b: t.Tensor, solver: Any) -> float:
    """
    Computes approximated Optimal Transport (Sinkhorn) on the GPU using a
    Max-Norm scaling trick to prevent GeomLoss divergence errors while
    preserving exact Euclidean geometry.

    REFERENCE IMPLEMENTATION, kept verbatim: every published result was produced with it. Its
    un-scaling is off by one power of `global_max_norm` -- see `compute_sinkhorn_ot_gpu` below for
    the derivation and the correction. Only call this to reproduce old numbers.
    """
    # 1. Clean data to ensure no float16 NaNs/Infs leaked
    X_a_f = t.nan_to_num(X_a.float(), nan=0.0, posinf=0.0, neginf=0.0)
    X_b_f = t.nan_to_num(X_b.float(), nan=0.0, posinf=0.0, neginf=0.0)
    # If a feature only fired once, duplicate the point and halve the weight
    if X_a_f.shape[0] == 1:
        X_a_f = X_a_f.repeat(2, 1)
        W_a = W_a.repeat(2) / 2.0

    if X_b_f.shape[0] == 1:
        X_b_f = X_b_f.repeat(2, 1)
        W_b = W_b.repeat(2) / 2.0

    # Inject microscopic jitter to prevent diameter == 0.0 (-inf log crash)
    X_a_f = X_a_f + t.randn_like(X_a_f) * 1e-6
    X_b_f = X_b_f + t.randn_like(X_b_f) * 1e-6

    # 2. Find the absolute maximum L2 norm across BOTH point clouds
    max_norm_a = t.norm(X_a_f, p=2, dim=1).max()
    max_norm_b = t.norm(X_b_f, p=2, dim=1).max()
    global_max_norm = t.max(max_norm_a, max_norm_b)

    # Clamp to avoid division by zero in empty edge cases
    global_max_norm = t.clamp(global_max_norm, min=1e-9)

    # 3. Scale the entire space down.
    # All vectors now have a magnitude <= 1, but relative geometry is perfectly preserved.
    X_a_scaled = X_a_f / global_max_norm
    X_b_scaled = X_b_f / global_max_norm

    # 4. Normalize probability weights (must sum exactly to 1.0)
    W_a_norm = (W_a / W_a.sum()).float()
    W_b_norm = (W_b / W_b.sum()).float()

    # 5. Compute the scaled OT distance
    scaled_ot_dist = solver(W_a_norm, X_a_scaled, W_b_norm, X_b_scaled)

    # 6. Un-scale the distance
    # Because W_2(X/M, Y/M) = (1/M) * W_2(X, Y), multiplying by M restores the true distance
    true_ot_dist = scaled_ot_dist * global_max_norm

    return float(true_ot_dist.item())


# =================================================================================================
# What `compute_sinkhorn_ot_gpu_old` got wrong.
#
# geomloss `SamplesLoss(loss="sinkhorn", p=2)` does NOT return W_2. Its p=2 ground cost is the
# *halved squared* Euclidean distance (geomloss/sinkhorn_samples.py: `squared_distances(x,y)/2`),
# so the solver returns ~ 0.5 * W_2^2, which scales as 1/s^2 under x -> x/s, not 1/s. Multiplying
# by s once leaves a residual, so the value actually being ranked is
#
#     0.5 * W_2^2(X_a, X_b) / max(||X_a||_max, ||X_b||_max)
#
# and `s` is recomputed per candidate pair, so that is not a monotone transform of W_2 across a
# candidate list. For a fixed query, any candidate holding a higher-norm point than the query's has
# its distance divided by a larger number -- a one-sided discount that grows with the candidate's
# max norm. A cloud's max norm is a max over its support points, so it grows with how many points
# the feature has, and dense features (support at the `max_points_per_feature` cap) are
# systematically favoured. That is the suspected driver of the hub collapse seen in the M-scaling
# ablation; `diagnose_ot_scaling.py` measures how much of it this accounts for.
#
# The correction is the right exponent. `scale` additionally lets the caller pin ONE scale for
# every pair (e.g. the largest point norm over the whole layer), which also makes the entropic
# regularisation uniform: `blur` lives in scaled units, so a per-pair `s` means an effective blur
# of `blur * s` that varies from candidate to candidate as well.
#
# NOTE ON UNITS: this returns a true W_p, where the old function returned 0.5*W_2^2/s. Rankings are
# what the matchers consume, but the raw `ot_distance` numbers stored in results/*.pt are on a
# different scale than before -- don't compare them across the fix.
# =================================================================================================
def compute_sinkhorn_ot_gpu(
    X_a: t.Tensor,
    W_a: t.Tensor,
    X_b: t.Tensor,
    W_b: t.Tensor,
    solver: Any,
    scale: Optional[Union[float, t.Tensor]] = None,
) -> float:
    """Sinkhorn OT with the max-norm scaling correctly undone; returns a true W_p distance.

    Identical to `compute_sinkhorn_ot_gpu_old` in cleaning, the single-point duplication hack, the
    jitter and the solver call -- only the scale choice and the un-scaling differ.

    Args:
        scale: the value both clouds are divided by before the solve. `None` (default) keeps the
            old per-pair `max(||X_a||_max, ||X_b||_max)`, so the only change is the exponent.
            Passing a fixed number makes the scaling uniform across candidates, which is what
            removes the ranking distortion entirely.

    Returns:
        W_p(X_a, X_b) in the ORIGINAL units (p read off `solver.p`, so the value is comparable
        across every choice of `scale`).
    """
    # 1. Clean data to ensure no float16 NaNs/Infs leaked
    X_a_f = t.nan_to_num(X_a.float(), nan=0.0, posinf=0.0, neginf=0.0)
    X_b_f = t.nan_to_num(X_b.float(), nan=0.0, posinf=0.0, neginf=0.0)
    # If a feature only fired once, duplicate the point and halve the weight
    if X_a_f.shape[0] == 1:
        X_a_f = X_a_f.repeat(2, 1)
        W_a = W_a.repeat(2) / 2.0

    if X_b_f.shape[0] == 1:
        X_b_f = X_b_f.repeat(2, 1)
        W_b = W_b.repeat(2) / 2.0

    # Inject microscopic jitter to prevent diameter == 0.0 (-inf log crash)
    X_a_f = X_a_f + t.randn_like(X_a_f) * 1e-6
    X_b_f = X_b_f + t.randn_like(X_b_f) * 1e-6

    # 2. Pick the scale: caller-supplied (uniform across candidates) or the per-pair max norm.
    if scale is None:
        max_norm_a = t.norm(X_a_f, p=2, dim=1).max()
        max_norm_b = t.norm(X_b_f, p=2, dim=1).max()
        s = t.max(max_norm_a, max_norm_b)
    else:
        s = t.as_tensor(scale, dtype=X_a_f.dtype, device=X_a_f.device)
    # Clamp to avoid division by zero in empty edge cases
    s = t.clamp(s, min=1e-9)

    # 3. Scale the space down so GeomLoss sees an O(1) diameter.
    X_a_scaled = X_a_f / s
    X_b_scaled = X_b_f / s

    # 4. Normalize probability weights (must sum exactly to 1.0)
    W_a_norm = (W_a / W_a.sum()).float()
    W_b_norm = (W_b / W_b.sum()).float()

    # 5. Compute the scaled loss
    scaled_loss = solver(W_a_norm, X_a_scaled, W_b_norm, X_b_scaled)

    # 6. Un-scale, accounting for what the solver's ground cost actually is.
    p = int(getattr(solver, "p", 2))
    if p == 2:
        # cost = 0.5*|x-y|^2  ->  loss ~ 0.5 * W_2^2 / s^2. The debiased Sinkhorn divergence can
        # come out microscopically negative, so clamp before the sqrt.
        true_dist = s * t.sqrt(2.0 * t.clamp(scaled_loss, min=0.0))
    elif p == 1:
        # cost = |x-y|  ->  loss ~ W_1 / s
        true_dist = s * scaled_loss
    else:
        raise ValueError(f"compute_sinkhorn_ot_gpu: unsupported solver.p={p} (expected 1 or 2)")

    return float(true_dist.item())


def cloud_max_norm(*processed: Dict[int, Any]) -> float:
    """Largest point norm over one or more `processed` cloud dicts ({f_id: {'X': (n, d), ...}}).

    The natural argument for `compute_sinkhorn_ot_gpu`'s `scale`: dividing by it puts every cloud
    in the unit ball with a single, candidate-independent factor.
    """
    m = 0.0
    for proc in processed:
        for entry in proc.values():
            m = max(m, float(entry["X"].norm(dim=1).max()))
    return m
