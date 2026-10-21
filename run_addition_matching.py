"""Two-digit addition case study: match features to each other *within* a single SAE layer.

Builds the "Calc:p1+p2=" dataset, caches SAE activations for it, keeps the features whose
last-token firing count falls inside --act_freq_range, and runs each intra-SAE matching method
from `feature_matching_intra/` over that pool.

Output: addition_matching_<hook>_<method>_<k>_<range>_<freq_range>_<model>_<sae>.pt
Launcher: scripts/run_addition_matching.sh
"""

import argparse
import os

from Evaluation_Utils import *
from math import log2
import einops
from functools import partial
from addition_data import MathAdditionDataset, plot_feature_activations
from feature_matching_intra.ot_matching import *
from feature_matching_intra.naive_matching import *
from feature_matching_intra.mech_permute_matching import *
from feature_matching_intra.feat_flow_matching import *
from feature_matching_intra.llm_matching import *

import exp_config

METHODS = ["ot", "naive", "feat_flow", "mech_permute", "llm"]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run intra-SAE feature matching on the two-digit addition dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    exp_config.add_model_args(parser, default_preset="gemma_mlp", default_device_id=5)

    group = parser.add_argument_group("addition matching")
    group.add_argument("--method_names", type=str, nargs="+", default=["llm"], choices=METHODS,
                       help="Matching method(s) to run against the same loaded model.")
    group.add_argument("--sae_layer", type=int, default=19, help="Layer whose SAE features are matched.")
    group.add_argument("--k", type=int, default=1, help="Matches kept per feature.")
    group.add_argument("--batch_size", type=int, default=32, help="Batch size for activation extraction.")
    group.add_argument("--data_range", type=int, nargs=2, default=[0, 9], metavar=("MIN_VAL", "MAX_VAL"),
                       help="Operand range for the addition dataset (inclusive).")
    group.add_argument("--act_freq_range", type=int, nargs=2, default=[5, 50], metavar=("LO", "HI"),
                       help="Keep features whose last-token firing count is within this range.")
    group.add_argument("--llm_chunk_size", type=int, default=20, help="Features per LLM request.")

    cache_group = parser.add_argument_group("activation cache")
    cache_group.add_argument("--cache_dir", type=str, default="data/cache_data",
                             help="Directory holding the cached addition_*.pt activations.")
    cache_group.add_argument("--full_data_cache", type=str, default=None,
                             help="Explicit cache path; derived from --cache_dir when omitted.")

    parser.add_argument("--output_dir", type=str, default=".", help="Where the matching .pt files are written.")

    return parser.parse_args(argv)


def load_or_build_addition_data(args, cfg, model, list_saes, dataset):
    """Cached `(data_toks, full_data_dict)` for the addition prompts."""
    min_val, max_val = args.data_range
    if args.full_data_cache is not None:
        cache_path = args.full_data_cache
    else:
        cache_path = os.path.join(
            args.cache_dir,
            f"addition_{cfg.model_name_flat}_{cfg.sae_type_flat}_{min_val}_{max_val}.pt",
        )

    if not os.path.exists(cache_path):
        print(f"[run_addition_matching] no cache at {cache_path}; extracting activations")
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        data_toks = extract_data(data_loader, num_batches=len(data_loader)).squeeze(0)

        full_data_dict = get_sae_activation_from_tok(
            {sae.cfg.hook_name: sae for sae in list_saes},
            model,
            data_toks,
            args.batch_size,
            None,
            False,
        )
        exp_config.ensure_parent_dir(cache_path)
        t.save((data_toks, full_data_dict), cache_path)
    else:
        print(f"[run_addition_matching] loading cached activations from {cache_path}")
        data_toks, full_data_dict = t.load(cache_path, map_location="cpu")

    return data_toks, full_data_dict


def run_method(args, method, model, sae, data_toks, filtered_unique):
    """One intra-SAE matching run; bodies are unchanged from the original."""
    if method == "ot":
        return match_intra_sae_features_ot(
            data_toks=data_toks,
            model=model,
            sae=sae,
            feature_pool=filtered_unique,
            k=args.k,
            batch_size=args.batch_size,
            caching_hook=sae.cfg.hook_name
        )
    if method == "naive":
        return match_intra_sae_features_naive(
            data_toks=data_toks,
            model=model,
            sae=sae,
            feature_pool=filtered_unique,
            k=args.k,
            batch_size=args.batch_size,
            caching_hook=sae.cfg.hook_name
        )
    if method == "feat_flow":
        return match_intra_sae_features_feat_flow(
            sae=sae,
            feature_pool=filtered_unique,
            k=args.k,
        )
    if method == "mech_permute":
        return match_intra_sae_features_mech_permute(
            sae=sae,
            feature_pool=filtered_unique,
            k=args.k,
        )
    if method == "llm":
        return match_intra_sae_features_llm(
            data_toks=data_toks,
            model=model,
            sae=sae,
            # gpt-oss-120b via call_llm_for_clustering (LLMSEL_URL / LLMSEL_API_KEY).
            api_key=os.getenv("LLMSEL_API_KEY") or os.getenv("UF_AI_API_KEY") or API_KEY,  # type: ignore
            feature_pool=filtered_unique,
            k=args.k,
            batch_size=args.batch_size,
            chunk_size=args.llm_chunk_size,
        )
    raise ValueError(f"Unknown method: {method}")


def main(args: argparse.Namespace) -> None:
    cfg = exp_config.ModelConfig(args)
    device = cfg.device

    model, dict_saes, list_saes = exp_config.load_model_and_saes(cfg)

    min_val, max_val = args.data_range
    dataset = MathAdditionDataset(model, min_val=min_val, max_val=max_val)

    data_toks, full_data_dict = load_or_build_addition_data(args, cfg, model, list_saes, dataset)

    sae = list_saes[args.sae_layer]
    indices = full_data_dict[sae.cfg.hook_name][1][:, -1, :]
    values = full_data_dict[sae.cfg.hook_name][0][:, -1, :]

    pad_value = sae.cfg.d_sae
    indices[values < 1e-3] = pad_value
    unique_indices = indices.unique()

    feature_counts = t.bincount(indices.flatten(), minlength=sae.cfg.d_sae + 1)
    feature_counts[pad_value] = -1  # remove frequency of pad_indice

    activation_feq = feature_counts[unique_indices]

    freq_lo, freq_hi = args.act_freq_range
    filtered_unique = unique_indices[(activation_feq >= freq_lo) & (activation_feq <= freq_hi)]
    print(f"[run_addition_matching] {len(filtered_unique)} features in freq range [{freq_lo}, {freq_hi}]")

    os.makedirs(args.output_dir, exist_ok=True)

    for method in args.method_names:
        print(f"Running {method} matching...")
        matches = run_method(args, method, model, sae, data_toks, filtered_unique)

        file_name = (
            f"addition_matching_{sae.cfg.hook_name}_{method}_{args.k}_{min_val}_{max_val}_"
            f"{freq_lo}_{freq_hi}_{cfg.model_name_flat}_{cfg.sae_type_flat}.pt"
        )
        out_path = os.path.join(args.output_dir, file_name)
        t.save(matches, out_path)
        print(f"[run_addition_matching] saved {out_path}")


if __name__ == "__main__":
    main(parse_args())
