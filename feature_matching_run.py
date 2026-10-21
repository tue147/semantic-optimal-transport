"""Produce cross-layer SAE feature matchings and save them to `results/<model>_<sae_type>/`.

One method per invocation (or several via `--method_name a b c`, which reuses the loaded model
and activation cache). Output file names encode the hyperparameters and are unchanged:

    ot            -> {method}_{a}_{b}_{k}_{M}_{metric}_{max_points_per_feature}_{max_density}.pt
    naive         -> {method}_{a}_{b}_{k}_{metric}_{max_density}.pt
    others        -> {method}_{a}_{b}_{k}_{max_density}.pt

Launcher: scripts/feature_matching_run.sh
"""

import argparse
import os

from Evaluation_Utils import *
from math import log2
import einops
from functools import partial
from feature_matching.ot_matching import *
from feature_matching.mech_permute import *
from feature_matching.feature_flow import *
from feature_matching.naive_matching import *

import exp_config

METHODS = ["ot", "mech_permute", "feat_flow", "naive", "mech_permute_ml", "feat_flow_ml"]


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run cross-layer SAE feature matching and save the alignments.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    exp_config.add_model_args(parser, default_preset="gemma_mlp", default_device_id=0)
    exp_config.add_data_args(parser, default_batchsize=64, default_max_length=128, default_num_batches=10000)

    group = parser.add_argument_group("matching")
    group.add_argument("--method_name", type=str, nargs="+", default=["ot"], choices=METHODS,
                       help="Matching method(s) to run against the same loaded model.")
    group.add_argument("--sae_a_id", type=int, default=25, help="Target (later) layer; must exceed --sae_b_id.")
    group.add_argument("--sae_b_id", type=int, default=12, help="Source (earlier) layer.")
    group.add_argument("--k", type=int, default=25, help="Number of candidate matches kept per feature.")
    group.add_argument("--max_density", type=float, default=0.1, help="Drop features firing on more than this fraction.")
    group.add_argument("--batch_size", type=int, default=32, help="Batch size for the matching forward passes.")
    group.add_argument("--chunk_size", type=int, default=2048, help="Feature chunk size for the pairwise stage.")
    group.add_argument("--metric", type=str, default="euclidean", help="Stage-1 metric for ot / naive.")

    ot_group = parser.add_argument_group("ot only")
    ot_group.add_argument("--M", type=int, default=25, help="Rerank shortlist size.")
    ot_group.add_argument("--max_points_per_feature", type=int, default=500, help="Support size per feature measure.")
    ot_group.add_argument("--ot_method", type=str, default="sinkhorn", help="OT solver.")
    ot_group.add_argument("--sinkhorn_impl", type=str, default="old", choices=["old", "fixed"],
                          help="'old' = the Sinkhorn helper behind the paper's numbers; 'fixed' = the "
                               "corrected true-W_2 helper (see feature_matching/utils.py).")
    ot_group.add_argument("--caching_hook", type=exp_config.optional_str, default=None,
                          help="Reference space R. None = layer A's SAE hook. The paper's GPT2 setups with "
                               "target layer 11 used 'ln_final.hook_normalized'.")

    mp_group = parser.add_argument_group("mech_permute only")
    mp_group.add_argument("--use_lap", type=exp_config.str_to_bool, default=True,
                          help="Solve the assignment with a linear assignment problem.")

    parser.add_argument("--results_dir", type=str, default=None,
                        help="Output directory; defaults to results/<model>_<sae_type>.")

    return parser.parse_args(argv)


def run_method(args, method_name, model, list_saes, data_toks, device, cfg):
    """One matching run. Returns `(final_alignments, file_name)`; the bodies are unchanged."""
    sae_a = list_saes[args.sae_a_id]
    sae_b = list_saes[args.sae_b_id]

    k = args.k
    max_density = args.max_density
    batch_size = args.batch_size
    chunk_size = args.chunk_size

    if method_name == "ot":
        M = args.M
        max_points_per_feature = args.max_points_per_feature
        metric = args.metric
        final_alignments = align_saes_retrieve_and_rerank(
            data_toks=data_toks,
            sae_a=sae_a,
            sae_b=sae_b,
            model=model,
            M=M,
            k=k,
            caching_hook=args.caching_hook,
            max_density=max_density,
            max_points_per_feature=max_points_per_feature,
            batch_size=batch_size,
            device=device,  # type: ignore
            chunk_size=chunk_size,
            stage1_metric=metric,
            ot_method=args.ot_method,
            sinkhorn_impl=args.sinkhorn_impl,
        )
        file_name = f"{method_name}_{args.sae_a_id}_{args.sae_b_id}_{k}_{M}_{metric}_{max_points_per_feature}_{max_density}.pt"
    elif method_name == "mech_permute":
        final_alignments = align_saes_mechanistic_match(
            data_toks=data_toks,
            sae_a=sae_a,
            sae_b=sae_b,
            model=model,
            k=k,
            max_density=max_density,
            batch_size=batch_size,
            device=device,  # type: ignore
            chunk_size=chunk_size,
            use_lap=args.use_lap,
        )
        file_name = f"{method_name}_{args.sae_a_id}_{args.sae_b_id}_{k}_{max_density}.pt"
    elif method_name == "feat_flow":
        final_alignments = align_saes_feature_flow_match(
            data_toks=data_toks,
            sae_a=sae_a,
            sae_b=sae_b,
            model=model,
            k=k,
            max_density=max_density,
            batch_size=batch_size,
            device=device,  # type: ignore
            chunk_size=chunk_size,
        )
        file_name = f"{method_name}_{args.sae_a_id}_{args.sae_b_id}_{k}_{max_density}.pt"
    elif method_name == "naive":
        metric = args.metric
        final_alignments = align_saes_centroid_match(
            data_toks=data_toks,
            sae_a=sae_a,
            sae_b=sae_b,
            model=model,
            k=k,
            caching_hook=None,
            max_density=max_density,
            batch_size=batch_size,
            metric=metric,
            device=device,  # type: ignore
            chunk_size=chunk_size,
        )
        file_name = f"{method_name}_{args.sae_a_id}_{args.sae_b_id}_{k}_{metric}_{max_density}.pt"
    elif method_name == "mech_permute_ml":
        final_alignments = align_mechanistic_permute_multilayer(
            data_toks=data_toks,
            sae_list=list_saes[args.sae_b_id:args.sae_a_id + 1],
            model=model,
            max_density=max_density,
            batch_size=batch_size,
            device=device,  # type: ignore
            chunk_size=chunk_size,
        )
        file_name = f"{method_name}_{args.sae_a_id}_{args.sae_b_id}_{k}_{max_density}.pt"
    elif method_name == "feat_flow_ml":
        final_alignments = align_feature_flow_multilayer(
            data_toks=data_toks,
            sae_list=list_saes[args.sae_b_id:args.sae_a_id + 1],
            model=model,
            k=k,
            max_density=max_density,
            batch_size=batch_size,
            device=device,  # type: ignore
            chunk_size=chunk_size,
        )
        file_name = f"{method_name}_{args.sae_a_id}_{args.sae_b_id}_{k}_{max_density}.pt"
    else:
        raise ValueError(f"Unknown method: {method_name}")

    return final_alignments, file_name


def main(args: argparse.Namespace) -> None:
    assert args.sae_a_id > args.sae_b_id, "sae_a_id must be the later layer"

    cfg = exp_config.ModelConfig(args)
    device = cfg.device

    model, dict_saes, list_saes = exp_config.load_model_and_saes(cfg)

    from sae.Load_Data import load_data
    from sae.SAE_Trainer import DataConfig

    # Only the token pool is needed here -- every matching method rebuilds its own activation cache
    # from data_toks, and nothing in this script reads full_data_dict. Extracting the tokens
    # directly (the way the eval scripts do) avoids holding load_or_build_full_data's large per-SAE
    # activation dict resident for the whole run, which was pushing host RAM over the SLURM --mem
    # limit (OOM).
    data_config = DataConfig(
        data_path=args.data_path,
        batch_size=1,
        max_length=args.max_length,
    )
    _, data_loader = load_data(cfg.tok_name, data_config)
    data_toks = extract_data(data_loader, num_batches=args.num_batches)

    results_dir = args.results_dir if args.results_dir is not None else cfg.results_dir
    os.makedirs(results_dir, exist_ok=True)

    for method_name in args.method_name:
        print(f"[feature_matching_run] running {method_name} for layers {args.sae_a_id} -> {args.sae_b_id}")
        final_alignments, file_name = run_method(args, method_name, model, list_saes, data_toks, device, cfg)

        out_path = os.path.join(results_dir, file_name)
        t.save(final_alignments, out_path)
        print(f"[feature_matching_run] saved {out_path}")


if __name__ == "__main__":
    main(parse_args())
