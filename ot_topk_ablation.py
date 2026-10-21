"""Appendix D ablation: OT matching quality vs the support size K (max_points_per_feature).

Each feature is an activation-weighted empirical distribution over the hidden states where it fires;
K caps how many of the top-activating points make up that distribution. At K=1 the measure collapses
to a single (top-activating) hidden vector and the OT distance reduces to the distance between two
points.

Everything else is fixed at the main-experiment defaults (M=25 shortlist, top-1 match, euclidean
Stage 1, Sinkhorn). Activations are extracted ONCE at the largest K and the point clouds are rebuilt
by truncation for each smaller K -- no re-extraction. With --query_prev_file the 100 query features of
Table 1 are reused for every seed and K, so seeds only resample the LLM judge. Metrics per (K, seed):
LLM eval score + accuracy, aggregated over seeds (mean / std / sem / 90% CI).

WARNING: fans out paid LLM-judge calls. Launcher: scripts/ot_topk_ablation.sh
"""

import argparse
import asyncio
import csv
import os
import random

import numpy as np
import torch as t

from Evaluation_Utils import *
from test_matching.util import extract_pair
from test_matching.llm_eval import llm_matching_eval
from feature_matching.ot_matching_ablation import (
    extract_ot_state,
    build_clouds,
    rerank_queries,
    assignments_at_M,
)

import exp_config


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="OT feature-matching support-size (K) ablation (Appendix D).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    exp_config.add_model_args(parser, default_preset="gemma_mlp", default_device_id=0)
    exp_config.add_data_args(parser, default_batchsize=64, default_max_length=128, default_num_batches=10000)
    exp_config.add_judge_args(parser, default_judge="gpt-oss-120b")

    group = parser.add_argument_group("ablation")
    group.add_argument("--pairs", type=str, nargs="+", default=["13,12", "25,12", "25,0"],
                       help="Layer pairs 'A,B' (A later).")
    group.add_argument("--K_grid", type=int, nargs="+", default=[1, 16, 256, 500],
                       help="Support sizes (max_points_per_feature) to sweep.")
    group.add_argument("--number_of_pairs", type=int, default=100, help="Query features sampled per seed.")
    group.add_argument("--min_activation_examples", type=int, default=30,
                       help="A query feature must fire on more than this many tokens (non-dead).")
    group.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])

    ot_group = parser.add_argument_group("ot")
    ot_group.add_argument("--M", type=int, default=25, help="Stage-1 shortlist size.")
    ot_group.add_argument("--max_density", type=float, default=0.1)
    ot_group.add_argument("--metric", type=str, default="euclidean")
    ot_group.add_argument("--ot_method", type=str, default="sinkhorn")
    ot_group.add_argument("--caching_hook", type=str, default=None,
                          help="Reference space R for the activation clouds. None = layer A's SAE hook. "
                               "The GPT-2 setups with target layer 11 used 'ln_final.hook_normalized'.")
    ot_group.add_argument("--sinkhorn_impl", type=str, default="old", choices=["fixed", "old"],
                          help="'old' = compute_sinkhorn_ot_gpu_old, the helper behind the paper's numbers; "
                               "'fixed' = the corrected true-W_2 helper (see feature_matching/utils.py).")
    ot_group.add_argument("--query_prev_file", type=str, default=None,
                          help="feature_matching_test.py run file with '{a}' and '{b}' placeholders "
                               "(e.g. 'ot_{a}_{b}_1_100_30_run1.pt'), looked up in results/<model>_<sae>/ "
                               "and then the output dir. Its matching_details give the query "
                               "features, reused for every seed and K (the paper protocol). None = sample "
                               "fresh non-dead queries per seed.")
    ot_group.add_argument("--chunk_size", type=int, default=2048)
    ot_group.add_argument("--batch_size", type=int, default=96)

    parser.add_argument("--results_dir", type=str, default=None,
                        help="Output dir; default results/<model>_<sae>_<judge>.")
    parser.add_argument("--out_prefix", type=str, default="ottopk")
    parser.add_argument("--save_matches", type=exp_config.str_to_bool, default=True)

    return parser.parse_args(argv)


def accuracy_from_details(res) -> float:
    """Accuracy = fraction of judged pairs scored 3 (perfect match)."""
    details = res["matching_details"]
    if not details:
        return 0.0
    return float(np.mean([1.0 if d["score"] == 3 else 0.0 for d in details]))


def eligible_queries(state, feature_counts, min_activation_examples):
    """Valid (non-dead, non-high-density) side-A features that fire on > min_activation_examples tokens."""
    return [int(f) for f in state["valid_a_list"] if feature_counts[int(f)] > min_activation_examples]


def evaluate_matches(model, data_toks, assignments, caches, judge, query_order=None):
    """LLM eval (score + accuracy) of one K setting's top-1 matches.

    `query_order` (optional) is the query list to judge, duplicates included -- feature_matching_test.py
    judges one pair per entry of its (with-replacement) query list, so a repeated query is judged
    twice. None = one pair per distinct query.
    """
    indices_a, values_a, indices_b, values_b = caches
    all_feat_pairs = []
    for feat_a in (assignments.keys() if query_order is None else query_order):
        all_feat_pairs.extend(extract_pair(feat_a, assignments, top=1))
    assert len(all_feat_pairs) > 0

    res = asyncio.run(llm_matching_eval(
        all_feat_pairs, model, data_toks, indices_a, values_a, indices_b, values_b, judge,
    ))
    out = {
        "llm_eval_score": res["overall_score"],
        "llm_eval_accuracy": accuracy_from_details(res),
        "n_pairs": len(all_feat_pairs),
    }
    return out, all_feat_pairs


def run_pair(args, cfg, model, list_saes, data_toks, sae_a_id, sae_b_id, judge, device, output_dir):
    sae_a = list_saes[sae_a_id]
    sae_b = list_saes[sae_b_id]
    K_grid = sorted(set(int(k) for k in args.K_grid))
    K_max = max(K_grid)

    print(f"\n[ot_topk_ablation] ===== pair A={sae_a_id} B={sae_b_id} =====")
    state = extract_ot_state(
        data_toks=data_toks, sae_a=sae_a, sae_b=sae_b, model=model, caching_hook=args.caching_hook,
        max_density=args.max_density, max_points_per_feature=K_max,
        batch_size=args.batch_size, device=str(device),
    )
    # Build the point clouds once per K (seed-independent) by truncating the shared extraction.
    clouds_by_K = {k: build_clouds(state, device=str(device), K=k) for k in K_grid}

    indices_a, values_a, _, _ = get_cache_with_toks_dataset_on_target_components(
        sae_a, model, data_toks, [sae_a.cfg.hook_name], args.batch_size, device)
    indices_b, values_b, _, _ = get_cache_with_toks_dataset_on_target_components(
        sae_b, model, data_toks, [sae_b.cfg.hook_name], args.batch_size, device)
    pad_value = sae_a.cfg.d_sae
    indices_a[values_a < 1e-3] = pad_value
    feature_counts = t.bincount(indices_a.flatten(), minlength=sae_a.cfg.d_sae + 1)
    feature_counts[pad_value] = 0
    caches = (indices_a, values_a, indices_b, values_b)

    eligible = eligible_queries(state, feature_counts, args.min_activation_examples)
    print(f"[ot_topk_ablation] {len(eligible)} eligible query features")

    fixed_key = None
    if args.query_prev_file is not None:
        name = args.query_prev_file.format(a=sae_a_id, b=sae_b_id)
        candidates = [os.path.join(d, name) for d in (cfg.results_dir, output_dir)]
        prev_path = next((c for c in candidates if os.path.exists(c)), None)
        if prev_path is None:
            raise FileNotFoundError(f"--query_prev_file not found in any of {candidates}")
        details = t.load(prev_path, weights_only=False)["llm_eval"]["llm_eval"]["matching_details"]
        fixed_key = [int(d["feature_pair"][0]) for d in details]
        not_eligible = sorted(set(fixed_key) - set(int(x) for x in eligible))
        print(f"[ot_topk_ablation] query set from {prev_path}: {len(fixed_key)} queries "
              f"({len(set(fixed_key))} distinct); {len(not_eligible)} not in the eligible set: {not_eligible}")

    per_seed = {k: {"llm_eval_score": [], "llm_eval_accuracy": []} for k in K_grid}
    csv_rows = []
    saved_matches = {}

    for seed in args.seeds:
        print(f"[ot_topk_ablation] pair ({sae_a_id},{sae_b_id}) seed {seed}")
        exp_config.set_seed(seed)
        if fixed_key is not None:
            chosen_key = list(fixed_key)  # paper protocol: same queries for every seed and K
        else:
            chosen_key = random.choices(eligible, k=args.number_of_pairs)  # same sample across all K

        seed_matches = {}
        for k in K_grid:
            rer = rerank_queries(
                state, clouds_by_K[k], chosen_key, M=args.M, device=str(device),
                chunk_size=args.chunk_size, stage1_metric=args.metric, ot_method=args.ot_method,
                sinkhorn_impl=args.sinkhorn_impl,
            )
            assignments = assignments_at_M(rer["per_query"], args.M)
            out, feat_pairs = evaluate_matches(
                model, data_toks, assignments, caches, judge,
                query_order=[q for q in chosen_key if q in assignments] if fixed_key is not None else None)
            if fixed_key is not None and len(feat_pairs) != len(chosen_key):
                print(f"    WARNING K={k}: only {len(feat_pairs)}/{len(chosen_key)} queries have a match")
            seed_matches[k] = {int(a): int(b) for a, b in feat_pairs}

            per_seed[k]["llm_eval_score"].append(out["llm_eval_score"])
            per_seed[k]["llm_eval_accuracy"].append(out["llm_eval_accuracy"])
            print(f"    K={k}: score={out['llm_eval_score']:.3f} acc={out['llm_eval_accuracy']:.3f}")
            csv_rows.append({"pair": f"{sae_a_id}_{sae_b_id}", "seed": seed, "K": k,
                             "llm_eval_score": out["llm_eval_score"],
                             "llm_eval_accuracy": out["llm_eval_accuracy"]})

        if args.save_matches:
            saved_matches[seed] = {"chosen_key": [int(x) for x in chosen_key], "matches": seed_matches}

    aggregated = {k: {key: exp_config.summarize_stats(vals) for key, vals in per_seed[k].items() if vals}
                  for k in K_grid}
    results = {
        "config": {
            "sae_a_id": sae_a_id, "sae_b_id": sae_b_id, "K_grid": K_grid, "M": args.M,
            "number_of_pairs": args.number_of_pairs, "min_activation_examples": args.min_activation_examples,
            "seeds": list(args.seeds), "metric": args.metric, "ot_method": args.ot_method,
            "judge": judge["name"], "n_eligible_queries": len(eligible),
            "sinkhorn_impl": args.sinkhorn_impl, "caching_hook": args.caching_hook, "query_prev_file": args.query_prev_file,
        },
        "aggregated": aggregated,
        "per_seed": per_seed,
        "csv_rows": csv_rows,
    }
    if args.save_matches:
        results["matches"] = saved_matches
    return results, csv_rows


def main(args: argparse.Namespace) -> None:
    cfg = exp_config.ModelConfig(args)
    device = cfg.device
    judge = exp_config.resolve_judge(args.judge, args.judge_effort)

    model, dict_saes, list_saes = exp_config.load_model_and_saes(cfg)

    from sae.Load_Data import load_data
    from sae.SAE_Trainer import DataConfig
    data_config = DataConfig(data_path=args.data_path, batch_size=1, max_length=args.max_length)
    _, data_loader = load_data(cfg.tok_name, data_config)
    data_toks = extract_data(data_loader, num_batches=args.num_batches)

    output_dir = (args.results_dir if args.results_dir is not None
                  else cfg.results_dir_for_judge(judge["name"])).rstrip("/") + "/"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[ot_topk_ablation] writing results to {output_dir}")

    all_csv_rows = []
    for pair in args.pairs:
        sae_a_id, sae_b_id = (int(x) for x in pair.split(","))
        assert sae_a_id > sae_b_id, f"pair {pair}: A must be the later layer"
        results, csv_rows = run_pair(args, cfg, model, list_saes, data_toks,
                                     sae_a_id, sae_b_id, judge, device, output_dir)
        all_csv_rows.extend(csv_rows)

        fname = (f"{args.out_prefix}_{sae_a_id}_{sae_b_id}_{args.number_of_pairs}_"
                 f"{args.min_activation_examples}_{len(args.seeds)}.pt")
        t.save(results, output_dir + fname)
        print(f"[ot_topk_ablation] saved {output_dir + fname}")

    if all_csv_rows:
        csv_path = f"{output_dir}{args.out_prefix}_{args.number_of_pairs}_{args.min_activation_examples}_{len(args.seeds)}_runs.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(all_csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(all_csv_rows)
        print(f"[ot_topk_ablation] saved {csv_path}")


if __name__ == "__main__":
    main(parse_args())
