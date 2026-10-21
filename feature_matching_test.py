"""Evaluate cross-layer feature matchings: LLM judge score + layer-skipping CE loss / VE.

Loads the saved alignments for a layer pair (see `exp_config.LAYER_PAIR_PRESETS`), optionally
computes the `causal` (attribution) baseline on the fly, then scores every method.

Repeat runs: `--seeds 1 2 3 4` runs the whole evaluation once per seed and writes one file per
seed. The output name is unchanged: `{method}_{a}_{b}_{top}_{n_pairs}_{min_act}_run{seed}.pt`.

Alongside those, one **details log** per layer pair collects the seeds so no separate aggregation
step is needed: `matching_{a}_{b}_{top}_{n_pairs}_{min_act}_seedagg.json` holds, per method, every
per-seed record (LLM score, accuracy = fraction scored 3, the 2/1 score fractions, layer-skipping
CE and VE, and every judged feature pair) plus mean / std / sem / 90% CI across seeds from
`exp_config.summarize_stats`. It is rewritten after every method, so a killed job keeps what it
already paid for, and re-running with more seeds merges into the same file.

`--backfill true` rebuilds that JSON from result `.pt` files already on disk instead of running
anything -- no model, no GPU, no API calls, so it is safe on a login node. Runs written before
`--save_skipping` was on carry no CE/VE in the `.pt`; `--backfill_logs "logs/*.out"` recovers those
from the job logs (the values are printed there next to the file they belong to).

WARNING: the LLM eval fans out paid OpenAI calls. Launcher: scripts/feature_matching_test.sh
"""

import argparse
import asyncio
import glob as globlib
import json
import os
import random
import re
from typing import Any, Dict, Optional, Tuple

from Evaluation_Utils import *
from math import log2
import einops
from functools import partial
from test_matching.util import *
from test_matching.llm_eval import *
from test_matching.patching import *
from test_matching.layer_skipping import *
from feature_matching.causal_matching import *

import exp_config


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate cross-layer SAE feature matchings (LLM judge + layer skipping).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    exp_config.add_model_args(parser, default_preset="gemma_mlp", default_device_id=0)
    exp_config.add_data_args(parser, default_batchsize=96, default_max_length=128, default_num_batches=10000)
    exp_config.add_judge_args(parser)

    group = parser.add_argument_group("evaluation")
    group.add_argument("--sae_a_id", type=int, default=25, help="Target (later) layer.")
    group.add_argument("--sae_b_id", type=int, default=12, help="Source (earlier) layer.")
    group.add_argument("--methods", type=str, nargs="+", default=["causal"],
                       help="Which methods of the layer-pair preset to evaluate.")
    group.add_argument("--batch_size", type=int, default=96, help="Batch size for caching and patching.")
    group.add_argument("--min_activation_examples", type=int, default=30,
                       help="Minimum firings for a feature to be eligible for sampling.")
    group.add_argument("--number_of_pairs", type=int, default=100, help="Feature pairs sampled per method.")
    group.add_argument("--top", type=int, default=1, help="Top-k matches taken per source feature.")

    seed_group = parser.add_argument_group("repeat runs")
    seed_group.add_argument("--seeds", type=int, nargs="+", default=[4],
                            help="One full evaluation per seed; the seed is the runN filename suffix.")
    seed_group.add_argument("--run_prefix", type=str, default="run", help="Filename suffix prefix, e.g. run4.")

    key_group = parser.add_argument_group("feature selection")
    key_group.add_argument("--extract_from_prev", type=exp_config.str_to_bool, default=True,
                           help="True reuses the feature set from --prev_file; False samples fresh keys.")
    key_group.add_argument("--prev_file", type=str, default="feat_flow_13_12_1_100_30.pt",
                           help="Result file whose matching_details supply the evaluated features.")

    causal_group = parser.add_argument_group("causal (attribution) baseline")
    causal_group.add_argument("--causal_grad_only", type=exp_config.str_to_bool, default=True)
    causal_group.add_argument("--causal_use_error_term", type=exp_config.str_to_bool, default=True)
    causal_group.add_argument("--causal_direct_effect", type=exp_config.str_to_bool, default=True)

    parser.add_argument("--results_dir", type=str, default=None,
                        help="Override both input and output dir (verbatim). Default: read inputs "
                             "from results/<model>_<sae_type>, write results to "
                             "results/<model>_<sae_type>_<judge>.")
    parser.add_argument("--save_skipping", type=exp_config.str_to_bool, default=False,
                        help="Also store the layer-skipping metrics in the saved .pt (they are printed "
                             "and written to the seedagg JSON either way).")

    back = parser.add_argument_group("backfill (no model / no API calls)")
    back.add_argument("--backfill", type=exp_config.str_to_bool, default=False,
                      help="Rebuild the seedagg JSON from the .pt files already in the results dir "
                           "and exit. Aggregates every method found, ignoring --methods.")
    back.add_argument("--backfill_logs", type=str, nargs="*", default=None,
                      help="Glob(s) of feature_matching_test job logs used to recover CE/VE for runs "
                           "whose .pt predates --save_skipping, e.g. 'logs/feature_matching_test_*.out'.")
    back.add_argument("--backfill_unsuffixed", type=exp_config.str_to_bool, default=False,
                      help="Also count a file with no _run tag (the old run-1 naming) as a seed.")

    return parser.parse_args(argv)


def _seed_order(seed) -> Tuple[int, Any]:
    """Numeric seeds first, in order; anything else (e.g. the untagged 'base' run) after them."""
    s = str(seed)
    return (0, int(s)) if s.isdigit() else (1, s)


def load_report(path: str, config: Dict[str, Any]) -> Dict[str, Any]:
    """Read an existing details log so more seeds merge into it, or start a fresh one."""
    if os.path.exists(path):
        with open(path) as fh:
            report = json.load(fh)
        report.setdefault("methods", {})
        report["config"] = {**report.get("config", {}), **config}
        return report
    return {"config": dict(config), "methods": {}}


def record_run(report: Dict[str, Any], method_name: str, seed, record: Dict[str, Any]) -> None:
    """Add/replace one (method, seed) record and recompute that method's across-seed stats."""
    entry = report["methods"].setdefault(method_name, {"per_seed": {}})
    entry["per_seed"][str(seed)] = record
    entry["per_seed"] = {k: entry["per_seed"][k] for k in sorted(entry["per_seed"], key=_seed_order)}
    entry["n_runs"] = len(entry["per_seed"])
    entry["stats"] = exp_config.summarize_matching_seeds(entry["per_seed"])


def save_report(report: Dict[str, Any], path: str) -> None:
    report["config"]["seeds"] = sorted(
        {s for m in report["methods"].values() for s in m["per_seed"]}, key=_seed_order)
    with open(path, "w") as fh:
        json.dump(report, fh, indent=2)


def report_config(args, cfg, judge_name: Optional[str], results_dir: str,
                  backfilled: bool = False) -> Dict[str, Any]:
    """The config block of the details log: everything that changes the numbers.

    A backfill only sees the `.pt` files, so it records nothing it cannot read off them --
    `--batch_size` / `--extract_from_prev` / `--prev_file` describe *this* invocation, not the job
    that produced the runs, and claiming them would be a guess. Merging into a log an actual run
    wrote keeps the values that run recorded.
    """
    config = {
        "results_dir": results_dir,
        "model": cfg.model_name,
        "sae_type": cfg.sae_type,
        "judge": judge_name,
        "sae_a_id": args.sae_a_id,
        "sae_b_id": args.sae_b_id,
        "top": args.top,
        "number_of_pairs": args.number_of_pairs,
        "min_activation_examples": args.min_activation_examples,
        "run_prefix": args.run_prefix,
        "source": "backfill" if backfilled else "run",
    }
    if not backfilled:
        config |= {
            "batch_size": args.batch_size,
            "extract_from_prev": args.extract_from_prev,
            "prev_file": args.prev_file if args.extract_from_prev else None,
        }
    return config


def parse_skipping_from_logs(patterns, file_pattern: re.Pattern) -> Dict[str, Tuple[float, float]]:
    """{result_file_basename: (CE, VE)} recovered from job logs.

    A log prints, per method and in this order, `<method> Skipping Effect: x`,
    `<method> Var Explained Effect: y`, then `[feature_matching_test] saved <path>`. The pending
    values are keyed by method name and flushed on the `saved` line -- with the method read back
    out of the saved file name, so `feat_flow` can't claim `feat_flow_ml`'s numbers. Files from
    other layer pairs don't match `file_pattern` and are skipped; a later log wins.
    """
    skip_re = re.compile(r"^(\S+) Skipping Effect: ([-\d.eE+]+)\s*$")
    var_re = re.compile(r"^(\S+) Var Explained Effect: ([-\d.eE+]+)\s*$")
    saved_re = re.compile(r"^\[feature_matching_test\] saved (\S+)\s*$")

    out: Dict[str, Tuple[float, float]] = {}
    paths = [p for pat in (patterns or []) for p in sorted(globlib.glob(pat))]
    for log_path in paths:
        pending: Dict[str, Dict[str, float]] = {}
        with open(log_path, errors="replace") as fh:
            for line in fh:
                m = skip_re.match(line)
                if m:
                    pending.setdefault(m.group(1), {})["ce"] = float(m.group(2))
                    continue
                m = var_re.match(line)
                if m:
                    pending.setdefault(m.group(1), {})["ve"] = float(m.group(2))
                    continue
                m = saved_re.match(line)
                if m is None:
                    continue
                base = os.path.basename(m.group(1))
                hit = file_pattern.match(base)
                if hit is None:
                    continue
                vals = pending.pop(hit.group("method"), {})
                if "ce" in vals and "ve" in vals:
                    out[base] = (vals["ce"], vals["ve"])
    if paths:
        print(f"[feature_matching_test] recovered CE/VE for {len(out)} run files from {len(paths)} log(s)")
    return out


def backfill(args) -> None:
    """Rebuild the seedagg JSON from the .pt files on disk. No model, no GPU, no API calls."""
    cfg = exp_config.ModelConfig(args)
    judge_name = None if args.results_dir is not None else args.judge
    output_dir = (args.results_dir if args.results_dir is not None
                  else cfg.results_dir_for_judge(args.judge)).rstrip("/") + "/"
    print(f"[feature_matching_test] backfilling from {output_dir}")

    pattern = exp_config.matching_run_file_regex(args)
    log_skipping = parse_skipping_from_logs(args.backfill_logs, pattern)

    found: Dict[str, Dict[Any, str]] = {}
    for path in sorted(globlib.glob(os.path.join(output_dir, "*.pt"))):
        m = pattern.match(os.path.basename(path))
        if m is None:
            continue
        seed = m.group("seed")
        if seed is None:
            if not args.backfill_unsuffixed:
                continue
            seed = "base"
        found.setdefault(m.group("method"), {})[seed] = path

    if not found:
        raise SystemExit(
            f"[feature_matching_test] no files matching *_{args.sae_a_id}_{args.sae_b_id}_"
            f"{args.top}_{args.number_of_pairs}_{args.min_activation_examples}_"
            f"{args.run_prefix}<seed>.pt in {output_dir}"
        )

    path = os.path.join(output_dir, exp_config.matching_seedagg_file_name(args))
    report = load_report(path, report_config(args, cfg, judge_name, output_dir, backfilled=True))
    for method_name in sorted(found):
        for seed in sorted(found[method_name], key=_seed_order):
            file_path = found[method_name][seed]
            results = t.load(file_path, map_location="cpu", weights_only=False)
            base = os.path.basename(file_path)
            from_log = "skipping" not in results and base in log_skipping
            if from_log:
                ce, ve = log_skipping[base]
                results = dict(results) | {"skipping": {"skipping_effect": ce, "var_ex_effect": ve}}
            record = exp_config.matching_run_metrics(results, with_pairs=True)
            record["file"] = base
            if from_log:
                record["skipping_source"] = "log"
            record_run(report, method_name, seed, record)

    save_report(report, path)
    print("\n" + exp_config.format_matching_table(report["methods"]))
    print(f"\n[feature_matching_test] saved {path}")


def select_chosen_key(args, base_assignments, feature_counts, file_dirs):
    """The evaluated feature set: either sampled fresh or lifted from a previous result file.

    `file_dirs` are searched in order for --prev_file (the base results dir, then the judge-scoped
    output dir, where a first run with --extract_from_prev false writes its files).
    """
    if not args.extract_from_prev:
        valid_key = None
        for k, assignments in base_assignments.items():
            if valid_key is None:
                valid_key = set([k for k in assignments.keys() if feature_counts[k] > args.min_activation_examples])
            else:
                temp = set([k for k in assignments.keys() if feature_counts[k] > args.min_activation_examples])
                valid_key = valid_key.intersection(temp)

        valid_key = list(valid_key)  # type: ignore
        assert len(valid_key) > 0
        print("Number of valid_key: ", len(valid_key))

        return random.choices(valid_key, k=args.number_of_pairs)

    candidates = [d + args.prev_file for d in file_dirs]
    prev_path = next((c for c in candidates if os.path.exists(c)), None)
    if prev_path is None:
        raise FileNotFoundError(f"--prev_file not found in any of {candidates}")
    temp = t.load(prev_path, weights_only=False)["llm_eval"]["llm_eval"]["matching_details"]
    chosen_key = []
    for tmp in temp:
        chosen_key.append(tmp["feature_pair"][0])
    return chosen_key


def evaluate_seed(args, seed, cfg, model, sae_a, sae_b, data_toks, base_assignments, file_names,
                  feature_counts, caches, input_dir, output_dir, device, judge, report, report_path):
    """One full evaluation pass. Identical to the original body, wrapped for repeat runs.

    input_dir holds the (judge-independent) alignment / prev_file inputs; output_dir is where the
    judge-scoped eval results are written. `report` is the shared seedagg details log, rewritten
    after every method so a killed job keeps the seeds it already paid for.
    """
    indices_a, values_a, indices_b, values_b = caches

    exp_config.set_seed(seed)
    chosen_key = select_chosen_key(args, base_assignments, feature_counts, [input_dir, output_dir])

    # Copy so the on-the-fly causal baseline of one seed never leaks into the next.
    all_assignments = dict(base_assignments)

    if "causal" in file_names:
        all_assignments |= {
            "causal": attribution(
                model=model,
                dict_saes={sae_a.cfg.hook_name: sae_a, sae_b.cfg.hook_name: sae_b},
                dataloader=get_loader(data_toks, args.batch_size),
                feature_set_a=chosen_key,
                hook_position_a=sae_a.cfg.hook_name,
                hook_position_b=sae_b.cfg.hook_name,
                k=args.top,
                grad_only=args.causal_grad_only,
                use_error_term=args.causal_use_error_term,
                direct_effect=args.causal_direct_effect,
            )
        }

    for method_name, assignments in all_assignments.items():
        all_feat_pairs = []
        for feat_a in chosen_key:
            all_feat_pairs.extend(extract_pair(feat_a, assignments, top=args.top))

        assert len(all_feat_pairs) > 0

        llm_eval_results = {}
        res = asyncio.run(
            llm_matching_eval(
                all_feat_pairs,
                model,
                data_toks,
                indices_a,
                values_a,
                indices_b,
                values_b,
                judge,
            )
        )
        llm_eval_results["llm_eval"] = res
        llm_eval_results["llm_eval_effect"] = res["overall_score"]
        print(f"{method_name} LLM Eval Effect:", llm_eval_results["llm_eval_effect"])

        layer_skip_results = {"total_skipping_effect": 0, "total_var_ex_effect": 0, "count": 0, "details": {}}

        end_nodes = []
        end_indices = []
        start_nodes = []
        start_indices = []
        for feat_a, feat_b in all_feat_pairs:
            end_node = ConnectionNode(sae_a.cfg.hook_name)
            end_index = FeatureIndex([int(feat_a)], 3)

            start_node = ConnectionNode(sae_b.cfg.hook_name)
            start_index = FeatureIndex([int(feat_b)], 3)

            end_nodes.append(end_node)
            end_indices.append(end_index)
            start_nodes.append(start_node)
            start_indices.append(start_index)

        loss_score, var_ex = get_activation_layer_skipping(
            model,
            sae_a,
            sae_b,
            end_nodes,
            end_indices,
            start_nodes,
            start_indices,
            data_toks,
            use_error_term=True,
            batch_size=args.batch_size,
            device=device,
            verbose=False,
        )
        layer_skip_results["details"][(end_node, end_index, start_node, start_index)] = (loss_score, var_ex)
        layer_skip_results["total_skipping_effect"] = loss_score
        layer_skip_results["total_var_ex_effect"] = var_ex
        layer_skip_results["count"] = len(all_feat_pairs)

        layer_skip_results["skipping_effect"] = layer_skip_results["total_skipping_effect"]  # / count
        layer_skip_results["var_ex_effect"] = layer_skip_results["total_var_ex_effect"]  # / count
        print(f"{method_name} Skipping Effect:", layer_skip_results["skipping_effect"])
        print(f"{method_name} Var Explained Effect:", layer_skip_results["var_ex_effect"])

        file_name = exp_config.matching_run_file_name(args, method_name, seed)

        results = {"llm_eval": llm_eval_results, "skipping": layer_skip_results}
        t.save(results if args.save_skipping else {"llm_eval": llm_eval_results},
               output_dir + file_name)
        print(f"[feature_matching_test] saved {output_dir + file_name}")

        # The details log always carries CE/VE; --save_skipping only controls the .pt. It is
        # updated *after* the .pt so nothing paid for is lost if a record turns out unserializable
        # -- and `--backfill true` can rebuild the log from the .pt files anyway.
        try:
            record = exp_config.matching_run_metrics(results, with_pairs=True)
            record["file"] = file_name
            record_run(report, method_name, seed, record)
            save_report(report, report_path)
        except Exception as exc:  # noqa: BLE001 - never let the details log kill a paid eval
            print(f"[feature_matching_test] WARNING: could not update {report_path}: {exc!r}")


def main(args: argparse.Namespace) -> None:
    if args.backfill:
        backfill(args)
        return

    cfg = exp_config.ModelConfig(args)
    device = cfg.device

    judge = exp_config.resolve_judge(args.judge, args.judge_effort)

    model, dict_saes, list_saes = exp_config.load_model_and_saes(cfg)

    from sae.Load_Data import load_data
    from sae.SAE_Trainer import DataConfig

    # This script re-extracts tokens rather than using the full_data cache, matching the original.
    data_config = DataConfig(
        data_path=args.data_path,
        batch_size=1,
        max_length=args.max_length,
    )
    _, data_loader = load_data(cfg.tok_name, data_config)
    data_toks = extract_data(data_loader, num_batches=args.num_batches)

    # Inputs (alignments, prev_file) are judge-independent and live in the base results dir; eval
    # outputs go to a judge-scoped dir so judges don't overwrite each other. An explicit
    # --results_dir overrides both (verbatim, no judge suffix), preserving the old behaviour.
    input_dir = (args.results_dir if args.results_dir is not None else cfg.results_dir).rstrip("/") + "/"
    output_dir = (args.results_dir if args.results_dir is not None
                  else cfg.results_dir_for_judge(judge["name"])).rstrip("/") + "/"
    os.makedirs(output_dir, exist_ok=True)
    print(f"[feature_matching_test] reading inputs from {input_dir}, writing results to {output_dir}")

    file_names = exp_config.resolve_file_names(args.sae_a_id, args.sae_b_id, args.methods)
    print(f"[feature_matching_test] evaluating {list(file_names)} for pair ({args.sae_a_id}, {args.sae_b_id})")

    sae_a = list_saes[args.sae_a_id]
    sae_b = list_saes[args.sae_b_id]
    base_assignments = {
        k: t.load(input_dir + v, weights_only=False) for k, v in file_names.items() if v is not None
    }

    indices_a, values_a, _, _ = get_cache_with_toks_dataset_on_target_components(
        sae_a,
        model,
        data_toks,
        [sae_a.cfg.hook_name],
        args.batch_size,
        device
    )

    indices_b, values_b, _, _ = get_cache_with_toks_dataset_on_target_components(
        sae_b,
        model,
        data_toks,
        [sae_b.cfg.hook_name],
        args.batch_size,
        device
    )

    pad_value = sae_a.cfg.d_sae
    indices_a[values_a < 1e-3] = pad_value
    feature_counts = t.bincount(indices_a.flatten(), minlength=sae_a.cfg.d_sae + 1)
    feature_counts[pad_value] = 0  # remove frequency of pad_indice

    caches = (indices_a, values_a, indices_b, values_b)

    # One details log per layer pair, merged into across runs of this script.
    report_path = os.path.join(output_dir, exp_config.matching_seedagg_file_name(args))
    report = load_report(report_path, report_config(args, cfg, judge["name"], output_dir))

    for seed in args.seeds:
        print(f"\n[feature_matching_test] ===== seed {seed} =====")
        evaluate_seed(
            args, seed, cfg, model, sae_a, sae_b, data_toks, base_assignments, file_names,
            feature_counts, caches, input_dir, output_dir, device, judge, report, report_path,
        )

    print("\n" + exp_config.format_matching_table(report["methods"]))
    print(f"\n[feature_matching_test] saved {report_path}")


if __name__ == "__main__":
    main(parse_args())
