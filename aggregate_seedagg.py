"""Merge `*_seedagg.json` files from truncated data ranges and recompute the statistics.

`circuit_compress_eval.py --aggregate_seeds true` writes one JSON per cluster type per data range
(`{cluster_type}_{node_threshold}_{n_clusters}_{start}_{end}_seedagg.json`). Run the compression
eval over disjoint slices (`--data_range 0 25`, `25 50`, ...) to parallelize the circuit builds,
then this script stitches the per-range JSONs back together: it concatenates the per-prompt records
and recomputes every statistic with `exp_config.aggregate_seed_prompts` -- the *same* function the
eval uses -- so the merged numbers are identical to what a single combined run would have produced.

No model or GPU is loaded (it only reads JSON), so it is safe to run on a login node.

Usage:
    python aggregate_seedagg.py --inputs A_50_5_0_25_seedagg.json A_50_5_25_50_seedagg.json
    python aggregate_seedagg.py --glob "results/google_gemma-2-2b_mlp/naive_50_5_*_seedagg.json"
    python aggregate_seedagg.py --glob "results/.../*_seedagg.json" --output_dir results/merged
"""
import argparse
import glob as globlib
import json
import os
from collections import defaultdict
from typing import Any, Dict, List

import exp_config

# Config fields that must be identical for a merge to be meaningful (a mismatch is an error).
CRITICAL_KEYS = ("model", "sae_type", "cluster_type", "cluster_seed", "n_clusters",
                 "node_threshold", "seeds")
# Config fields that should match but only warn (they change the numbers subtly, not fatally).
WARN_KEYS = ("judge", "judge_model", "llm")


def parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Merge truncated-range *_seedagg.json files and recompute statistics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--inputs", type=str, nargs="+", help="Explicit list of *_seedagg.json files.")
    src.add_argument("--glob", type=str, help="Glob pattern matching the *_seedagg.json files.")
    p.add_argument("--output_dir", type=str, default=None,
                   help="Where to write the merged JSON(s); defaults to the first input's directory.")
    p.add_argument("--suffix", type=str, default="merged",
                   help="Filename tag: {cluster_type}_{nt}_{nc}_{start}_{end}_seedagg_{suffix}.json.")
    p.add_argument("--strict", action="store_true",
                   help="Treat WARN-level config mismatches (judge/llm params) as errors too.")
    return p.parse_args(argv)


def load_files(paths: List[str]) -> List[Dict[str, Any]]:
    reports = []
    for path in paths:
        with open(path) as fh:
            rep = json.load(fh)
        rep["_path"] = path
        reports.append(rep)
    return reports


def check_compatible(group: List[Dict[str, Any]], strict: bool) -> None:
    """Raise on CRITICAL config mismatches; print a warning on WARN-level ones (or raise if strict)."""
    ref = group[0]["config"]
    for rep in group[1:]:
        cfg = rep["config"]
        for k in CRITICAL_KEYS:
            if cfg.get(k) != ref.get(k):
                raise SystemExit(
                    f"[merge] incompatible '{k}': {rep['_path']} has {cfg.get(k)!r} but "
                    f"{group[0]['_path']} has {ref.get(k)!r}. Refusing to merge."
                )
        for k in WARN_KEYS:
            if cfg.get(k) != ref.get(k):
                msg = (f"[merge] WARNING: '{k}' differs ({cfg.get(k)!r} vs {ref.get(k)!r}) between "
                       f"{rep['_path']} and {group[0]['_path']}")
                if strict:
                    raise SystemExit(msg.replace("WARNING", "ERROR") + " (--strict)")
                print(msg)


def merge_prompts(group: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Concatenate prompt records across ranges, deduping by global_prompt_idx.

    Ranges are meant to be disjoint; if two files cover the same global prompt we keep the first
    and warn, so an accidental overlap can't silently double-count.
    """
    # order by data_range start for stable, human-readable output
    group = sorted(group, key=lambda r: r["config"]["data_range"][0])
    ranges = [tuple(r["config"]["data_range"]) for r in group]

    # overlap check on the declared ranges
    for (s1, e1), (s2, e2) in zip(ranges, ranges[1:]):
        if s2 < e1:
            print(f"[merge] WARNING: data ranges {(s1, e1)} and {(s2, e2)} overlap; "
                  f"deduping by global_prompt_idx.")

    merged: List[Dict[str, Any]] = []
    seen = set()
    for rep in group:
        base = rep["config"]["data_range"][0]
        for i, prec in enumerate(rep.get("prompts", [])):
            gidx = prec.get("global_prompt_idx", base + prec.get("prompt_idx", i))
            if gidx in seen:
                print(f"[merge] WARNING: duplicate global_prompt_idx {gidx} "
                      f"(from {rep['_path']}); skipping the duplicate.")
                continue
            seen.add(gidx)
            merged.append(prec)
    merged.sort(key=lambda p: p.get("global_prompt_idx", 0))
    return merged


def build_merged_report(group: List[Dict[str, Any]]) -> Dict[str, Any]:
    ref_cfg = dict(group[0]["config"])
    seeds = ref_cfg["seeds"]
    prompts = merge_prompts(group)

    starts = [r["config"]["data_range"][0] for r in group]
    ends = [r["config"]["data_range"][1] for r in group]
    merged_cfg = dict(ref_cfg)
    merged_cfg["data_range"] = [min(starts), max(ends)]
    merged_cfg["merged_from"] = [
        {"file": os.path.basename(r["_path"]), "data_range": r["config"]["data_range"]}
        for r in sorted(group, key=lambda r: r["config"]["data_range"][0])
    ]
    merged_cfg["source_ranges"] = sorted([list(r["config"]["data_range"]) for r in group])

    report = {
        "config": merged_cfg,
        "n_prompts": len(prompts),
        **exp_config.aggregate_seed_prompts(prompts, seeds),
        "prompts": prompts,
    }
    return report


def _fmt(s: Dict[str, Any]) -> str:
    if s.get("mean") is None:
        return "n=0"
    return (f"mean={s['mean']:.4f} std={s['std']:.4f} "
            f"90%CI=[{s['ci90_low']:.4f}, {s['ci90_high']:.4f}] "
            f"min={s['min']:.4f} max={s['max']:.4f} median={s['median']:.4f} n={s['n']}")


def main(argv=None) -> None:
    args = parse_args(argv)
    paths = args.inputs if args.inputs else sorted(globlib.glob(args.glob))
    if not paths:
        raise SystemExit("[merge] no input files matched.")

    reports = load_files(paths)

    by_ctype: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for rep in reports:
        by_ctype[rep["config"]["cluster_type"]].append(rep)

    out_dir = args.output_dir or os.path.dirname(os.path.abspath(paths[0]))
    os.makedirs(out_dir, exist_ok=True)

    for ctype, group in by_ctype.items():
        check_compatible(group, args.strict)
        report = build_merged_report(group)
        cfg = report["config"]
        name = (f"{ctype}_{cfg['node_threshold']}_{cfg['n_clusters']}_"
                f"{cfg['data_range'][0]}_{cfg['data_range'][1]}_seedagg_{args.suffix}.json")
        out_path = os.path.join(out_dir, name)
        with open(out_path, "w") as fh:
            json.dump(report, fh, ensure_ascii=False, indent=2)

        print(f"\nCluster type: {ctype}  ({report['n_prompts']} prompts from {len(group)} files, "
              f"range {cfg['data_range']}, seeds {cfg['seeds']})")
        print(f"  llm_acc[micro]: {_fmt(report['llm_accuracy']['micro_stats'])}")
        print(f"  llm_acc[macro]: {_fmt(report['llm_accuracy']['macro_stats'])}")
        print(f"  inter: {_fmt(report['inter_stats'])}")
        print(f"  intra: {_fmt(report['intra_stats'])}")
        print(f"[merge] saved {out_path}")


if __name__ == "__main__":
    main()
