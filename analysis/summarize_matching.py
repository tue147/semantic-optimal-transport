"""Table 1 / Table 6: summarize feature-matching runs written by feature_matching_test.py.

For one layer pair, reads every judge run of each method (`{method}_{a}_{b}_1_100_30_run{N}.pt`, plus
an untagged `{method}_{a}_{b}_1_100_30.pt` if present) and prints LLM Eval and Acc as
mean ± 1.645 * std / sqrt(n) over the runs (the 90% interval reported in the paper), and CE / VE when
the run files carry them (`--save_skipping true`).

No model, no GPU, no API calls:

    python analysis/summarize_matching.py --results_dir results/google_gemma-2-2b_mlp_gpt-oss-120b --pair 13 12
"""
import argparse
import glob
import sys
import os

import numpy as np
import torch

# Result files pickle repo classes (graph.*), so the repo root must be importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NAMES = {"ot": "Our", "feat_flow": "FeatFlow", "mech_permute": "Match", "feat_flow_ml": "FeatFlow ML",
         "mech_permute_ml": "Match ML", "causal": "Attrib", "naive": "Our Naive"}


def load_run(path):
    r = torch.load(path, weights_only=False)
    llm = r["llm_eval"]["llm_eval"]
    scores = [int(d["score"]) for d in llm["matching_details"]]
    skip = r.get("skipping") or {}
    return {"llm": float(llm["overall_score"]), "acc": 100.0 * float(np.mean([s == 3 for s in scores])),
            "ce": skip.get("skipping_effect"), "ve": skip.get("var_ex_effect")}


def interval(xs):
    x = np.asarray(xs, float)
    return x.mean(), 1.645 * x.std() / np.sqrt(len(x))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--pair", type=int, nargs=2, required=True, metavar=("A", "B"), help="target and source layer")
    p.add_argument("--methods", nargs="+", default=list(NAMES))
    p.add_argument("--stem", default="1_100_30", help="top_n-pairs_min-activation part of the file name")
    args = p.parse_args()

    a, b = args.pair
    print(f"layer {b} -> layer {a}, {args.results_dir}")
    print(f"{'method':<12} {'n':>2} {'LLM Eval':>14} {'Acc':>12} {'CE':>9} {'VE':>8}")
    for m in args.methods:
        stem = os.path.join(args.results_dir, f"{m}_{a}_{b}_{args.stem}")
        files = sorted(set(glob.glob(stem + ".pt") + glob.glob(stem + "_run[0-9]*.pt")))
        if not files:
            continue
        runs = [load_run(f) for f in files]
        llm, llm_e = interval([r["llm"] for r in runs])
        acc, acc_e = interval([r["acc"] for r in runs])
        ce = [r["ce"] for r in runs if r["ce"] is not None]
        ve = [r["ve"] for r in runs if r["ve"] is not None]
        ce_s = f"{np.mean(ce):.5f}" if ce else "-"
        ve_s = f"{np.mean(ve):.4f}" if ve else "-"
        print(f"{NAMES.get(m, m):<12} {len(runs):>2} {f'{llm:.2f} ± {llm_e:.2f}':>14} "
              f"{f'{acc:.1f} ± {acc_e:.1f}':>12} {ce_s:>9} {ve_s:>8}")


if __name__ == "__main__":
    main()
