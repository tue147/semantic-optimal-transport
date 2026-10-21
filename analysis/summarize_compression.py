"""Table 2 / Table 7: summarize circuit-compression results written by circuit_compress_eval.py.

Merges every prompt slice of each cluster type, then reports the LLM-eval accuracy two ways:
macro = mean of the per-circuit accuracies over gradable circuits, with a 90% confidence interval
(1.645 x standard error) -- the number in the tables -- and micro = pooled sum(correct) /
sum(judged) over every circuit. Also a paired t-test of the macro scores against our method on
the circuits both could grade. A circuit is gradable when the judge graded at
least one feature, i.e. at least one supernode has >= 2 features (Appendix C.4).

No model, no GPU, no API calls:

    python analysis/summarize_compression.py --results_dir results/google_gemma-2-2b_mlp_llama70b-vllm
    python analysis/summarize_compression.py --results_dir results/gpt2-small_ds_llama70b-vllm \
        --node_threshold 200 --n_clusters 20
"""
import argparse
import glob
import os
import sys
import math
import re

import numpy as np
import torch
from scipy import stats

# Result files pickle repo classes (graph.*), so the repo root must be importable.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

NAMES = {"ot_global_10": "Our", "feat_flow_llama": "FeatFlow", "mech_permute_llama": "Match",
         "llm": "LLM Sel", "causal": "Attrib", "naive": "Our Naive"}


def load_circuits(files):
    """{global circuit index: (llm_eval, gradable, n_correct, n_judged)}; the counts are None in
    result files written before they were recorded."""
    rec = {}
    for f in files:
        start = int(re.search(r"_(\d+)_(\d+)_(?:seed\d+_)?results\.pt$", f).group(1))
        r = torch.load(f, weights_only=False)
        totals = r.get("llm_total") or [None] * len(r["llm_eval"])
        corrects = r.get("llm_correct") or [None] * len(r["llm_eval"])
        for i, (v, n, c) in enumerate(zip(r["llm_eval"], totals, corrects)):
            v = float(v)
            ok = (n > 0) if n is not None else (not math.isnan(v) and v > 0)
            rec[start + i] = (v if ok else 0.0, ok, c, n)
    return rec


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--results_dir", required=True)
    p.add_argument("--node_threshold", type=int, default=50)
    p.add_argument("--n_clusters", type=int, default=5)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--cluster_types", nargs="+", default=list(NAMES))
    args = p.parse_args()

    pattern = "{d}/{ct}_{nt}_{nc}_*_seed{s}_results.pt"
    recs = {ct: load_circuits(glob.glob(pattern.format(d=args.results_dir, ct=ct, nt=args.node_threshold,
                                                      nc=args.n_clusters, s=args.seed)))
            for ct in args.cluster_types}
    ours = recs.get("ot_global_10", {})

    print(f"{args.node_threshold} nodes -> {args.n_clusters} supernodes, {args.results_dir}")
    print(f"{'method':<12} {'gradable/n':>11} {'macro (90% CI)':>22} {'micro':>16} {'p vs Our':>9}")
    for ct, rec in recs.items():
        if not rec:
            print(f"{NAMES.get(ct, ct):<12} {'no files':>11}")
            continue
        x = np.array([v for v, ok, _, _ in rec.values() if ok])
        ci = 1.645 * x.std(ddof=1) / np.sqrt(len(x))
        pstr = "-"
        if ct != "ot_global_10" and ours:
            k = [i for i in rec if i in ours and rec[i][1] and ours[i][1]]
            if len(k) > 2:
                pstr = f"{stats.ttest_rel([ours[i][0] for i in k], [rec[i][0] for i in k]).pvalue:.3f}"
        counts = [(c, n) for _, _, c, n in rec.values()]
        if all(c is not None for c, _ in counts):
            tc, tn = sum(c for c, _ in counts), sum(n for _, n in counts)
            micro = f"{tc / tn:.4f} ({tc}/{tn})" if tn else "-"
        else:
            micro = "-"
        print(f"{NAMES.get(ct, ct):<12} {f'{len(x)}/{len(rec)}':>11} {f'{x.mean():.4f} ± {ci:.4f}':>22} {micro:>16} {pstr:>9}")


if __name__ == "__main__":
    main()
