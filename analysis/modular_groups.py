"""Section 6.2 / Appendix D.3: modular feature groups from modular_feature_group.py's output.

Builds the Jaccard similarity between circuit features,
    co_occurrence(u, v) / (count(u) + count(v) - co_occurrence(u, v)),
where co_occurrence counts the circuits in which u and v land in the same supernode, then
spectral-clusters that affinity matrix into modular groups. Features seen in fewer than
--min_occurrences circuits are dropped first. No model, no GPU:

    python analysis/modular_groups.py modular_google_gemma-2-2b_mlp_307_200_10_<...>.pt --n_groups 10
"""
import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from sklearn.cluster import SpectralClustering

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("modular_file", help="output .pt of modular_feature_group.py")
    p.add_argument("--n_groups", type=int, default=10)
    p.add_argument("--min_occurrences", type=int, default=3)
    p.add_argument("--random_state", type=int, default=42)
    p.add_argument("--out", type=str, default=None, help="JSON output; default <modular_file>_groups.json")
    args = p.parse_args()

    res = torch.load(args.modular_file, weights_only=False)
    node_counts, co_occurrences = res["noun_counts"], res["co_occurrences"]

    features = [f for f, c in node_counts.items() if c >= args.min_occurrences]
    index = {f: i for i, f in enumerate(features)}
    affinity = np.zeros((len(features), len(features)), dtype=np.float32)
    for (fa, fb), c in co_occurrences.items():
        if fa in index and fb in index:
            sim = c / (node_counts[fa] + node_counts[fb] - c)
            affinity[index[fa], index[fb]] = affinity[index[fb], index[fa]] = sim
    np.fill_diagonal(affinity, 1.0)
    print(f"[modular_groups] {len(features)} features seen in >= {args.min_occurrences} circuits")

    labels = SpectralClustering(n_clusters=args.n_groups, affinity="precomputed", assign_labels="kmeans",
                                random_state=args.random_state).fit_predict(affinity)
    groups = defaultdict(list)
    for f, lbl in zip(features, labels):
        groups[int(lbl)].append({"hook": f[0], "feature": int(f[1]), "count": int(node_counts[f])})
    for g in groups.values():
        g.sort(key=lambda d: -d["count"])

    out = args.out or os.path.splitext(args.modular_file)[0] + "_groups.json"
    with open(out, "w") as fh:
        json.dump({str(k): v for k, v in sorted(groups.items())}, fh, indent=2)
    for k, g in sorted(groups.items()):
        print(f"group {k}: {len(g)} features, e.g. " + ", ".join(f"{d['hook']}#{d['feature']}" for d in g[:5]))
    print(f"[modular_groups] wrote {out}")


if __name__ == "__main__":
    main()
