"""Inspect one behaviour cluster produced by circuit_behavior_clustering.py.

For every prompt in the chosen cluster: prune a circuit, OT-cluster its nodes into supernodes,
and accumulate (a) how often each feature appears and (b) how often each feature pair lands in
the same supernode. Those co-occurrence counts are the modular feature groups.

Output: modular_<model>_<sae>_<cluster_id>_<node_threshold>_<n_clusters>_<clustering_file>.pt
Launcher: scripts/modular_feature_group.sh
"""

import argparse
import itertools
import os
from collections import defaultdict

from Evaluation_Utils import *
from math import log2
import einops
from functools import partial
from graph_visualizer import *

import exp_config


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build modular feature groups from one behaviour cluster.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    exp_config.add_model_args(parser, default_preset="gemma_mlp", default_device_id=7)
    exp_config.add_data_args(parser, default_batchsize=64, default_max_length=128, default_num_batches=10000)
    exp_config.add_circuit_args(parser, default_node_threshold=200, include_edge_pruning=False)

    group = parser.add_argument_group("behaviour cluster input")
    group.add_argument("--cluster_id", type=int, default=307, help="Behaviour cluster to inspect.")
    group.add_argument("--clustering_file", type=str, default=None,
                       help="circuit_behavior_clustering output; derived from the flags below when omitted.")
    group.add_argument("--clustering_dir", type=str, default=".", help="Directory holding the clustering file.")
    group.add_argument("--cluster_node_threshold", type=int, default=50,
                       help="node_threshold used by the clustering run (part of its file name).")
    group.add_argument("--cluster_n_clusters", type=int, default=500,
                       help="n_clusters used by the clustering run (part of its file name).")
    group.add_argument("--cluster_data_range", type=int, nargs=2, default=[0, 18000], metavar=("START", "END"),
                       help="data_range used by the clustering run (part of its file name).")
    group.add_argument("--show_max_to_show", type=int, default=20, help="Examples printed for the cluster.")

    ot_group = parser.add_argument_group("supernode clustering")
    ot_group.add_argument("--n_clusters", type=int, default=10, help="Supernodes per circuit.")
    ot_group.add_argument("--alpha", type=float, default=0.8, help="OT clustering alpha.")
    ot_group.add_argument("--lambda_pen", type=float, default=1.0, help="OT clustering penalty weight.")
    ot_group.add_argument("--n_intervals", type=int, default=5, help="Activation intervals sampled per feature.")
    ot_group.add_argument("--k_per_interval", type=int, default=5, help="Examples per interval.")
    ot_group.add_argument("--buffer", type=int, default=10, help="Context buffer around each example.")

    parser.add_argument("--output_dir", type=str, default=".", help="Where the results .pt is written.")

    return parser.parse_args(argv)


def clustering_file_name(args, cfg) -> str:
    if args.clustering_file is not None:
        return args.clustering_file
    start, end = args.cluster_data_range
    return (
        f"circuit_behavior_clustering_{cfg.model_name_flat}_{cfg.sae_type_flat}_"
        f"{args.cluster_node_threshold}_{args.cluster_n_clusters}_{start}_{end}.pt"
    )


def main(args: argparse.Namespace) -> None:
    cfg = exp_config.ModelConfig(args)
    device = cfg.device

    model, dict_saes, list_saes = exp_config.load_model_and_saes(cfg)

    data_toks, full_data_dict = exp_config.load_or_build_full_data(args, cfg, model, list_saes)

    file_name = clustering_file_name(args, cfg)
    results = t.load(os.path.join(args.clustering_dir, file_name))

    print(results["cluster_entropy"])

    cluster_id = args.cluster_id
    indices = np.where(results["cluster_labels"] == cluster_id)[0]
    cluster_str = [results["string"][idx] for idx in indices]
    exp_config.show_cluster_strings(
        model,
        cluster_id=cluster_id,
        strings_list=results["string"],
        labels=results["cluster_labels"],
        max_to_show=args.show_max_to_show,
    )

    remove_dir_file_name = os.path.basename(file_name).replace("/", "_").replace(".pt", "_")
    save_file_name = (
        f"modular_{cfg.model_name_flat}_{cfg.sae_type_flat}_{cluster_id}_"
        f"{args.node_threshold}_{args.n_clusters}_{remove_dir_file_name}.pt"
    )
    node_dict = {}

    # Initialize these OUTSIDE your main loop over `cluster_str`
    node_counts = defaultdict(int)        # Tracks N(A)
    co_occurrences = defaultdict(int)     # Tracks Co-occurrences(A, B)

    for example in tqdm(cluster_str):
        clean_token = model.to_tokens(example, prepend_bos=False)
        corrupt_token, important_token_idx = exp_config.make_corrupt_and_index(model, clean_token, device)
        metrics, metrics_top_tokens_and_pos = exp_config.make_metrics(important_token_idx)

        t.cuda.empty_cache()

        fg = load_graph(
            model,
            dict_saes,
            args.method_name,
            args.use_error_term,
        )

        pruner = exp_config.build_pruner(fg, metrics, device, args, seq_len=clean_token.shape[1])

        # prune node
        exp_config.run_node_pruning(pruner, clean_token, corrupt_token, args)

        del pruner

        t.cuda.empty_cache()

        for (node_key, _), sparse_act in fg.nodes.items():
            node_hook = node_key.name
            scores = fg.node_scores.get((node_key, _))

            if sparse_act.act is not None:
                for s, f in zip(*sparse_act.act.nonzero(as_tuple=True)):
                    f_item = f.item()
                    if (node_hook, f_item) not in node_dict:
                        node_dict[(node_hook, f_item)] = 0
                    node_dict[(node_hook, f_item)] += 1

        with suppress_print(True):
            graph_vis = SAEGraphVisualizer(
                graph_obj=fg,
                model=model,
                sae_dict=fg.dict_saes,
                prompt_str=model.to_string(clean_token),  # type: ignore
                full_data_dict=full_data_dict,
                batchsize=args.batchsize,
                data_toks=data_toks,
                important_output_tokens=metrics_top_tokens_and_pos(model(clean_token))
            )
            graph_vis.build_graph_json(skip_edges=True)  # we don't need edge for this task
            graph_vis.run_clustering_ot_fast(
                args.n_clusters,
                alpha=args.alpha,
                lambda_pen=args.lambda_pen,
                n_intervals=args.n_intervals,
                k_per_interval=args.k_per_interval,
                buffer=args.buffer,
            )

        # 1. Group features in THIS specific graph by their local cluster_id
        clusters_in_graph = defaultdict(list)

        for node in graph_vis.graph_json["nodes"]:  # type: ignore
            if node["label"] == "Err":
                continue

            feat_tuple = (node["hook"], node["feat_idx"])
            local_cluster_id = node["cluster"]

            # Track that this feature appeared in this graph at least once
            clusters_in_graph[local_cluster_id].append(feat_tuple)
            node_counts[feat_tuple] += 1

        # 2. Count pairwise co-occurrences for features in the SAME local cluster
        for local_cluster_id, feats in clusters_in_graph.items():
            # Deduplicate features within the same cluster to prevent double-counting
            unique_feats = sorted(list(set(feats)))

            # Generate all unique pairs in this cluster and increment their edge weight
            for feat_a, feat_b in itertools.combinations(unique_feats, 2):
                # Sorting the tuple ensures (A, B) and (B, A) hash to the same key
                pair = tuple(sorted([feat_a, feat_b]))
                co_occurrences[pair] += 1

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, save_file_name)
    t.save(
        {
            "noun_counts": node_counts,
            "co_occurrences": co_occurrences,
        },
        out_path
    )
    print(f"[modular_feature_group] saved {out_path}")


if __name__ == "__main__":
    main(parse_args())
