"""Circuit compression benchmark: build a pruned circuit per prompt, cluster it into supernodes
with every requested method, and score each clustering with the LLM coherence eval.

Repeat runs: `--seeds 1 2 3` runs the whole sweep once per seed and appends `_seed<N>` to the
output name. With no `--seeds` (the default) a single unseeded run is written under the original
name: `{cluster_type}_{node_threshold}_{n_clusters}_{start}_{end}_results.pt`.

WARNING: the LLM eval fans out paid OpenAI calls. Launcher: scripts/circuit_compress_eval.sh
"""

import argparse
import asyncio
import os

from Evaluation_Utils import *
from math import log2
import einops
from functools import partial
from graph_visualizer import *
from test_clustering.llm_eval import *

import exp_config


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark circuit compression (supernode clustering) across methods.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    exp_config.add_model_args(parser, default_preset="gemma_mlp", default_device_id=0)
    exp_config.add_data_args(parser, default_batchsize=64, default_max_length=128, default_num_batches=10000)
    exp_config.add_circuit_args(parser, default_node_threshold=50, include_edge_pruning=True)
    exp_config.add_judge_args(parser)

    group = parser.add_argument_group("clustering benchmark")
    group.add_argument("--cluster_types", type=str, nargs="+", default=["naive"],
                       choices=sorted(exp_config.CLUSTER_TYPE_SPECS.keys()),
                       help="Clustering methods to benchmark.")
    group.add_argument("--n_clusters", type=int, default=5, help="Number of supernodes per circuit.")
    group.add_argument("--data_range", type=int, nargs=2, default=[50, 100], metavar=("START", "END"),
                       help="Slice of the prompt set to evaluate.")
    group.add_argument("--prompt_file", type=str, default=None,
                       help="Prompt set; defaults to loss_below_0.1_<model>_sequences.json.")
    group.add_argument("--loss_threshold", type=str, default="0.1", help="Used to build the default prompt file name.")

    seed_group = parser.add_argument_group("repeat runs")
    seed_group.add_argument("--seeds", type=int, nargs="+", default=None,
                            help="One full sweep per seed, each saved with a _seed<N> suffix. "
                                 "Omit for a single unseeded run under the original file name.")

    llm_group = parser.add_argument_group("LLM eval")
    llm_group.add_argument("--llm_n_cluster_feature", type=int, default=3)
    llm_group.add_argument("--llm_n_rand_feature", type=int, default=3)
    llm_group.add_argument("--llm_n_intervals", type=int, default=3)
    llm_group.add_argument("--llm_max_parallel_requests", type=int, default=20,
                           help="Capped at --n_clusters, as in the original.")

    cache_group = parser.add_argument_group("circuit cache")
    cache_group.add_argument("--use_circuit_cache", type=exp_config.str_to_bool, default=False,
                             help="Reload a saved circuit when one exists. The matching fg.save() is still "
                                  "commented out below, so this is off by default and circuits are rebuilt.")
    cache_group.add_argument("--circuit_cache_dir", type=str, default="data/circuit_file",
                             help="Directory for the per-prompt circuit .pt files.")

    parser.add_argument("--save_dir", type=str, default=None,
                        help="Output directory; defaults to results/<model>_<sae_type>_<judge>.")
    parser.add_argument("--suppress_output", type=exp_config.str_to_bool, default=False,
                        help="Silence the clustering methods' stdout.")

    return parser.parse_args(argv)


def circuit_cache_path(args, cfg, prompt_file_flat, idx):
    """Per-prompt circuit file name; encodes every hyperparameter that affects the circuit."""
    name = (
        f"circuit_file_{cfg.model_name_flat}_{cfg.sae_type_flat}_{prompt_file_flat}_{idx}_"
        f"{args.node_threshold}_{args.threshold}_{args.gradient_mode}_{args.edge_gradient_mode}_"
        f"{args.direct_effect}_{args.prune_type}_{args.cut_mode}_{args.use_error_term}_"
        f"{args.method_name}_{args.ig_steps}_{args.transfer_grad}.pt"
    )
    return os.path.join(args.circuit_cache_dir, name)


def build_or_load_circuit(args, cfg, model, dict_saes, clean_token, corrupt_token, metrics, cache_path, device):
    """Return the pruned feature graph for one prompt (node pruning then edge pruning)."""
    if args.use_circuit_cache and os.path.exists(cache_path):
        fg = load_graph(model, dict_saes, args.method_name, args.use_error_term)
        return fg.load(cache_path, model, dict_saes)

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

    # prune edge
    pruned_graph, retained_components = exp_config.run_edge_pruning(pruner, clean_token, corrupt_token, args)

    del pruner

    t.cuda.empty_cache()
    # fg.save(cache_path)  # enable together with --use_circuit_cache to reuse circuits

    return fg


def run_sweep(args, cfg, model, dict_saes, test_data, prompt_file_flat, data_toks, full_data_dict, device, seed, judge):
    """One full pass over the prompt slice for every cluster type. Returns the metric dict."""
    exp_config.set_seed(seed)

    # The `llm` cluster type reads each node's act/logits HTML; every other method and the eval
    # ignore it, so we can skip building those strings when llm clustering isn't requested.
    compute_html = "llm" in args.cluster_types

    all_results = {
        cluster_type: {
            "llm_eval": [],       # per-prompt overall_accuracy (mean-of-circuits / "macro" aggregation)
            "llm_correct": [],    # per-prompt total_correct  \ enable the pooled ("micro") accuracy:
            "llm_total": [],      # per-prompt total_predictions/ sum(correct)/sum(total) across circuits
        }
        for cluster_type in args.cluster_types
    }

    for idx, example in enumerate(tqdm(test_data, total=len(test_data), desc="Processing")):
        cache_path = circuit_cache_path(args, cfg, prompt_file_flat, idx)

        clean_token = t.tensor([example['sequence']], device=device)
        corrupt_token, important_token_idx = exp_config.make_corrupt_and_index(model, clean_token, device)
        metrics, metrics_top_tokens_and_pos = exp_config.make_metrics(important_token_idx)

        fg = build_or_load_circuit(
            args, cfg, model, dict_saes, clean_token, corrupt_token, metrics, cache_path, device
        )

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

        # Build the (expensive) per-node graph json ONCE per prompt and share it across every
        # cluster type; each clustering method reuses this memoized graph and only overwrites
        # the cluster labels.
        graph_vis.build_graph_json(compute_html=compute_html)

        for cluster_type in args.cluster_types:
            with suppress_print(args.suppress_output):
                exp_config.run_clustering_by_type(graph_vis, cluster_type, args.n_clusters)

                graph_data = graph_vis.build_graph_json()

                # 1. Map nodes to clusters and filter valid clustered nodes
                node_to_cluster = {}
                for n in graph_data['nodes']:
                    if n.get('cluster') is not None:
                        node_to_cluster[n['id']] = n['cluster']

                if not node_to_cluster:
                    break

                out = asyncio.run(llm_clustering_eval(
                    graph_vis,
                    judge=judge,
                    n_cluster_feature=args.llm_n_cluster_feature,
                    n_rand_feature=args.llm_n_rand_feature,
                    n_intervals=args.llm_n_intervals,
                    max_parallel_requests=min(args.n_clusters, args.llm_max_parallel_requests)
                ))
                all_results[cluster_type]['llm_eval'].append(out['overall_accuracy'])  # type: ignore
                all_results[cluster_type]['llm_correct'].append(int(out['total_correct']))  # type: ignore
                all_results[cluster_type]['llm_total'].append(int(out['total_predictions']))  # type: ignore

                # On-the-fly running report: the per-circuit mean ("macro") next to the pooled
                # accuracy ("micro" = sum correct / sum total across circuits).
                print_running_metrics(cluster_type, all_results[cluster_type])

    return all_results


def pooled_accuracy(correct_list, total_list):
    """Micro / pooled LLM accuracy: sum(correct) / sum(total) across circuits (e.g. 3/5 & 4/10 ->
    7/15), as opposed to the mean of the per-circuit accuracies. Returns (acc, tot_correct, tot)."""
    tc, tp = int(sum(correct_list)), int(sum(total_list))
    return ((tc / tp) if tp > 0 else float("nan")), tc, tp


def print_running_metrics(cluster_type, res):
    """On-the-fly running averages for one cluster type."""
    n = len(res.get("llm_eval", []))
    macro = float(np.nanmean(res["llm_eval"])) if res.get("llm_eval") else float("nan")  # skips ungradable
    micro, tc, tp = pooled_accuracy(res.get("llm_correct", []), res.get("llm_total", []))
    print(f"[circuit_compress_eval] running {cluster_type} (n={n}): "
          f"llm macro(mean-of-circuits)={macro:.4f} | llm micro(pooled)={micro:.4f} ({tc}/{tp})")


def main(args: argparse.Namespace) -> None:
    cfg = exp_config.ModelConfig(args)
    device = cfg.device

    judge = exp_config.resolve_judge(args.judge, args.judge_effort)

    model, dict_saes, list_saes = exp_config.load_model_and_saes(cfg)

    prompt_file = args.prompt_file or exp_config.default_prompt_file(cfg, args.loss_threshold)
    data = exp_config.load_prompt_sequences(prompt_file)
    prompt_file_flat = os.path.basename(prompt_file).replace(".", "_")

    # Default output dir is judge-scoped so different judges don't overwrite each other; an
    # explicit --save_dir is used verbatim. Run file names inside are unchanged.
    save_dir = args.save_dir if args.save_dir is not None else cfg.results_dir_for_judge(judge["name"])
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(args.circuit_cache_dir, exist_ok=True)

    data_toks, full_data_dict = exp_config.load_or_build_full_data(args, cfg, model, list_saes)

    start, end = args.data_range
    test_data = data[start:end]

    seeds = args.seeds if args.seeds is not None else [None]

    for seed in seeds:
        if seed is not None:
            print(f"\n[circuit_compress_eval] ===== seed {seed} =====")

        all_results = run_sweep(
            args, cfg, model, dict_saes, test_data, prompt_file_flat,
            data_toks, full_data_dict, device, seed, judge,
        )

        seed_tag = "" if seed is None else f"_seed{seed}"
        for cluster_type, results in all_results.items():
            print("Cluster type:", cluster_type)
            path_file = f"{cluster_type}_{args.node_threshold}_{args.n_clusters}_{start}_{end}{seed_tag}_results.pt"
            for key, val in results.items():
                print(f"metric: {key}, mean: {np.nanmean(val)}, std: {np.nanstd(val)}")
            n_ungradable = int(sum(1 for v in results.get('llm_eval', []) if v != v))
            print(f"metric: llm_eval ungradable circuits (NaN, excluded from the mean): {n_ungradable}")
            # Pooled ("micro") accuracy alongside the per-circuit mean llm_eval ("macro").
            pooled, tc, tp = pooled_accuracy(results.get('llm_correct', []), results.get('llm_total', []))
            print(f"metric: llm_eval_pooled (micro = correct/total across circuits): {pooled}  ({tc}/{tp})")
            print()
            t.save(results, save_dir + "/" + path_file)


if __name__ == "__main__":
    main(parse_args())
