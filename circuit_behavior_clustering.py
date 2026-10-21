"""Behaviour clusters: cache last-token hidden states over the prompt set, MiniBatchKMeans them,
then score each behaviour cluster by the entropy of the circuit features its prompts recruit.

Output: circuit_behavior_clustering_<model>_<sae>_<node_threshold>_<n_clusters>_<start>_<end>.pt

The entropy loop rebuilds a pruned circuit for every prompt in every cluster, so it is by far the
most expensive stage; use --cluster_ids to restrict it. Launcher:
scripts/circuit_behavior_clustering.sh
"""

import argparse
import os

from Evaluation_Utils import *
from math import log2
import einops
from functools import partial
from graph_visualizer import *

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import exp_config


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cluster prompts by last-token hidden state and measure circuit-feature entropy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    exp_config.add_model_args(parser, default_preset="gemma_mlp", default_device_id=2)
    exp_config.add_circuit_args(parser, default_node_threshold=50, include_edge_pruning=True)

    group = parser.add_argument_group("behaviour clustering")
    group.add_argument("--prompt_file", type=str, default=None,
                       help="Prompt set; defaults to loss_below_0.1_<model>_sequences.json.")
    group.add_argument("--loss_threshold", type=str, default="0.1", help="Used to build the default prompt file name.")
    group.add_argument("--hook_template", type=str, default="blocks.{layer}.hook_resid_post",
                       help="Hook to cache, formatted with the layer index.")
    group.add_argument("--position", type=int, default=-1, help="Token position whose hidden state is cached.")
    group.add_argument("--data_range", type=int, nargs=2, default=[0, 18000], metavar=("START", "END"),
                       help="Slice of the prompt set to cluster.")

    kmeans_group = parser.add_argument_group("kmeans")
    kmeans_group.add_argument("--n_clusters", type=int, default=500, help="Number of behaviour clusters.")
    kmeans_group.add_argument("--kmeans_batch_size", type=int, default=1000, help="MiniBatchKMeans batch size.")
    kmeans_group.add_argument("--random_state", type=int, default=42, help="MiniBatchKMeans seed.")
    kmeans_group.add_argument("--kmeans_n_init", type=str, default="auto", help="MiniBatchKMeans n_init.")

    inspect_group = parser.add_argument_group("inspection")
    inspect_group.add_argument("--show_cluster_id", type=int, default=9,
                               help="Cluster printed after fitting; -1 disables the print.")
    inspect_group.add_argument("--show_max_to_show", type=int, default=20, help="Examples printed for that cluster.")

    entropy_group = parser.add_argument_group("entropy stage")
    entropy_group.add_argument("--cluster_ids", type=int, nargs="+", default=None,
                               help="Restrict the entropy loop to these clusters; default is all of them.")
    entropy_group.add_argument("--skip_entropy", type=exp_config.str_to_bool, default=False,
                               help="Stop after kmeans; saves the labels with an empty entropy list.")

    parser.add_argument("--output_dir", type=str, default=".", help="Where the results .pt is written.")

    return parser.parse_args(argv)


def cache_last_token_states(args, model, data, device):
    """Concatenate the cached hidden state at every layer for the last token of each prompt."""
    act_cache = [t.tensor(0) for _ in range(model.cfg.n_layers)]

    def hook_fn(tens: Tensor, hook: HookPoint):
        act_cache[hook.layer()] = tens[:, args.position, ...].detach().cpu()

    start, end = args.data_range
    all_strings = []
    all_vectors = []
    with t.no_grad():
        with model.hooks(
            fwd_hooks=[
                (args.hook_template.format(layer=layer), hook_fn)
                for layer in range(model.cfg.n_layers)
            ]
        ):
            for example in tqdm(data[start:end]):
                clean_token = t.tensor([example['sequence']], device=device)
                all_strings.append(example["string"])

                model(clean_token)
                all_vectors.append(t.cat(act_cache, dim=1))

    all_vectors = t.cat(all_vectors)
    print(all_vectors.shape)
    return all_strings, all_vectors


def fit_kmeans(args, all_vectors):
    # 1. Convert PyTorch tensor to NumPy array
    # Ensure the tensor is on the CPU and converted to float32 (better for sklearn)
    X = all_vectors.cpu().numpy().astype(np.float32)  # type: ignore

    # 2. Initialize and fit MiniBatchKMeans
    kmeans = MiniBatchKMeans(
        n_clusters=args.n_clusters,
        batch_size=args.kmeans_batch_size,  # Adjust based on your memory/dataset size
        random_state=args.random_state,     # For reproducibility
        n_init=args.kmeans_n_init,
    )

    # 3. Get the cluster assignments for each vector
    cluster_labels = kmeans.fit_predict(X)

    print(f"Successfully clustered {len(X)} items into {args.n_clusters} clusters.")
    return cluster_labels


def compute_cluster_entropy(args, model, dict_saes, data, cluster_labels, device):
    """For each behaviour cluster, entropy of the (hook, feature) distribution its circuits recruit."""
    cluster_ids = args.cluster_ids if args.cluster_ids is not None else np.unique(cluster_labels)

    cluster_entropy = []
    for cluster_id in tqdm(cluster_ids):
        # Positions within the clustered slice; the original indexes `data` directly, which lines
        # up as long as --data_range starts at 0.
        indices = np.where(cluster_labels == cluster_id)[0]

        cluster_node_dict = {}
        for example_idx in tqdm(indices):
            example = data[example_idx]
            clean_token = t.tensor([example['sequence']], device=device)
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
                        if (node_hook, f_item) not in cluster_node_dict:
                            cluster_node_dict[(node_hook, f_item)] = 0
                        cluster_node_dict[(node_hook, f_item)] += 1

        # Compute entropy
        total_count = sum(cluster_node_dict.values())
        probabilities = {k: v / total_count for k, v in cluster_node_dict.items()}
        entropy = -sum(p * log2(p) for p in probabilities.values() if p > 0)

        cluster_entropy.append((cluster_id, entropy))

    return cluster_entropy


def main(args: argparse.Namespace) -> None:
    cfg = exp_config.ModelConfig(args)
    device = cfg.device

    model, dict_saes, list_saes = exp_config.load_model_and_saes(cfg)

    prompt_file = args.prompt_file or exp_config.default_prompt_file(cfg, args.loss_threshold)
    data = exp_config.load_prompt_sequences(prompt_file)

    all_strings, all_vectors = cache_last_token_states(args, model, data, device)
    cluster_labels = fit_kmeans(args, all_vectors)

    if args.show_cluster_id >= 0:
        exp_config.show_cluster_strings(
            model,
            cluster_id=args.show_cluster_id,
            strings_list=all_strings,
            labels=cluster_labels,
            max_to_show=args.show_max_to_show,
        )

    if args.skip_entropy:
        cluster_entropy = []
    else:
        cluster_entropy = compute_cluster_entropy(args, model, dict_saes, data, cluster_labels, device)
        cluster_entropy.sort(key=lambda x: x[1])

    start, end = args.data_range
    file_name = (
        f"circuit_behavior_clustering_{cfg.model_name_flat}_{cfg.sae_type_flat}_"
        f"{args.node_threshold}_{args.n_clusters}_{start}_{end}.pt"
    )
    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, file_name)

    t.save({
        "cluster_entropy": cluster_entropy,
        "string": all_strings,
        "cluster_labels": cluster_labels,
    }, out_path)
    print(f"[circuit_behavior_clustering] saved {out_path}")


if __name__ == "__main__":
    main(parse_args())
