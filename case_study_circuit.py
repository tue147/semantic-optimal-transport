"""Section 6.1 (Figures 1, 4, 6, 7): automated circuit compression case studies on Gemma-2-2B.

Builds the attribution circuit for one of the three Neuronpedia prompts (Appendix D.2, Table 5),
compresses it with our method (Wasserstein distance in the all-layer reference space), and writes
two interactive HTML dashboards: the full circuit and the compressed supernode graph.

    python case_study_circuit.py --circuit dallas
    python case_study_circuit.py --circuit rhythm --n_clusters 10

Launcher: scripts/case_study_circuit.sh
"""
import argparse
import os

from Evaluation_Utils import *
from graph_visualizer import *

import exp_config

# Table 5. `target_pos` is the position (in the generated sequence) of the output token the circuit
# explains. Dallas uses a corrupt prompt; the other two patch every non-BOS activation to zero.
CIRCUITS = {
    "dallas": dict(prompt="Fact: the capital of the state containing Dallas is", new_tokens=1,
                   target_pos=[11], corrupt="Fact: the city of the country containing Paris is Paris",
                   nodes=50, supernodes=7),
    "rhythm": dict(prompt="A rhyming couplet: He saw a carrot and had to grab it,\nHis hunger was like a starving",
                   new_tokens=1, target_pos=[23], corrupt=None, nodes=100, supernodes=10),
    "opposite": dict(prompt="The opposite of small is", new_tokens=1, target_pos=[6], corrupt=None,
                     nodes=40, supernodes=9),
}


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    exp_config.add_model_args(parser, default_preset="gemma_mlp", default_device_id=0)
    parser.add_argument("--circuit", choices=list(CIRCUITS), required=True)
    parser.add_argument("--node_threshold", type=int, default=None, help="Circuit size; default from Table 5.")
    parser.add_argument("--n_clusters", type=int, default=None, help="Number of supernodes; default from Table 5.")
    parser.add_argument("--max_points_per_feature", type=int, default=500)
    exp_config.add_data_args(parser, default_batchsize=64, default_max_length=128, default_num_batches=10000)
    parser.add_argument("--out_dir", type=str, default="case_studies")
    return parser.parse_args(argv)


def build_circuit(model, dict_saes, clean_token, corrupt_token, metric, node_threshold, device):
    """Attribution-patching circuit (Marks et al.): node pruning, then edge scoring."""
    fg = load_graph(model, dict_saes, "fc", True)
    fg.build_default_connection(clean_token.shape[1], token_wise=True, inter=True, intra=True)

    if corrupt_token is None:
        clean_logit, clean_cache = fg.run_model(clean_token)
        corrupt_cache = {}
        for k, v in clean_cache.items():
            corrupt_cache[k] = t.zeros_like(v)
            corrupt_cache[k][:, 0] = v[:, 0]  # keep the BOS activation
        del clean_logit, clean_cache
    else:
        corrupt_logit, corrupt_cache = fg.run_model(model.to_tokens(corrupt_token))
        del corrupt_logit

    pruner = Pruner(fg, metric, device, verbose=False)
    common = dict(prune_type="attrib", scoring_mode="less", modify_inplace=True, return_type="retained",
                  reverse_pruning=False, verbose=False, pass_through_grad=True)
    pruner(clean_token, corrupt_cache, node_threshold, cut_mode="node", threshold_type="number",
           gradient_mode="standard", transfer_grad=True, steps=10, save_node_grads=False, **common)
    pruner(clean_token, corrupt_cache, 1, cut_mode="edge", threshold_type="percen",
           edge_gradient_mode="gradient", direct_effect=False, **common)

    for store in (fg.nodes, fg.node_scores):
        for key, val in store.items():
            store[key] = val.to("cpu")
    for store in (fg.edges, fg.edge_scores):
        for val in store.values():
            for key2, val2 in val.items():
                val[key2] = val2.to("cpu")
    del corrupt_cache, pruner
    t.cuda.empty_cache()
    return fg


def main(args: argparse.Namespace) -> None:
    spec = CIRCUITS[args.circuit]
    node_threshold = args.node_threshold or spec["nodes"]
    n_clusters = args.n_clusters or spec["supernodes"]

    cfg = exp_config.ModelConfig(args)
    device = cfg.device
    model, dict_saes, list_saes = exp_config.load_model_and_saes(cfg)

    output = model.generate(spec["prompt"], max_new_tokens=spec["new_tokens"], do_sample=False, return_type="str")
    clean_token = model.to_tokens(output)
    target_pos = t.tensor(spec["target_pos"])
    print(f"[case_study] {args.circuit}: {output!r}")

    def metric(logit: Tensor, k: int = 1) -> Tensor:
        """Sum of the top-k logits at the target positions."""
        vocab_logits = logit[:, :-1][0, target_pos - 1, :]
        return t.topk(vocab_logits, k=k, dim=-1).values.sum()

    def top_tokens_and_pos(logit: Tensor, k: int = 1):
        vocab_logits = logit[:, :-1][0, target_pos - 1, :]
        top_ids = t.topk(vocab_logits, k=k, dim=-1).indices
        return [(int(p), int(idx)) for i, p in enumerate(target_pos - 1) for idx in top_ids[i]]

    fg = build_circuit(model, dict_saes, clean_token, spec["corrupt"], metric, node_threshold, device)

    data_toks, full_data_dict = exp_config.load_or_build_full_data(args, cfg, model, list_saes)

    graph_vis = SAEGraphVisualizer(
        graph_obj=fg, model=model, sae_dict=fg.dict_saes, prompt_str=model.to_string(clean_token),
        full_data_dict=full_data_dict, batchsize=args.batchsize, data_toks=data_toks,
        important_output_tokens=top_tokens_and_pos(model(clean_token)),
    )
    graph_vis.run_clustering_ot_global_concat_fast(
        n_clusters, alpha=1, lambda_pen=1.0, n_intervals=5, k_per_interval=5, buffer=10,
        max_points_per_feature=args.max_points_per_feature,
    )

    os.makedirs(args.out_dir, exist_ok=True)
    full_path = os.path.join(args.out_dir, f"{args.circuit}_circuit.html")
    compressed_path = os.path.join(args.out_dir, f"{args.circuit}_compressed.html")
    graph_vis.generate_html(full_path)
    graph_vis.generate_aggregated_html(compressed_path)
    print(f"[case_study] wrote {full_path} and {compressed_path}")


if __name__ == "__main__":
    main(parse_args())
