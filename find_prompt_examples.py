"""Build the prompt set for circuit compression (Section 5.2, Appendix C.4) and behavior clustering.

Chunks 1000 Pile sequences of 128 tokens, and keeps every prefix whose next token the model predicts
with cross-entropy below --target_loss nats. Writes loss_below_<target_loss>_<model>_sequences.json,
which circuit_compress_eval.py and circuit_behavior_clustering.py read (--loss_threshold).

    python find_prompt_examples.py --preset gemma_mlp
    python find_prompt_examples.py --preset gpt2_ds
"""
import argparse
import json

from Evaluation_Utils import *

import exp_config


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    exp_config.add_model_args(parser, default_preset="gemma_mlp", default_device_id=0)
    parser.add_argument("--data_path", type=str, default="monology/pile-uncopyrighted")
    parser.add_argument("--num_sequences", type=int, default=1000)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--target_loss", type=float, default=0.1)
    return parser.parse_args(argv)


def main(args: argparse.Namespace) -> None:
    cfg = exp_config.ModelConfig(args)
    device = cfg.device
    # Only the language model is needed here, not its SAEs.
    model, _, _ = load_saes(model_name=cfg.model_name, sae_types=[], device=device, dtype=t.float16,
                            layer_range=None)

    from sae.SAE_Trainer import DataConfig
    from sae.Load_Data import load_data

    data_config = DataConfig(data_path=args.data_path, batch_size=args.num_sequences, max_length=args.max_length)
    _, data_loader = load_data(cfg.tok_name, data_config)
    data_toks = extract_data(data_loader, num_batches=1)

    extracted_sequences = []
    loss_fct = t.nn.CrossEntropyLoss(reduction="none")
    with t.inference_mode():
        for clean_token in batchify(data_toks, batch_size=args.batchsize, show_progress=True):
            tokens_with_eos = model.to_tokens(model.to_string(clean_token))
            logits = model(tokens_with_eos)

            # Logits at index i predict the token at index i+1.
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = tokens_with_eos[..., 1:].contiguous()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1).to(device))
            loss = loss.view(tokens_with_eos.size(0), -1)

            for b_idx, seq_idx in zip(*t.where(loss < args.target_loss)):
                # seq_idx indexes the shifted labels, so the predicted token is at seq_idx + 1.
                true_next_token = tokens_with_eos[b_idx, seq_idx + 1]
                sequence_up_to_target = tokens_with_eos[b_idx, :seq_idx + 2]
                extracted_sequences.append({
                    "sequence": sequence_up_to_target.tolist(),
                    "string": model.to_string(sequence_up_to_target),
                    "true_next_token_string": model.to_string(true_next_token),
                    "loss": loss[b_idx, seq_idx].item(),
                })

    out_path = exp_config.default_prompt_file(cfg, str(args.target_loss))
    with open(out_path, "w") as f:
        json.dump(extracted_sequences, f, indent=4)
    print(f"[find_prompt_examples] {len(extracted_sequences)} prompts -> {out_path}")


if __name__ == "__main__":
    main(parse_args())
