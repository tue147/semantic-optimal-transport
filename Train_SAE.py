import torch as t
import argparse

from typing import List, Tuple, Dict, Any, Union, Callable, cast
from sae.SAE_Trainer import train, TrainSAEConfig, DataConfig
from sae.Load_Data import load_data
from itertools import product
from functools import partial
from transformer_lens.hook_points import HookPoint
from transformer_lens import HookedTransformer
from sae.Training_Utils import *

def preproc(
    toks: Tuple[t.Tensor],
    hook_name: str,
    transformer: HookedTransformer,
    device: t.device | str = "cpu",
) -> t.Tensor:
    act_cache = {}
    def hook_fn(inputs: t.Tensor, hook: HookPoint):
        act_cache[hook_name] = inputs.detach()
        
    transformer.run_with_hooks(
        toks[0],
        fwd_hooks=[(hook_name, hook_fn)], 
    )
    stacked_tensors = t.cat(list(act_cache.values())).to(device)
    return stacked_tensors

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Downstream SAE.")
    parser.add_argument('--model_name', type=str, default="gpt2-small", help="The model to train the SAE on.")
    parser.add_argument('--tok_name', type=str, default="gpt2", help="The tokenizer to encode the data.")
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--max_tokens', type=int, default=1_000_000_000, help='Maximum tokens for training')
    parser.add_argument('--layer', type=int, default=11, help='Layer to use for metric calculation')
    parser.add_argument('--hook_name', type=str, required=True, help='Hook name to use in training')
    parser.add_argument('--sae_name', type=str, default="topk", help='SAE name')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--sparsity_coef', type=float, default=1e-2, help='The sparsity coefficient')
    parser.add_argument('--expansion_factor', type=float, default=0.5, help='Expansion factor')
    parser.add_argument('--k', type=int, default=8, help='K value')
    parser.add_argument('--auxk', type=int, default=32, help='Auxiliary K value')
    parser.add_argument('--auxk_coef', type=float, default=0.05, help='Auxiliary K coefficient')
    parser.add_argument('--use_loss_var', type=str_to_bool, default=True, help='Use variance in the loss.')
    parser.add_argument('--device_id', type=int, default=7, help='Device IDs')
    parser.add_argument('--max_epochs', type=int, default=5, help='Maximum number of epochs')
    parser.add_argument('--dead_tokens_threshold', type=int, default=10_000_000, help='Dead tokens threshold')
    parser.add_argument('--log_every_n_steps', type=int, default=32, help='Log every N steps')
    parser.add_argument('--save_every_n_training_steps', type=int, default=1, help='Save the model every N training steps')
    parser.add_argument('--save_step', type=str_to_bool, default=False, help="Saving model with training steps.")
    parser.add_argument('--patience', type=int, default=1, help='patience')
    parser.add_argument('--num_workers', type=int, default=63, help='Number of workers')

    args = parser.parse_args()
    
    device = t.device(f"cuda:{int(args.device_id)}" if t.cuda.is_available() else "cpu")
    model = HookedTransformer.from_pretrained(args.model_name, device=device)
    
    # default config
    data_config = DataConfig(
        batch_size=args.batch_size,
        max_tokens=args.max_tokens,
        max_length=1024, # the tokenizer of gpt2 only allow max 1024 tokens
        num_workers=args.num_workers,
    )
    hook_name = args.hook_name
    
    default_config = TrainSAEConfig(
        sae_name=args.sae_name,
        lr=args.lr,
        sparsity_coef=args.sparsity_coef,
        hook_names=[hook_name],
        expansion_factor=args.expansion_factor,
        k=args.k,
        auxk=args.auxk,
        auxk_coef=args.auxk_coef,
        device_id=[args.device_id],
        max_epochs=args.max_epochs,
        dead_tokens_threshold=args.dead_tokens_threshold,
        log_every_n_steps=args.log_every_n_steps,
        save_every_n_training_steps=args.save_every_n_training_steps,
        patience=args.patience,
        use_loss_var=args.use_loss_var,
        monitor="loss/total",
    )
    model_name_float = args.model_name.replace("/", "_")
    default_config.save_file_name = "{}_{}_{}_{}_{}_{}_{}_{}_{}".format(
        default_config.sae_name,
        default_config.expansion_factor,
        default_config.k,
        hook_name,
        model_name_float,
        default_config.lr,
        default_config.auxk,
        default_config.auxk_coef,
        default_config.seed
    )
    
    if args.save_step:
        default_config.save_file_name += "-{step}"
        default_config.save_top_k = -1
    
    # load data
    train_loader, val_loader = load_data(args.tok_name, data_config)
    
    preprocess = partial(
        preproc, 
        hook_name=args.hook_name,
        transformer=model, 
        device=device
    )
    
    train(default_config, data_config, model, train_loader, val_loader, preprocess=preprocess)
