import torch as t
import argparse
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import List, Tuple, Dict, Any, Union, Callable, cast

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    
def str_to_torch_dtype(dtype_str: str) -> t.dtype:
    dtype_mapping = {
        "float16": t.float16,
        "bfloat16": t.bfloat16,
        "float32": t.float32,
        "float64": t.float64,
        "int8": t.int8,
        "int16": t.int16,
        "int32": t.int32,
        "int64": t.int64,
        "uint8": t.uint8,
        "bool": t.bool,
    }

    if dtype_str not in dtype_mapping:
        raise ValueError(f"Invalid dtype string: {dtype_str}. Supported dtypes are: {list(dtype_mapping.keys())}")

    return dtype_mapping[dtype_str]

def freeze_model(model: t.nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = False
        
def calculate_kl_div(inputs, targets, reduction='batchmean'):
    # Convert logits to probabilities using softmax
    input_log_prob = t.nn.functional.log_softmax(inputs, dim=-1)
    target_prob = t.nn.functional.softmax(targets, dim=-1)
    
    # Calculate KL divergence
    kl_div = t.nn.functional.kl_div(
        input_log_prob, 
        target_prob,                         
        reduction=reduction
    )
    
    return kl_div

def split_data(
    dataset: TensorDataset,
    train_split: float = 0.8,
    seed: int = 42,
) -> Tuple[TensorDataset, TensorDataset]:
    num_samples = len(dataset)
    train_size = int(train_split * num_samples)
    val_size = num_samples - train_size  # Remaining for test

    t.manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [train_size, val_size]
    )
    
    return train_dataset, val_dataset # type: ignore