import math
from dataclasses import dataclass

import torch
from datasets import IterableDataset, load_dataset
from datasets.formatting.formatting import LazyBatch
from jaxtyping import Int
from torch import Tensor
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, PreTrainedTokenizerBase
from typing import Callable
from .SAE_Trainer import DataConfig

def collate_fn(batch) -> tuple[torch.Tensor]:
    return (torch.stack([item["input_ids"] for item in batch]),)

def concat_and_tokenize(
    dataset: IterableDataset,
    tokenizer: PreTrainedTokenizerBase,
    max_length: int,
) -> IterableDataset:
    return dataset.map(
        _concat_and_tokenize,
        batched=True,
        # Large batch size minimizes the number of tokens dropped
        batch_size=1024,
        # TODO: Column names are not always available
        remove_columns=dataset.column_names or ["text", "meta"],
        fn_kwargs={"tokenizer": tokenizer, "max_length": max_length},
    ).with_format("torch")


# Based on https://github.com/EleutherAI/sae/blob/19d95a401e9d17dbf7d6fb0fa7a91081f1b0d01f/sae/data.py
def _concat_and_tokenize(
    batch: LazyBatch, tokenizer: PreTrainedTokenizerBase, max_length: int
) -> dict:
    output = tokenizer(
        # Concatenate the batch of text with the EOS token
        tokenizer.eos_token.join([""] + batch["text"]),  # type: ignore
        truncation=True,
        max_length=max_length,
        return_attention_mask=False,
        return_overflowing_tokens=True,
    )

    overflowing_tokens = output.pop("overflowing_tokens", None)
    _ = output.pop("overflow_to_sample_mapping", None)

    # Split the overflowing tokens into sequences of the maximum length
    if overflowing_tokens is not None:
        output["input_ids"] += [
            overflowing_tokens[i * max_length : (i + 1) * max_length]
            for i in range(math.ceil(len(overflowing_tokens) / max_length))
        ]  # type: ignore

    # Drop the last batch, which is probably incomplete
    return {k: v[:-1] for k, v in output.items()}


def get_dataloader(
    dataset: IterableDataset,
    model_name: str,
    max_length: int,
    batch_size: int,
    num_workers: int = 1,
    collate_fn: Callable | None = collate_fn,
) -> DataLoader[Int[Tensor, "batch pos"]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Constrain the maximum length of a tokenized input sequence
    max_length = min(tokenizer.model_max_length, max_length)

    return DataLoader(
        concat_and_tokenize(dataset, tokenizer, max_length), # type: ignore
        batch_size=batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

def get_train_dataloader(
    path: str,
    model_name: str,
    max_length: int,
    batch_size: int,
    num_workers: int = 1,
    collate_fn: Callable | None = collate_fn  # Function to apply transformations
) -> DataLoader[tuple[torch.Tensor]]:
    dataset = load_dataset(path, split="train", streaming=True)
    return get_dataloader(
        dataset, # type: ignore
        model_name, 
        max_length, 
        batch_size, 
        num_workers,
        collate_fn,
    )


def get_val_dataloader(
    path: str, 
    model_name: str, 
    max_length: int, 
    batch_size: int, 
    num_workers: int = 1,
    collate_fn: Callable | None = collate_fn  # Function to apply transformations
) -> DataLoader[tuple[torch.Tensor]]:
    dataset = load_dataset(
        path, data_files={"validation": "val.jsonl.zst"}, split="validation", streaming=True
    )
    return get_dataloader(
        dataset, # type: ignore
        model_name,
        max_length,
        batch_size,
        num_workers,
        collate_fn,
    )
    
def get_test_dataloader(
    path: str, 
    model_name: str, 
    max_length: int, 
    batch_size: int, 
    num_workers: int = 1,
    collate_fn: Callable | None = collate_fn  # Function to apply transformations
) -> DataLoader[tuple[torch.Tensor]]:
    dataset = load_dataset(
        path, data_files={"test": "test.jsonl.zst"}, split="test", streaming=True
    )
    return get_dataloader(
        dataset, # type: ignore
        model_name,
        max_length,
        batch_size,
        num_workers,
        collate_fn,
    )

def load_data(
    model_name: str, config: DataConfig, collate_fn: Callable | None = collate_fn,
) -> tuple[DataLoader, DataLoader]:
    train_dataloader = get_train_dataloader(
        config.datapath,
        model_name,
        config.max_length,
        config.batch_size,
        config.num_workers or 1,
        collate_fn,
    )

    val_dataloader = get_val_dataloader(
        config.datapath,
        model_name,
        config.max_length,
        config.batch_size,
        config.num_workers or 1,
        collate_fn,
    )

    return train_dataloader, val_dataloader