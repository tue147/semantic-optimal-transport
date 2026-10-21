from sae_lens import HookedSAETransformer
import torch as t
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import einops
from functools import partial
from typing import List, Tuple, Dict, Any, Union, Callable, cast
from transformer_lens import (
    utils,
    HookedTransformer,
)
from .SAE_Lightning import *

import os
import lightning as pl
from lightning.fabric.plugins.precision.precision import _PRECISION_INPUT
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
import wandb
import math

def initialize(seed: int) -> None:
    # Deterministic for reproducibility
    seed_everything(seed=seed, workers=True)

    # Fork processes via multiprocessing in Python
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Avoid PyTorch DataLoader "too many open files"
    t.multiprocessing.set_sharing_strategy("file_system")

    # Improve matmul performance
    t.set_float32_matmul_precision("high")
    
class DataConfig:
    
    def __init__(self, **kwargs):
        
        #
        # DATA PARAMETERS
        #
        
        self.datapath: str = "monology/pile-uncopyrighted"
        """The path to a HuggingFace text dataset."""

        self.max_length: int = 2048
        """The maximum length of a tokenized input sequence."""

        self.batch_size: int = 32
        """The number of sequences in a batch."""

        self.max_tokens: float = 1_000_000_000
        """The maximum number of tokens to train on."""

        self.num_workers: int = 0
        """The number of workers to use for data loading."""
        
        # Update only the keys present in kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def max_steps(self) -> int:
        """The maximum number of batches to train on."""
        return math.ceil(self.max_tokens / (self.batch_size * self.max_length))

class TrainSAEConfig:
    
    def __init__(self, **kwargs):
        # 
        # MODEL PARAMETERS
        #
        
        self.sae_name: str = "topk"
        """The type of SAE to use."""
        
        self.hook_names: List[str] = ["blocks.0.resid_pre"]
        
        self.dead_tokens_threshold: int = 10_000_000
        """The number of tokens after which a latent is flagged as dead during training."""

        self.expansion_factor: int | float = 1
        """The ratio of the number of latents to the number of inputs."""
        
        self.d_sae: int | None = None
        """The number of latents. If None, it is set to the number of inputs. If this is None, the
        expansion factor is used to determine the number of latents."""

        self.k: int = 16
        """The number of largest latents to keep."""
        
        self.mse_coef: float = 1.0
        """The coefficient of the MSE loss."""

        self.auxk: int | None = 256
        """The number of dead latents with which to model the reconstruction error."""

        self.auxk_coef: float | None = 1 / 16
        """The coefficient of the auxiliary loss."""
        
        self.sparsity_coef: float = 1 / 16
        """The coefficient of the sparsity loss."""

        self.dead_threshold: float = 1e-3
        """The threshold activation for a latent to be considered activated."""

        self.standardize: bool = True
        """Whether to standardize the inputs."""
        
        
        #
        # TRAINING PARAMETERS
        #
        
        # TODO: Make this optional and default to the scaling law from Gao et al [2024].
        self.lr: float = 1e-4
        """The learning rate."""
        
        self.use_loss_var: bool = True
        """Whether to use the variance in the loss."""
        
        self.device_id: int | str | List[int] = [3]
        """The number of devices to use for training."""
        
        self.precision: str = "16-mixed"
        """The precision of the training parameters."""

        self.accumulate_grad_batches: int = 1
        """The number of batches over which to accumulate gradients."""

        self.log_every_n_steps: int | None = 256
        """The number of training steps between logging metrics."""

        self.val_check_interval: int | float | None = None
        """The number of training batches between validation steps."""
        
        self.max_epochs: int = 2
        """The maximum number of training epochs."""
        
        self.min_delta: float = 0.005
        """The minimum change in the early stopping."""
        
        self.patience: int = 1
        """The number of epochs to wait before early stopping."""
        
        self.monitor: str = "val/loss/total"
        """The metric to monitor for early stopping."""
        
        self.early_stopping_mode: str = "min"
        """The mode for early stopping."""
        
        #
        # MICS
        #
        
        self.seed: int = 42
        """The seed for global random state."""

        self.project: str | None = "Subspace-Multilayer-SAE"
        """The Weights & Biases project name."""

        self.run: str | None = "Default"
        """The Weights & Biases run name."""
        
        self.save_model_path: str = "data/checkpoints/"
        # self.save_model_path: str = "checkpoints/"
        """The path to save the trained model."""
        
        self.save_every_n_training_steps: int | None = 1000
        """Save the model every n training steps"""
        
        self.save_every_n_epochs: int | None = None
        """Save the model every n epochs"""
        
        self.save_top_k: int = 1
        """The number of best models to save."""
        
        # Update only the keys present in kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
            
        if self.save_every_n_epochs is not None:
            self.save_every_n_training_steps = None
        
        # Update the save file name
        self.save_file_name: str = "{}_{}_{}_{}_{}_{}".format(
            self.expansion_factor,
            self.k,
            self.lr,
            self.auxk,
            self.auxk_coef,
            self.seed,
        )
        """The name of the saved model file."""
        
class SaveAutoencodersCallback(pl.Callback):
    def __init__(self, save_dir):
        self.save_dir = save_dir

    def on_train_end(self, trainer, pl_module):
        pl_module.save_autoencoders(self.save_dir) # type: ignore

    def on_save_checkpoint(self, trainer, pl_module, checkpoint):
        pl_module.save_autoencoders(self.save_dir) # type: ignore

def train(
    config: TrainSAEConfig, 
    data_cfg: DataConfig,
    model: HookedTransformer | HookedSAETransformer,
    train_dataloader: DataLoader, 
    val_dataloader: DataLoader,
    train_mode: str = "sae",
    **kwargs,
) -> None:

    initialize(config.seed)

    if train_mode == "sae":
        customMLSAE= CustomSAE(
            config.sae_name,
            config.hook_names,
            model,
            config.expansion_factor,
            config.d_sae,
            config.k,
            config.mse_coef,
            config.auxk,
            config.auxk_coef,
            config.sparsity_coef,
            config.dead_tokens_threshold,
            config.dead_threshold,
            config.lr,
            config.standardize,
            data_cfg.max_length,
            data_cfg.batch_size,
            config.use_loss_var,
            config.accumulate_grad_batches,
            **kwargs,
        )
    elif train_mode == "multisae":
        customMLSAE= CustomMultiSAE(
            config.sae_name,
            config.hook_names,
            model,
            config.expansion_factor,
            config.d_sae,
            config.k,
            config.mse_coef,
            config.auxk,
            config.auxk_coef,
            config.sparsity_coef,
            config.dead_tokens_threshold,
            config.dead_threshold,
            config.lr,
            config.standardize,
            data_cfg.max_length,
            data_cfg.batch_size,
            config.use_loss_var,
            config.accumulate_grad_batches,
            **kwargs,
        )
    else:
        raise ValueError(f"{train_mode} is not supported.")
    
    early_stop_callback = EarlyStopping(
        monitor=config.monitor,  
        min_delta=config.min_delta,      
        patience=config.patience,
        verbose=True,
        mode=config.early_stopping_mode,
    )
    checkpoint_callback = ModelCheckpoint(
        monitor=config.monitor,
        dirpath=config.save_model_path, 
        filename=config.save_file_name, 
        save_top_k=config.save_top_k, 
        mode=config.early_stopping_mode,
        save_weights_only=True, 
        enable_version_counter=False,
        every_n_train_steps=config.save_every_n_training_steps,
        every_n_epochs=config.save_every_n_epochs,
    )

    wandb.login()  # type: ignore
    print("Logged in to wandb.")
    trainer = Trainer(
        max_epochs=config.max_epochs,
        precision=cast(_PRECISION_INPUT, config.precision),
        logger=WandbLogger(
            name=config.save_file_name,
            save_dir="wandb_logs",
            project=config.project,
            log_model=False,
        ),
        max_steps=data_cfg.max_steps,
        val_check_interval=config.val_check_interval,
        log_every_n_steps=config.log_every_n_steps,
        accumulate_grad_batches=config.accumulate_grad_batches,
        accelerator="gpu",
        devices=config.device_id,
        callbacks=[early_stop_callback, checkpoint_callback],
    )
    print("Starting training...")
    trainer.fit(
        customMLSAE,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )
    print("Training complete.")
    wandb.finish()