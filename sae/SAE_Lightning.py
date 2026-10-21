import torch as t
from torch import Tensor, kl_div
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass
from functools import partial
import os

from typing import List, Tuple, Dict, Any, Union, Callable
from transformer_lens.hook_points import HookPoint
from transformer_lens import (
    utils,
    HookedTransformer,
    HookedTransformerConfig,
)
from .metrics import (
    AuxiliaryLoss,
    MLAuxiliaryLoss,
    DeadLatents,
    MultiLayerLossWeighter,
    L1Norm,
    MSELoss,
    MLMSELoss,
    MetricWrapper,
)
from huggingface_hub import PyTorchModelHubMixin
from jaxtyping import Float, Int
from lightning.pytorch import LightningModule
import wandb
from torchmetrics import MetricCollection
from .autoencoder import (
    SAE_Template,
    HookedSAE,
    TopKSAE,
    SAEOut,
    unit_norm_decoder,
    unit_norm_decoder_gradient,
    geometric_median,
)
from .Training_Utils import *

class CustomSAE(
    LightningModule,
    PyTorchModelHubMixin,
):

    def __init__(
        self,
        sae_name: str,
        hook_names: List[str],
        model: HookedTransformer,
        expansion_factor: int | float = 16,
        d_sae: int | None = None,
        k: int = 32,
        mse_coef: float = 1.0,
        auxk: int | None = 256,
        auxk_coef: float | None = 1 / 32,
        sparsity_coef: float = 1 / 16,
        dead_tokens_threshold: int = 10_000_000,
        dead_threshold: float = 1e-3,
        lr: float = 1e-4,
        standardize: bool = True,
        max_length: int = 2048,
        batch_size: int = 1,
        use_loss_var: bool = False,
        accumulate_grad_batches: int = 64,
        dead_steps_threshold: int | None = None,
        **kwargs,
    ) -> None:
        """
        Multi-Layer Sparse Autoencoder (MLSAE) PyTorch Lightning module.
        Includes the underlying transformer.

        References:

        - [Gao et al., 2024. Scaling and evaluating sparse autoencoders.](https://cdn.openai.com/papers/sparse-autoencoders.pdf)
        - [Bricken et al., 2023. Towards Monosemanticity.](https://transformer-circuits.pub/2023/monosemantic-features)

        Args:
            model (HookedTransformer): The pretrained model.

            expansion_factor (int | float): The ratio of the number of latents to the number of
                inputs. Defaults to 16.

            k (int): The number of largest latents to keep. Defaults to 32.

            auxk (int | None): The number of dead latents with which to model the
                reconstruction error. Defaults to 256.

            auxk_coef (float | None): The coefficient of the auxiliary loss.
                Defaults to 1 / 32.

            dead_tokens_threshold (int): The number of tokens after which a latent is
                flagged as dead during training. Defaults to 10 million.

            dead_threshold (float): The threshold for a latent to be considered
                activated. Defaults to 1e-3.

            lr (float): The learning rate. Defaults to 1e-4.

            standardize (bool): Whether to standardize the inputs. Defaults to True.

            skip_special_tokens (bool): Whether to ignore special tokens.
                Defaults to True.

            max_length (int): The maximum length of a tokenized input sequence.
                Defaults to 2048.

            batch_size (int): The number of sequences in a batch. Defaults to 1.

            accumulate_grad_batches (int): The number of batches over which to
                accumulate gradients. Defaults to 64.
                
        """

        super().__init__()

        self.sae_name = sae_name
        self.hook_names = hook_names
        self.expansion_factor = expansion_factor
        self.k = k
        self.mse_coef = mse_coef
        self.auxk = auxk
        self.auxk_coef = auxk_coef
        self.sparsity_coef = sparsity_coef
        self.dead_tokens_threshold = dead_tokens_threshold
        self.dead_threshold = dead_threshold
        self.lr = lr
        self.standardize = standardize
        self.max_length = max_length
        self.batch_size = batch_size
        self.use_loss_var = use_loss_var
        self.accumulate_grad_batches = accumulate_grad_batches

        # Set the number of steps after which a latent is flagged as dead from the
        # number of tokens per batch and the number of batches per step.
        self.dead_steps_threshold = (
            dead_steps_threshold
            or self.dead_tokens_threshold
            // (self.batch_size * self.max_length * self.accumulate_grad_batches)
        )
        self.model_cfg = model.cfg
        self.n_layers = self.model_cfg.n_layers
        self.d_in = self.model_cfg.d_model
        self.d_sae = int(self.d_in * self.expansion_factor) if d_sae == None else d_sae

        self.preprocess = kwargs.get("preprocess", lambda x: x[0])
        self.save_hyperparameters(ignore=["preprocess", "model"] + list(kwargs.keys())) # ignore the preprocess function when saving
        
        if self.sae_name == "topk":
            sae = TopKSAE
        else:
            raise NotImplementedError(f"The sae {self.sae_name} is not supported.")

        self.autoencoder: SAE_Template = sae(
            d_in=self.d_in,
            d_sae=self.d_sae,
            hook_names=self.hook_names,
            k=self.k,
            dead_steps_threshold=self.dead_steps_threshold,
            dead_threshold=self.dead_threshold,
            auxk=self.auxk,
            standardize=self.standardize,
            sparsity_coef=self.sparsity_coef,
            target_l0=self.k,
            **kwargs
        )  # type: ignore

        self.mse_loss = MSELoss(self.mse_coef, use_loss_var=self.use_loss_var)
        self.aux_loss = AuxiliaryLoss(self.auxk_coef or 0.0)

    def forward_preprocess(
        self, inputs: Any
    ) -> Any:
        preprocessed = self.preprocess(inputs)
        return preprocessed

    def training_step(
        self, batch: Tuple[Float[t.Tensor, "batch pos d_in"]], batch_idx: int
    ) -> Float[t.Tensor, ""]:
        inputs = self.forward_preprocess(batch)
        
        if batch_idx == 0:
            self.autoencoder.pre_encoder_bias.data = geometric_median(inputs)

        latents, recons, auxk, auxk_recons, dead, additional_loss, log_dict = self.autoencoder.forward_training(inputs)

        mse_loss = self.mse_loss.forward(inputs=inputs, recons=recons)
        aux_loss = self.aux_loss.forward(
            inputs=inputs, recons=recons, auxk_recons=auxk_recons
        )
        loss = mse_loss + aux_loss + additional_loss

        self.log_dict(
            {
                "loss/total": loss,
                "loss/mse": mse_loss,
                "loss/auxk": aux_loss,
                "train/dead/abs": dead,
                "loss/additional_loss": additional_loss,
                **log_dict,
            }
        )

        return loss

    @t.no_grad()
    def validation_step(self, batch: Tuple[Float[t.Tensor, "batch pos d_in"]]) -> None:
        inputs = self.forward_preprocess(batch)

        latents, recons, auxk, auxk_recons, dead, additional_loss, log_dict = self.autoencoder.forward_training(inputs)

        mse_loss = self.mse_loss.forward(inputs=inputs, recons=recons)
        aux_loss = self.aux_loss.forward(
            inputs=inputs, recons=recons, auxk_recons=auxk_recons
        )
        loss = mse_loss + aux_loss + additional_loss

        self.log_dict(
            {
                "val/loss/total": loss,
                "val/loss/mse": mse_loss,
                "val/loss/auxk": aux_loss,
                "val/dead/abs": dead,
                "val/loss/additional_loss": additional_loss,
                **log_dict,
            }
        )

    @t.no_grad()
    def test_step(self, batch: Float[t.Tensor, "batch pos d_in"]) -> None:
        raise NotImplementedError("Test step not yet implemented.")

    def on_after_backward(self) -> None:
        unit_norm_decoder(self.autoencoder.decoder)
        unit_norm_decoder_gradient(self.autoencoder.decoder)
        
        for dec in self.autoencoder.on_after_backward():
            unit_norm_decoder(dec)
            unit_norm_decoder_gradient(dec)

    def on_train_end(self) -> None:
        del self.autoencoder.last_nonzero
        
        for item in self.autoencoder.on_train_end():
            del item

    def configure_optimizers(self):
        return t.optim.Adam(  # type: ignore
            self.autoencoder.parameters(), lr=self.lr, eps=6.25e-10
        )
        
    def state_dict(self, **kwargs): # remove orig_sae from state_dict (for ESAE)
        state = super().state_dict(**kwargs)
        return {k: v for k, v in state.items() if not "orig_sae" in k}
    