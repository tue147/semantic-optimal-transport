# Based on https://github.com/openai/sparse_autoencoder/blob/4965b941e9eb590b00b253a2c406db1e1b193942/sparse_autoencoder/train.py

import torch
from torch import Tensor

from .decoder import decode
from .types import Stats, TopK, SAEOut
from .Standard import SAE_Template, SAEConfig
from .SAE_Wrapper import _disable_hooks
from typing import List, Dict, Tuple, Optional, Iterable, Set
import math

from .Utils import standardize, unit_norm_decoder

class TopKSAE(SAE_Template):
    last_nonzero: torch.Tensor
    """The number of steps since the latents have activated."""

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_names: list[str],
        k: int,
        dead_steps_threshold: int,
        dead_threshold: float = 1e-3,
        # TODO: Make this optional and default to a power of 2 close to d_model / 2.
        auxk: int | None = 256,
        standardize: bool = True,
        **kwargs,
    ) -> None:
        """
        Args:
            d_in (int): The number of inputs.

            d_sae (int): The number of latents.

            k (int): The number of largest latents to keep.

            dead_steps_threshold (int): The number of steps after which a latent is
                flagged as dead during training.

            dead_threshold (float): The threshold for a latent to be considered
                activated. Defaults to 1e-3.

            auxk (int | None): The number of dead latents with which to model the
                reconstruction error. Defaults to 256.

            standardize (bool): Whether to standardize the inputs. Defaults to True.
        """

        super().__init__(
            d_in=d_in,
            d_sae=d_sae,
            hook_names=hook_names,
            auxk=auxk,
            dead_steps_threshold=dead_steps_threshold,
            dead_threshold=dead_threshold,
            standardize=standardize,
        )

        self.k = k

    def encode(
        self, inputs: torch.Tensor
    ) -> tuple[Tensor, TopK, TopK | None, Stats | None, torch.Tensor]:
        hidden_pre, stats = self.get_pre_act(inputs)

        # Find the k largest latents
        values, indices = torch.topk(
            hidden_pre, 
            k=self.k, 
            sorted=False
        )
        topk = TopK(torch.relu(values), indices)
        
        latents = torch.zeros_like(hidden_pre)
        latents = self.hook_sae_acts_post(latents.scatter_(-1, topk.indices, topk.values))

        self.update_last_nonzero(topk, inputs.device)
        dead, auxk = self.compute_dead_latents_and_auxk(hidden_pre)

        return latents, topk, auxk, stats, dead
    
    def update_last_nonzero(self, topk: TopK, device: torch.device) -> None:
        # Update the number of steps since the latents have activated
        last_nonzero = torch.zeros_like(self.last_nonzero, device=device)
        last_nonzero.scatter_add_(
            dim=0,
            index=topk.indices.reshape(-1),
            src=(
                topk.values > self.cfg.dead_threshold
            ).to(last_nonzero.dtype).reshape(-1),
        )
        self.last_nonzero *= 1 - last_nonzero.clamp(max=1)
        self.last_nonzero += 1
        
    def compute_dead_latents_and_auxk(self, latents: Tensor) -> tuple[Tensor, TopK | None]:
        # Mask the latents flagged as dead during training
        dead_mask = self.last_nonzero >= self.cfg.dead_steps_threshold
        latents = latents * dead_mask

        # Compute the fraction of dead latents
        dead = torch.sum(dead_mask, dtype=torch.float32).detach() / self.cfg.d_sae

        # If auxk is not None, find the auxk largest dead latents
        auxk = None
        if self.cfg.auxk is not None:
            values_auxk , indices_auxk = torch.topk(
                latents, 
                k=self.cfg.auxk, 
                sorted=False
            )
            auxk = TopK(values_auxk , indices_auxk)
        return dead, auxk

    def decode(self, latents: Tensor, stats: Stats | None = None) -> torch.Tensor:
        recons = (latents @ self.decoder.weight.T) + self.pre_encoder_bias
            
        if stats is not None:
            recons = recons * stats.std + stats.mean
        return self.hook_sae_recons(recons)

    def forward_training(self, inputs: torch.Tensor) -> SAEOut:
        latents, topk, auxk, stats, dead = self.encode(inputs)

        recons = self.decode(latents, stats)

        auxk_recons = None
        if auxk is not None:
            auxk_latents = torch.zeros_like(latents)
            auxk_latents.scatter_(
                -1, 
                auxk.indices,
                torch.relu(auxk.values), 
            )
            auxk_recons = self.decode(auxk_latents)

        return SAEOut(latents, recons, auxk, auxk_recons, dead, 0)
