# Based on https://github.com/openai/sparse_autoencoder/blob/4965b941e9eb590b00b253a2c406db1e1b193942/sparse_autoencoder/train.py

from abc import ABC, abstractmethod

import torch
from torch.nn import Linear, Module, Parameter
from torch import Tensor

from .decoder import decode
from .types import Stats, TopK, SAEOut
from .SAE_Wrapper import _HookedSAE, _disable_hooks

from .Utils import standardize, unit_norm_decoder
from typing import List, Dict, Tuple, Any
from transformer_lens.hook_points import HookPoint
from dataclasses import dataclass

@dataclass
class SAEConfig:
    d_in: int
    
    d_sae: int
    
    hook_name: str
    
    hook_names: List[str]
    
    dead_steps_threshold: int
    
    dead_threshold: float
    
    auxk: int | None
    
    standardize: bool

class SAE_Template(
    ABC,
    _HookedSAE,
):
    last_nonzero: torch.Tensor
    """The number of steps since the latents have activated."""
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_names: List[str],
        dead_steps_threshold: int,
        dead_threshold: float = 1e-3,
        # TODO: Make this optional and default to a power of 2 close to d_model / 2.
        auxk: int | None = 256,
        standardize: bool = True,
    ) -> None:
        """
        Args:
            d_in (int): The number of inputs.

            d_sae (int): The number of latents.

            dead_steps_threshold (int): The number of steps after which a latent is
                flagged as dead during training.

            dead_threshold (float): The threshold for a latent to be considered
                activated. Defaults to 1e-3.

            auxk (int | None): The number of dead latents with which to model the
                reconstruction error. Defaults to 256.

            standardize (bool): Whether to standardize the inputs. Defaults to True.
        """

        super().__init__()
        self.cfg = SAEConfig(
            d_in=d_in,
            d_sae=d_sae,
            hook_name=hook_names[0],
            hook_names=hook_names,
            auxk=auxk,
            dead_steps_threshold=dead_steps_threshold,
            dead_threshold=dead_threshold,
            standardize=standardize,
        )
        self.use_error_term = False
        self.encoder = Linear(d_in, d_sae, bias=False)
        self.decoder = Linear(d_sae, d_in, bias=False)
        self.pre_encoder_bias = Parameter(torch.zeros(d_in))

        self.register_buffer("last_nonzero", torch.zeros(d_sae, dtype=torch.long))

        self.decoder.weight.data = self.encoder.weight.data.T.clone()
            
        self.decoder.weight.data = self.decoder.weight.data.T.contiguous().T
        unit_norm_decoder(self.decoder)
        
        # set up hooks
        self.hook_sae_input = HookPoint()
        self.hook_sae_acts_pre = HookPoint()
        self.hook_sae_acts_post = HookPoint()
        self.hook_sae_output = HookPoint()
        self.hook_sae_recons = HookPoint()
        self.hook_sae_error = HookPoint()
        super().setup()
    
    @abstractmethod
    def encode(self, *args, **kwargs) -> Tuple[Any, ...]:
        pass
    
    @abstractmethod
    def decode(self, *args, **kwargs) -> torch.Tensor:
        pass
    
    @abstractmethod
    def update_last_nonzero(self, *args, **kwargs) -> None:
        pass
    
    @abstractmethod
    def compute_dead_latents_and_auxk(self, *args, **kwargs) -> tuple[Tensor, TopK | None]:
        pass

    @abstractmethod
    def forward_training(self, inputs: torch.Tensor) -> SAEOut:
        pass
    
    def get_pre_act(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, Stats | None]:
        inputs = self.hook_sae_input(inputs)
        
        stats = None
        if self.cfg.standardize:
            inputs, stats = standardize(inputs)

        hidden_pre = self.hook_sae_acts_pre(self.encoder.forward(inputs - self.pre_encoder_bias))
        
        return hidden_pre, stats
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Modify the forward pass to allow gradient flows through the error term.
        '''
        latents, _, _, stats, _ = self.encode(x)
        sae_out = self.decode(latents, stats)

        if self.use_error_term:
            with torch.no_grad() if self.detach_error_term else torch.enable_grad():
                with _disable_hooks(self):
                    clead_sae, _, _, clean_stats, _ = self.encode(x)
                    x_reconstruct_clean = self.decode(clead_sae, clean_stats)
                    
            if self.disable_error_grad:
                # If disable_error_grad -> gradient will NOT flows through the error hook
                # Similar to SAE_LENS forward function
                with torch.no_grad():
                    sae_error = self.hook_sae_error(x - x_reconstruct_clean)
            else:
                # If not disable_error_grad -> gradient will flows through the error hook
                # If detach_error_term -> gradient of error term will not affect upstream components
                with torch.no_grad() if self.detach_error_term else torch.enable_grad():
                    temp_error = (x - x_reconstruct_clean)
                temp_error.requires_grad_() # allow gradient flows through the error hook
                
                sae_error = self.hook_sae_error(temp_error)
            sae_out = sae_out + sae_error
            
        return self.hook_sae_output(sae_out)
    
    def on_after_backward(self) -> list[Linear]:
        return []
    
    def on_train_end(self) -> list[Tensor]:
        return []
    
    @property
    def W_dec(self) -> Tensor:
        """The decoder weight matrix. Transpose to match sae_lens"""
        return self.decoder.weight.T
    
    @property
    def W_enc(self) -> Tensor:
        """The encoder weight matrix. Transpose to match sae_lens"""
        return self.encoder.weight.T
    
    @property
    def b_dec(self) -> Tensor:
        """The decoder bias vector."""
        return self.pre_encoder_bias
    
    @property
    def b_enc(self) -> Tensor:
        """The encoder bias vector."""
        return -self.pre_encoder_bias