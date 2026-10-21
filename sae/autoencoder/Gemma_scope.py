import torch
from torch.nn import Linear, Module, Parameter
from torch import Tensor

from .decoder import decode
from .SAE_Wrapper import _HookedSAE, _disable_hooks
from typing import List, Dict, Tuple, Any
from transformer_lens.hook_points import HookPoint
from dataclasses import dataclass

@dataclass
class GemmaScopeConfig:
    d_in: int
    
    d_sae: int
    
    hook_name: str
    
    hook_names: List[str]

class GemmaScope(
    _HookedSAE,
):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_names: List[str],
        affine_skip_connection: bool = False
    ) -> None:

        super().__init__()
        self.cfg = GemmaScopeConfig(
            d_in=d_in,
            d_sae=d_sae,
            hook_name=hook_names[0],
            hook_names=hook_names,
        )
        self.use_error_term = False
        self.w_enc = Parameter(torch.zeros(d_in, d_sae))
        self.w_dec = Parameter(torch.zeros(d_sae, d_in))
        self.threshold = Parameter(torch.zeros(d_sae))
        self.b_enc = Parameter(torch.zeros(d_sae))
        self.b_dec = Parameter(torch.zeros(d_in))
        if affine_skip_connection:
            self.affine_skip_connection = Parameter(torch.zeros(d_in, d_in))
        else:
            self.affine_skip_connection = None
        
        # set up hooks
        self.hook_sae_input = HookPoint()
        self.hook_sae_acts_pre = HookPoint()
        self.hook_sae_acts_post = HookPoint()
        self.hook_sae_output = HookPoint()
        self.hook_sae_recons = HookPoint()
        self.hook_sae_error = HookPoint()
        super().setup()
    
    def encode(self, input_acts):
        input_acts = self.hook_sae_input(input_acts)
        pre_acts = self.hook_sae_acts_pre(input_acts @ self.w_enc + self.b_enc)
        mask = (pre_acts > self.threshold)
        acts = mask * torch.nn.functional.relu(pre_acts)
        return self.hook_sae_acts_post(acts)

    def decode(self, acts):
        return self.hook_sae_recons(acts @ self.w_dec + self.b_dec)
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        '''
        Modify the forward pass to allow gradient flows through the error term.
        '''
        latents = self.encode(x)
        sae_out = self.decode(latents)

        if self.use_error_term:
            with torch.no_grad() if self.detach_error_term else torch.enable_grad():
                with _disable_hooks(self):
                    clead_sae = self.encode(x)
                    x_reconstruct_clean = self.decode(clead_sae)
                    
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
            
        if self.affine_skip_connection is not None:
            return self.hook_sae_output(sae_out + x @ self.affine_skip_connection)
            
        return self.hook_sae_output(sae_out)
    
    @property
    def W_dec(self) -> Tensor:
        return self.w_dec
    
    @property
    def W_enc(self) -> Tensor:
        return self.w_enc