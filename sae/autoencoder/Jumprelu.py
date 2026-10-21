from .Relu import *
from torch import autograd
import einops
from typing import Optional
from .types import Stats, TopK, SAEOut
from .SAE_Wrapper import _disable_hooks

class RectangleFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return ((x > -0.5) & (x < 0.5)).float()

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[(x <= -0.5) | (x >= 0.5)] = 0
        return grad_input


class JumpReLUFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return x * (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = (x > threshold).float() * grad_output
        threshold_grad = (
            -(threshold / bandwidth)
            * RectangleFunction.apply((x - threshold) / bandwidth)
            * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth


class StepFunction(autograd.Function):
    @staticmethod
    def forward(ctx, x, threshold, bandwidth):
        ctx.save_for_backward(x, threshold, torch.tensor(bandwidth))
        return (x > threshold).float()

    @staticmethod
    def backward(ctx, grad_output):
        x, threshold, bandwidth_tensor = ctx.saved_tensors
        bandwidth = bandwidth_tensor.item()
        x_grad = torch.zeros_like(x)
        threshold_grad = (
            -(1.0 / bandwidth) * RectangleFunction.apply((x - threshold) / bandwidth) * grad_output
        )
        return x_grad, threshold_grad, None  # None for bandwidth
   
class JumpReluSAE(ReluSAE):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_names: List[str],
        dead_steps_threshold: int,
        bandwidth: float = 0.001,
        sparsity_coef: float = 0.05,
        target_l0: float = 32,
        dead_threshold: float = 1e-3,
        auxk: int | None = 256,
        standardize: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(
            d_in=d_in,
            d_sae=d_sae,
            hook_names=hook_names,
            auxk=auxk,
            dead_steps_threshold=dead_steps_threshold,
            dead_threshold=dead_threshold,
            standardize=standardize,
            sparsity_coef=sparsity_coef,
            target_l0=target_l0,
            **kwargs,
        )
        self.threshold = torch.nn.Parameter(torch.ones(self.cfg.d_sae, device=self.W_dec.device) * 0.001)
        
        self.bandwidth = bandwidth

    def encode(
        self, inputs: torch.Tensor
    ) -> tuple[torch.Tensor, TopK | None, Stats | None, torch.Tensor]:
        hidden_pre, stats = self.get_pre_act(inputs)
        latents = self.hook_sae_acts_post(torch.relu(hidden_pre * (hidden_pre > self.threshold)))
        
        self.update_last_nonzero(latents, inputs.device)
        dead, auxk = self.compute_dead_latents_and_auxk(hidden_pre)

        return latents, auxk, stats, dead

    def forward_training(self, inputs: torch.Tensor) -> SAEOut:
        hidden_pre, stats = self.get_pre_act(inputs)
        latents: Tensor = JumpReLUFunction.apply(hidden_pre, self.threshold, self.bandwidth).to(hidden_pre.dtype) # type: ignore

        self.update_last_nonzero(latents, latents.device)
        dead, auxk = self.compute_dead_latents_and_auxk(hidden_pre)

        recons = self.decode(latents, stats)
        
        l0 = StepFunction.apply(latents, self.threshold, self.bandwidth).sum(dim=-1).mean() # type: ignore
        sparsity_loss = (
            self.sparsity_coef * ((l0 / self.target_l0) - 1).pow(2)
        )
        
        auxk_recons = None
        if auxk is not None:
            auxk_latents = torch.zeros_like(latents)
            auxk_latents.scatter_(
                -1, 
                auxk.indices,
                torch.relu(auxk.values), 
            )
            auxk_recons = self.decode(auxk_latents)
            
        return SAEOut(latents, recons, auxk, auxk_recons, dead, sparsity_loss)
    

class JumpReluMatryoshka(JumpReluSAE):
    def __init__(
        self,
        d_in: int,
        d_sae: int,
        hook_names: list[str],
        dead_steps_threshold: int,
        bandwidth: float = 0.001,
        sparsity_coef: float = 0.05,
        target_l0: float = 32,
        sizes: List[int] = [1],
        auxk: int | None = 256,
        dead_threshold: float = 1e-3,
        standardize: bool = True,
        use_loss_var: bool = True,
        **kwargs
    ):
        super().__init__(
            d_in=d_in,
            d_sae=d_sae,
            hook_names=hook_names,
            bandwidth=bandwidth,
            sparsity_coef=sparsity_coef,
            target_l0=target_l0,
            auxk=auxk,
            dead_steps_threshold=dead_steps_threshold,
            dead_threshold=dead_threshold,
            standardize=standardize,
            **kwargs,
        )
        assert len(sizes) >= 1, "The sizes should be provided."
        prev_size = 0
        for size in sizes:
            assert size > 0 and size < d_sae, f"Size: {size} must be between 1 and {d_sae}."
            assert size > prev_size, f"Sizes should be sorted from low to high."
            prev_size = size
        self.sizes = sizes
        self.use_loss_var = use_loss_var
        
    def decode_size(self, latents: Tensor, size: int, stats: Stats | None = None) -> torch.Tensor:
        recons = (latents[..., :size] @ self.decoder.weight[..., :size].T) + self.pre_encoder_bias
            
        if stats is not None:
            recons = recons * stats.std + stats.mean
        return self.hook_sae_recons(recons)
        
    def forward_training(self, inputs):
        hidden_pre, stats = self.get_pre_act(inputs)
        latents: Tensor = JumpReLUFunction.apply(hidden_pre, self.threshold, self.bandwidth).to(hidden_pre.dtype) # type: ignore

        self.update_last_nonzero(latents, latents.device)
        dead, auxk = self.compute_dead_latents_and_auxk(hidden_pre)

        recons = self.decode(latents, stats)
        
        l0 = StepFunction.apply(latents, self.threshold, self.bandwidth).sum(dim=-1).mean() # type: ignore
        sparsity_loss = (
            self.sparsity_coef * ((l0 / self.target_l0) - 1).pow(2)
        )

        auxk_recons = None
        if auxk is not None:
            auxk_latents = torch.zeros_like(latents)
            auxk_latents.scatter_(
                -1, 
                auxk.indices,
                torch.relu(auxk.values), 
            )
            auxk_recons = self.decode(auxk_latents)

        return SAEOut(latents, recons, auxk, auxk_recons, dead, self.mse(inputs, latents, stats) + sparsity_loss)
    
    def mse(
        self,
        inputs: torch.Tensor,
        latents: torch.Tensor,
        stats: Stats | None = None,
    ) -> torch.Tensor:
        """
        Compute the MSE loss for the Matryoshka SAE.
        """
        mse_loss = 0.0
        for size in self.sizes:
            size_outputs = self.decode_size(latents, size, stats)
            loss = torch.mean((inputs - size_outputs) ** 2)
            if self.use_loss_var:
                mse_loss += loss / (torch.var(inputs) + 1e-8)
            else:
                mse_loss += loss
        return mse_loss # type: ignore