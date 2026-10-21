import einops
import torch
from jaxtyping import Float
from torch.nn import Linear
from typing import Optional, Callable

from .types import Stats


def unit_norm_decoder(decoder: Linear | None) -> None:
    """Unit-normalize the decoder weight vectors."""
    if decoder is None:
        return

    decoder.weight.data /= decoder.weight.data.norm(dim=0)


# TODO: Use kernels.triton_add_mul_ if it's available
@torch.no_grad()
def unit_norm_decoder_gradient(decoder: Linear | None) -> None:
    """
    Remove the component of the gradient parallel to the decoder weight vectors.
    Assumes that the decoder weight vectors are unit-normalized.
    NOTE: Without `@torch.no_grad()`, this causes a memory leak!
    """
    if decoder is None:
        return
    
    if decoder.weight.grad is None:
        return
    scalar = einops.einsum(
        decoder.weight.grad,
        decoder.weight,
        "... n_latents n_inputs, ... n_latents n_inputs -> ... n_inputs",
    )
    vector = einops.einsum(
        scalar,
        decoder.weight,
        "... n_inputs, ... n_latents n_inputs -> ... n_latents n_inputs",
    )
    decoder.weight.grad -= vector


def standardize(
    x: Float[torch.Tensor, "... n_inputs"], eps: float = 1e-5
) -> tuple[Float[torch.Tensor, "... n_inputs"], Stats]:
    """Standardize the inputs to zero mean and unit variance."""

    mu = x.mean(dim=-1, keepdim=True)
    x = x - mu
    std = x.std(dim=-1, keepdim=True)
    x = x / (std + eps)
    return x, Stats(mu, std)
