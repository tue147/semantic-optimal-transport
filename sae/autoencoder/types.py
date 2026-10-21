from typing import NamedTuple

import torch
from jaxtyping import Float, Int


class TopK(NamedTuple):
    """The k largest latents. Wraps 'torch.return_types.topk'."""

    values: Float[torch.Tensor, "batch pos k"]
    """The values of the k largest latents."""

    indices: Int[torch.Tensor, "batch pos k"]
    """The indices of the k largest latents."""
    
    
class SAEOut(NamedTuple):
    """The output of the autoencoder forward pass."""

    latents: torch.Tensor
    """The k largest latents."""

    recons: torch.Tensor
    """The reconstructions from the k largest latents."""

    auxk: TopK | None
    """If auxk is not None, the auxk largest dead latents."""

    auxk_recons: torch.Tensor | None
    """If auxk is not None, the reconstructions from the auxk largest dead latents."""

    dead: torch.Tensor
    """The fraction of dead latents."""
    
    addtional_loss: torch.Tensor | float
    """The additional loss to backward."""
    
    additional_log_dict: dict[str, torch.Tensor | float] = {}
    """The additional log dictionary to log."""
    

class Stats(NamedTuple):
    """Used to standardize the input activation vectors."""

    mean: torch.Tensor
    std: torch.Tensor
