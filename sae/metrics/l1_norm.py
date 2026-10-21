import torch
from jaxtyping import Float, Int64
from torchmetrics import Metric


class L1Norm(Metric):
    """L1 norm. Average sum of absolute latent activations."""

    is_differentiable = False
    full_state_update = False

    abs: Float[torch.Tensor, ""]
    """Layerwise sum of absolute latent activations."""

    tokens: Int64[torch.Tensor, ""]
    """Layerwise count of tokens."""

    def __init__(self) -> None:
        super().__init__()
        self.add_state(
            "abs", torch.zeros(1, dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "tokens", torch.zeros(1, dtype=torch.int64), dist_reduce_fx="sum"
        )

    @torch.no_grad()
    def update(
        self, values: Float[torch.Tensor, "batch pos k"], **kwargs
    ) -> None:
        sum_ = torch.sum(torch.abs(values))
        self.abs.add_(sum_)
        self.tokens.add_(values.shape[0] * values.shape[1])

    @torch.no_grad()
    def compute(self) -> Float[torch.Tensor, ""]:
        return self.abs / self.tokens
