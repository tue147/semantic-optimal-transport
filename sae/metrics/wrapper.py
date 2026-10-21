from functools import partial

import torch
from jaxtyping import Float
from torchmetrics import ClasswiseWrapper, Metric


# Based on https://github.com/ai-safety-foundation/sparse_autoencoder/blob/b6ba6cb7c90372cb5462855c21e5f52fc9130557/sparse_autoencoder/metrics/wrappers/classwise.py
class MetricWrapper(ClasswiseWrapper):
    def __init__(self, metric: Metric, labels: list[str] | None = None, prefix: str | None = None) -> None:
        super().__init__(metric, labels=labels, prefix=prefix)

    def _convert_output(self, x: Float[torch.Tensor, "layer"]) -> dict:
        if x.ndim == 0:
            x = x.unsqueeze(0)
        metrics = super()._convert_output(x)
        return metrics
