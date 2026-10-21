import torch
from jaxtyping import Float
from torchmetrics import Metric

class MSELoss(Metric):
    
    is_differentiable = True
    full_state_update = False

    mse: Float[torch.Tensor, ""]
    var: Float[torch.Tensor, ""]

    def __init__(self, coef: float = 1.0, use_loss_var: bool = False) -> None:
        super().__init__()
        self.add_state(
            "mse", torch.zeros(1, dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "var", torch.zeros(1, dtype=torch.float), dist_reduce_fx="sum"
        )
        self.use_loss_var = use_loss_var
        self.coef = coef

    def update(
        self,
        inputs: Float[torch.Tensor, "batch pos n_inputs"],
        recons: Float[torch.Tensor, "batch pos n_inputs"],
        **kwargs,
    ) -> None:
        if self.coef != 0:
            mean = torch.mean((recons - inputs).pow(2))
            self.mse.add_(mean)
            
            if self.use_loss_var:
                var = torch.var(inputs)
                self.var.add_(var)

    def compute(self) -> Float[torch.Tensor, ""]:
        if self.use_loss_var:
            return self.coef * self.mse / (self.var + 1e-8)
        else:
            return self.coef * self.mse
        
import torch.nn as nn

class MultiLayerLossWeighter(nn.Module):
    def __init__(self, n_layers: int, momentum: float = 0.9, eps: float = 1e-8):
        """
        Balances loss across layers by dividing each layer's loss by its 
        running average magnitude.
        
        Args:
            n_layers: Number of layers/tasks.
            momentum: Momentum for the running mean (0.9 means mostly history).
            eps: Epsilon to avoid division by zero.
        """
        super().__init__()
        self.n_layers = n_layers
        self.momentum = momentum
        self.eps = eps
        
        # Register buffer so it is saved with the model state_dict 
        # but is NOT a learnable parameter (no gradients).
        self.register_buffer("running_loss_mean", torch.ones(n_layers))
        
        # Flag to handle the very first batch
        self.register_buffer("is_initialized", torch.tensor(0, dtype=torch.bool))

    def forward(self, per_layer_loss: torch.Tensor) -> torch.Tensor:
        """
        Args:
            per_layer_loss: Tensor of shape [n_layers] containing raw loss per layer.
                            (e.g., MSE or SAE reconstruction loss)
        """
        # Detach so we don't backprop through the running mean update
        current_loss_detached = per_layer_loss.detach()

        # 1. Initialize running mean on first batch to avoid huge jumps
        if not self.is_initialized:
            self.running_loss_mean.copy_(current_loss_detached) # type: ignore
            self.is_initialized.fill_(True)
        else:
            # 2. Update EMA (Exponential Moving Average)
            # new_mean = momentum * old_mean + (1 - momentum) * current
            self.running_loss_mean.mul_(self.momentum).add_( # type: ignore
                current_loss_detached * (1 - self.momentum)
            )

        # 3. Scale the loss
        # We divide by the running mean to normalize the scale to ~1.0
        # CRITICAL: The denominator is constant (detached) w.r.t the graph.
        scaling_factor = torch.max(self.running_loss_mean).item() / (self.running_loss_mean + self.eps) # type: ignore
        
        scaled_losses = per_layer_loss * scaling_factor

        # 4. Return sum (now all layers contribute equally)
        return scaled_losses
        
class MLMSELoss(Metric):
    
    is_differentiable = True
    full_state_update = False

    mse: Float[torch.Tensor, ""]
    var: Float[torch.Tensor, ""]

    def __init__(self, n_layers: int, coef: float = 1.0, use_loss_var: bool = False) -> None:
        super().__init__()
        self.add_state(
            "mse", torch.zeros(n_layers, dtype=torch.float), dist_reduce_fx="sum"
        )
        self.add_state(
            "var", torch.zeros(n_layers, dtype=torch.float), dist_reduce_fx="sum"
        )
        self.use_loss_var = use_loss_var
        self.coef = coef

    def update(
        self,
        inputs: Float[torch.Tensor, "layer batch pos n_inputs"],
        recons: Float[torch.Tensor, "layer batch pos n_inputs"],
        **kwargs,
    ) -> None:
        if self.coef != 0:
            mean = torch.mean((recons - inputs).pow(2), dim=tuple(range(1, inputs.ndim)))
            self.mse.add_(mean)
            
            if self.use_loss_var:
                var = torch.var(inputs, dim=tuple(range(1, inputs.ndim)))
                self.var.add_(var)

    def compute(self) -> Float[torch.Tensor, "layer"]:
        if self.use_loss_var:
            return self.coef * self.mse / (self.var + 1e-8)
        else:
            return self.coef * self.mse