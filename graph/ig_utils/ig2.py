import torch as t
from torch import Tensor
from typing import List, Tuple, Dict, Union, Callable, Any, Iterator
from .path_template import IGPathTemplate

class IG2Path(IGPathTemplate):
    def __init__(
        self,
        target_name: str,
        clean_act: Tensor,
        corrupt_act: Tensor,
        clean_token: Tensor,
        corrupt_logits: Tensor,
        grad_func: Callable[[Tensor, Dict[str, Tensor], Callable[[Tensor], Tensor]], Dict[str, Tensor]],
        metric: Callable[[Tensor], Tensor],
        n_steps: int = 10,
        coef: float = 0.05,
        **kwargs
    ):
        super().__init__()
        self.target_name = target_name
        self.clean_act = clean_act
        self.corrupt_act = corrupt_act
        self.clean_token = clean_token
        self.corrupt_logits = corrupt_logits
        self.grad_func = grad_func
        self.metric = metric
        self.n_steps = n_steps
        self.coef = coef
        self.act_path = self.pathgrad_construct()

    @t.no_grad()
    def pathgrad_construct(self) -> List[Tensor]:
        current_act = self.clean_act
        delta = 0
        corrupt_metric = self.metric(self.corrupt_logits)
        
        def metric_minus_corrupt(patched_logits: Tensor) -> Tensor:
            return t.abs(self.metric(patched_logits) - corrupt_metric.detach()) # ||f(x+delta) - f(x)||_2
        
        act_path = [current_act]
        for _ in range(self.n_steps-1):
            with t.set_grad_enabled(True):
                g = self.grad_func(
                    self.clean_token,
                    {self.target_name: current_act},
                    metric_minus_corrupt,
                )[self.target_name] # (b, seq, d_sae) | (b, seq, d_model) 
            
            W = t.norm(g, 2, dim=(-1, -2), keepdim=True) # (b, 1, 1)
            g = self.coef * g / W  
            delta = delta - g 
            current_act = self.clean_act + delta 
            
            act_path.append(current_act) # (b, seq, d_sae) | (b, seq, d_model)
            
        return act_path
        
    @t.no_grad()
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for path in self.act_path:
            yield (path, t.ones((1), device=self.clean_act.device)) # act path and scaler