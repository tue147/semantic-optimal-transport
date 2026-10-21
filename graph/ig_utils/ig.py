import torch as t
from torch import Tensor
from typing import List, Tuple, Dict, Union, Callable, Any, Iterator
from .path_template import IGPathTemplate

def interpolate(start: Tensor, end: Tensor, frac: float,) -> Tensor:
    assert 0 <= frac <= 1, "frac must be in [0, 1]"
    return start * frac + end * (1 - frac)

class IGPath(IGPathTemplate):
    def __init__(
        self,
        target_name: str,
        clean_act: Tensor,
        corrupt_act: Tensor,
        metric: Callable[[Tensor], Tensor],
        n_steps: int = 10,
        **kwargs,
    ):
        super().__init__()
        self.target_name = target_name
        self.clean_act = clean_act
        self.corrupt_act = corrupt_act
        self.metric = metric
        self.n_steps = n_steps
        
    @t.no_grad()
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        for step in range(self.n_steps):
            frac = step / self.n_steps
            yield (
                interpolate( # act path
                    self.corrupt_act, 
                    self.clean_act,
                    frac=frac,
                ),
                t.ones((1), device=self.clean_act.device) # scaler
            )
        
    