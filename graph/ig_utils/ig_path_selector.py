import torch as t
from torch import Tensor
from typing import List, Tuple, Dict, Union, Callable, Any, Literal, Iterator
from transformer_lens import (
    ActivationCache,
)

from .idg import IDGPath
from .path_template import IGPathTemplate, NonePath
from .ig import IGPath
from .ig2 import IG2Path

class IGPathSelector:
    def __init__(
        self,
        ig_name: str,
        target_names: List[str | None],
        clean_cache: Dict[str, Tensor] | ActivationCache,
        corrupt_cache: Dict[str, Tensor] | ActivationCache,
        clean_token: Tensor,
        fwd_func: Callable[[Tensor, Dict[str, Tensor], Callable[[Tensor], Tensor]], Dict[str, Tensor]],
        grad_func: Callable[[Tensor, Dict[str, Tensor], Callable[[Tensor], Tensor]], Dict[str, Tensor]],
        metric: Callable[[Tensor], Tensor],
        n_steps: int = 10,
        **kwargs,
    ):
        self.ig_name = ig_name
        self.target_names = target_names
        if ig_name in ['ig', 'ig_input']:
            self.paths = [
                IGPath(
                    target_name=target_name,
                    clean_act=clean_cache[target_name],
                    corrupt_act=corrupt_cache[target_name],
                    metric=metric,
                    n_steps=n_steps,
                ) if target_name is not None else NonePath()
                for target_name in target_names
            ]
        elif ig_name in ['ig2', 'ig2_input']:
            corrupt_logits = kwargs.get("corrupt_logits")
            if isinstance(corrupt_logits, Tensor):
                self.paths = [
                    IG2Path(
                        target_name=target_name,
                        clean_act=clean_cache[target_name],
                        corrupt_act=corrupt_cache[target_name],
                        clean_token=clean_token,
                        corrupt_logits=corrupt_logits,
                        grad_func=grad_func,
                        metric=metric,
                        n_steps=n_steps,
                        coef=0.05,
                    ) if target_name is not None else NonePath()
                    for target_name in target_names
                ]
            else:
                raise ValueError("The corrupt_logits must be provided for ig2.")
        elif ig_name in ['idg', 'idg_input']:
            self.paths = [
                IDGPath(
                    target_name=target_name,
                    clean_act=clean_cache[target_name],
                    corrupt_act=corrupt_cache[target_name],
                    clean_token=clean_token,
                    fwd_func=fwd_func,
                    grad_func=grad_func,
                    metric=metric,
                    n_steps=n_steps,
                ) if target_name is not None else NonePath()
                for target_name in target_names
            ]
        else:
            raise
        
    def __iter__(self) -> Iterator[Tuple[Tuple[Tensor, Tensor] | Tuple[None, None], ...]]:
        for paths in zip(*self.paths):
            yield paths
        
    