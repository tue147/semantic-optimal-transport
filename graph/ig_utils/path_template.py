from abc import abstractmethod
from httpx import patch
import torch as t
from torch import Tensor
from typing import List, Tuple, Dict, Union, Callable, Any, Iterator
from tqdm import tqdm
from transformer_lens.hook_points import HookPoint
from transformer_lens import (
    utils,
    HookedTransformer,
    ActivationCache,
)
from copy import deepcopy
from collections import OrderedDict, defaultdict
from warnings import warn
from contextlib import contextmanager
from sae_lens import (
    HookedSAETransformer,
)
from functools import partial
from itertools import product
from collections import OrderedDict
from torch import sparse_coo_tensor
import einops

class IGPathTemplate:
    def __init__(self):
        pass
    
    @abstractmethod
    def __iter__(self) -> Iterator[Tuple[Tensor, Tensor]]:
        pass
    
class NonePath:
    def __init__(self):
        pass
    
    def __iter__(self) -> Iterator[Tuple[None, None]]:
        yield (None, None)