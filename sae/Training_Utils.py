import torch as t
import argparse
from torch.utils.data import DataLoader, TensorDataset, random_split
from typing import List, Tuple, Dict, Any, Union, Callable, cast

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif value.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
    