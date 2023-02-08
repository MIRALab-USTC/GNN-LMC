import importlib
import os.path as osp

import torch

__version__ = '0.0.0'

for library in ['_relabel', '_async']:
    torch.ops.load_library(importlib.machinery.PathFinder().find_spec(
        library, [osp.dirname(__file__)]).origin)

for library in ['_relabel2hop',]:
    torch.classes.load_library(importlib.machinery.PathFinder().find_spec(
        library, [osp.dirname(__file__)]).origin)

from .data import get_data
from .history import History
from .pool import AsyncIOPool
from .metis import metis, random_subgraph, permute
from .utils import compute_micro_f1, gen_masks, dropout
from .loader import LMCLoader, SubgraphLoader, EvalSubgraphLoader
from .models import ScalableGNN

__all__ = [
    'get_data',
    'History',
    'AsyncIOPool',
    'random_subgraph',
    'metis',
    'permute',
    'compute_micro_f1',
    'gen_masks',
    'dropout',
    'LMCLoader',
    'SubgraphLoader',
    'EvalSubgraphLoader',
    'ScalableGNN',
    '__version__',
]
