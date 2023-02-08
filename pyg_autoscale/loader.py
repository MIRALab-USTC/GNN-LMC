from typing import NamedTuple, List, Tuple, Dict

import time

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torch_sparse import SparseTensor
from torch_geometric.data import Data

import pyg_autoscale.logger as logger
from .utils import index2mask
from .score_func import *
from torch_geometric.nn.conv.gcn_conv import gcn_norm

relabel_fn = torch.ops.torch_geometric_autoscale.relabel_one_hop
Relabel2hop = torch.classes.torch_geometric_autoscale.Relabel2hop

class SubData(NamedTuple):
    data: Data
    batch_size: int
    n_id: Tensor  # The indices of mini-batched nodes
    offset: Tensor  # The offset of contiguous mini-batched nodes
    count: Tensor  # The number of contiguous mini-batched nodes
    lmc_params: Dict

    def to(self, *args, **kwargs):
        return SubData(self.data.to(*args, **kwargs), self.batch_size,
                       self.n_id, self.offset, self.count, self.lmc_params)

class LMCLoader(DataLoader):
    r"""A simple subgraph loader that, given a pre-partioned :obj:`data` object,
    generates subgraphs from mini-batches in :obj:`ptr` (including their 1-hop
    neighbors)."""
    def __init__(self, data: Data, ptr: Tensor, batch_size: int = 1,
                    compensate: bool = True, rm1hop:  bool = True, merge_cluster: bool = True, score_func_name: str = 'linear',
                    bipartite: bool = True, log: bool = True, **kwargs):

        if merge_cluster:
            ptr = ptr[::batch_size]
            if int(ptr[-1]) != data.num_nodes:
                ptr = torch.cat([ptr, torch.tensor([data.num_nodes])], dim=0)
            batch_size = 1

        self.data = data
        self.ptr = ptr
        self.bipartite = bipartite
        self.log = log
        self.compensate = compensate
        self.merge_cluster = merge_cluster
        self.score_func_name = score_func_name
        self.deg_subgraph = torch.zeros(data.num_nodes, dtype=torch.int64)
        self.deg = self.data.adj_t.storage.rowcount()

        assert merge_cluster == 0 or batch_size != 0
        rowptr, col, value = self.data.adj_t.csr()
        if value is None:
            value = torch.ones_like(col)
            self.value_none = True
        else:
            self.value_none = False
        self.relabel_2hop = Relabel2hop(rowptr, col, value)

        n_id = torch.arange(data.num_nodes)
        batches = n_id.split((ptr[1:] - ptr[:-1]).tolist())
        batches = [(i, batches[i]) for i in range(len(batches))]

        if batch_size > 1:
            super().__init__(batches, batch_size=batch_size,
                             collate_fn=self.compute_subgraph, **kwargs)

        else:  # If `batch_size=1`, we pre-process the subgraph generation:
            if log:
                t = time.perf_counter()
                print('Pre-processing subgraphs...', end=' ', flush=True)

            data_list = list(
                DataLoader(batches, collate_fn=self.compute_subgraph,
                           batch_size=batch_size, **kwargs))

            if log:
                print(f'Done! [{time.perf_counter() - t:.2f}s]')

            super().__init__(data_list, batch_size=batch_size,
                             collate_fn=lambda x: x[0], **kwargs)


    def compute_subgraph(self, batches: List[Tuple[int, Tensor]]) -> SubData:
        batch_ids, n_ids = zip(*batches)
        
        n_id = torch.cat(n_ids, dim=0)
        batch_id = torch.tensor(batch_ids)

        # We collect the in-mini-batch size (`batch_size`), the offset of each
        # partition in the mini-batch (`offset`), and the number of nodes in
        # each partition (`count`)
        batch_size = n_id.numel()
        offset = self.ptr[batch_id]
        count = self.ptr[batch_id.add_(1)].sub_(offset)
        
                
        rowptr, col, value, n_id, lmc_params = self.lmc_sampler(n_id, self.bipartite,)
        lmc_params['num_cluster'] = (len(self.ptr)-1) / len(batch_ids)
        
        adj_t = SparseTensor(rowptr=rowptr, col=col, value=value,
                             sparse_sizes=(rowptr.numel() - 1, n_id.numel()),
                             is_sorted=True)

        data = self.data.__class__(adj_t=adj_t)
        for k, v in self.data:
            if isinstance(v, Tensor) and v.size(0) == self.data.num_nodes:
                data[k] = v.index_select(0, n_id)

        return SubData(data, batch_size, n_id, offset, count, lmc_params)

    def __repr__(self):
        return f'{self.__class__.__name__}()'
    
    def lmc_sampler(self, n_id, bipartite):
        value_1hop, n_id_result = self.relabel_2hop.relabel_one_hop(n_id, bipartite,)

        batch_size = n_id.numel()
        n_id_outofbatch = n_id_result[batch_size:]

        rowptr_result, col_result, value_out = self.relabel_2hop.relabel_one_hop_compensation_rm2hop(n_id_outofbatch, bipartite)

        if self.value_none:
            value_result = None
        else:
            value_result = torch.cat([value_1hop, value_out], dim=0)
            value_result = value_result[:rowptr_result[-1]-rowptr_result[0]]

        deg = rowptr_result[1:] - rowptr_result[:-1]

        alpha = deg[batch_size:]/self.deg[n_id_result[batch_size:]]

        return rowptr_result, col_result, value_result, n_id_result, {
            'onehop_size': n_id_result.numel(),
            'pull_mask_id': torch.where(alpha != 1)[0]+batch_size,
            'pull_nid': n_id_result[batch_size:][alpha != 1],
            'alpha': eval(self.score_func_name)(alpha[alpha != 1]),
        }

class SubgraphLoader(DataLoader):
    r"""A simple subgraph loader that, given a pre-partioned :obj:`data` object,
    generates subgraphs from mini-batches in :obj:`ptr` (including their 1-hop
    neighbors)."""
    def __init__(self, data: Data, ptr: Tensor, batch_size: int = 1,
                    compensate: bool = True, rm1hop:  bool = False, merge_cluster: bool = False, score_func_name: str = None,
                    bipartite: bool = True, log: bool = True, **kwargs):
        if merge_cluster:
            ptr = ptr[::batch_size]
            if int(ptr[-1]) != data.num_nodes:
                ptr = torch.cat([ptr, torch.tensor([data.num_nodes])], dim=0)
            batch_size = 1
        self.data = data
        self.ptr = ptr
        self.bipartite = bipartite
        self.log = log
        self.compensate = compensate
        self.rm1hop = rm1hop

        n_id = torch.arange(data.num_nodes)
        batches = n_id.split((ptr[1:] - ptr[:-1]).tolist())
        batches = [(i, batches[i]) for i in range(len(batches))]

        if batch_size > 1:
            super().__init__(batches, batch_size=batch_size,
                             collate_fn=self.compute_subgraph, **kwargs)

        else:  # If `batch_size=1`, we pre-process the subgraph generation:
            if log:
                t = time.perf_counter()
                print('Pre-processing subgraphs...', end=' ', flush=True)

            data_list = list(
                DataLoader(batches, collate_fn=self.compute_subgraph,
                           batch_size=batch_size, **kwargs))

            if log:
                print(f'Done! [{time.perf_counter() - t:.2f}s]')

            super().__init__(data_list, batch_size=batch_size,
                             collate_fn=lambda x: x[0], **kwargs)

    def compute_subgraph(self, batches: List[Tuple[int, Tensor]]) -> SubData:
        batch_ids, n_ids = zip(*batches)
        
        n_id = torch.cat(n_ids, dim=0)
        batch_id = torch.tensor(batch_ids)

        # We collect the in-mini-batch size (`batch_size`), the offset of each
        # partition in the mini-batch (`offset`), and the number of nodes in
        # each partition (`count`)
        batch_size = n_id.numel()
        offset = self.ptr[batch_id]
        count = self.ptr[batch_id.add_(1)].sub_(offset)
        
                
        rowptr, col, value = self.data.adj_t.csr()
        if not self.rm1hop:
            rowptr, col, value, n_id, lmc_params = self.gas_sampler(rowptr, col, value, n_id, self.bipartite,)
        
            adj_t = SparseTensor(rowptr=rowptr, col=col, value=value,
                                sparse_sizes=(rowptr.numel() - 1, n_id.numel()),
                                is_sorted=True)
        else:
            rowptr, col, value, n_id, lmc_params = self.cluster_sampler(rowptr, col, value, n_id, self.bipartite,)
            adj_t = SparseTensor(rowptr=rowptr, col=col, value=None,
                                sparse_sizes=(rowptr.numel() - 1, n_id.numel()),
                                is_sorted=True)
            adj_t = gcn_norm(adj_t, add_self_loops=False)
        lmc_params['num_cluster'] = (len(self.ptr)-1) / len(batch_ids)

        data = self.data.__class__(adj_t=adj_t)
        for k, v in self.data:
            if isinstance(v, Tensor) and v.size(0) == self.data.num_nodes:
                data[k] = v.index_select(0, n_id)

        return SubData(data, batch_size, n_id, offset, count, lmc_params)

    def __repr__(self):
        return f'{self.__class__.__name__}()'
    
    def cluster_sampler(self, rowptr, col, value, n_id, bipartite):
        batch_size = n_id.numel()
        rowptr, col, value, n_id = relabel_fn(rowptr, col, value, n_id, bipartite)
        row = torch.repeat_interleave(torch.arange(batch_size), rowptr[1:]-rowptr[:-1],)
        row = row[col<batch_size]
        if value is not None:
            value = value[col<batch_size]
        col = col[col<batch_size]
        rowptr = torch.ops.torch_sparse.ind2ptr(row, batch_size)
        n_id = n_id[:batch_size]
        return (rowptr, col, value, n_id,) + ({'pull_mask_id': torch.arange((n_id.numel()-batch_size), dtype=torch.long)+batch_size, 'pull_nid': n_id[batch_size:]},)

    def gas_sampler(self, rowptr, col, value, n_id, bipartite):
        batch_size = n_id.numel()
        result = relabel_fn(rowptr, col, value, n_id, bipartite)
        return result + ({'pull_mask_id': torch.arange((result[3].numel()-batch_size), dtype=torch.long)+batch_size,'pull_nid': result[3][batch_size:]},)

class EvalSubgraphLoader(SubgraphLoader):
    r"""A simple subgraph loader that, given a pre-partioned :obj:`data` object,
    generates subgraphs from mini-batches in :obj:`ptr` (including their 1-hop
    neighbors).
    In contrast to :class:`SubgraphLoader`, this loader does not generate
    subgraphs from randomly sampled mini-batches, and should therefore only be
    used for evaluation.
    """
    def __init__(self, data: Data, ptr: Tensor, compensate: bool = False, batch_size: int = 1, rm1hop: bool = False, merge_cluster: bool = False, score_func_name: str = None,
                 bipartite: bool = True, log: bool = True, **kwargs):

        ptr = ptr[::batch_size]
        if int(ptr[-1]) != data.num_nodes:
            ptr = torch.cat([ptr, torch.tensor([data.num_nodes])], dim=0)

        super().__init__(data=data, ptr=ptr, compensate=compensate, batch_size=1, rm1hop=False, merge_cluster=merge_cluster, score_func_name=score_func_name, bipartite=bipartite,
                         log=log, shuffle=False, num_workers=0, **kwargs)
