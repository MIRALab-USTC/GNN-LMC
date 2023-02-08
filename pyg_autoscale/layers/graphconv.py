
import torch
from torch import nn, Tensor

from torch_sparse import SparseTensor
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import GCNConv

class MaskGCNConv(GCNConv):
    def forward(self, x: Tensor, edge_index: Adj, batch_size = None,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        out = self.propagate(edge_index, x=x, edge_weight=edge_weight,
                             size=None)

        if batch_size is None:
            out = out @ self.weight
        else:
            out1 = out[:batch_size] @ self.weight
            out2 = out[batch_size:] @ self.weight.detach()
            out = torch.cat([out1,out2], dim=0)

        if self.bias is not None:
            out += self.bias

        return out
