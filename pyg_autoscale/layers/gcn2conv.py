from typing import Optional, Tuple
from torch_geometric.typing import Adj, OptTensor

from math import log

import torch
from torch import Tensor
from torch_sparse import SparseTensor
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import GCN2Conv

class MaskGCN2Conv(GCN2Conv):
    def forward(self, x: Tensor, x_0: Tensor, edge_index: Adj, batch_size = None,
                edge_weight: OptTensor = None) -> Tensor:
        """"""

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(  # yapf: disable
                        edge_index, edge_weight, x.size(self.node_dim), False,
                        self.add_self_loops, dtype=x.dtype)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        # propagate_type: (x: Tensor, edge_weight: OptTensor)
        x = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)

        x.mul_(1 - self.alpha)
        if batch_size is None:
            x_0 = self.alpha * x_0[:x.size(0)]

            if self.weight2 is None:
                out = x.add_(x_0)
                out = torch.addmm(out, out, self.weight1, beta=1. - self.beta,
                                alpha=self.beta)
            else:
                out = torch.addmm(x, x, self.weight1, beta=1. - self.beta,
                                alpha=self.beta)
                out += torch.addmm(x_0, x_0, self.weight2, beta=1. - self.beta,
                                alpha=self.beta)
        else:
            x_0 = self.alpha * x_0[:x.size(0)]
            x_0[batch_size:] = x_0[batch_size:].detach()
            if self.weight2 is None:
                out = x.add_(x_0)
                out1 = torch.addmm(out[:batch_size], out[:batch_size], self.weight1, beta=1. - self.beta,
                                alpha=self.beta)
                out2 = torch.addmm(out[batch_size:], out[batch_size:], self.weight1.detach(), beta=1. - self.beta,
                                alpha=self.beta)
                out = torch.cat([out1,out2], dim=0)
            else:
                out1 = torch.addmm(x[:batch_size], x[:batch_size], self.weight1, beta=1. - self.beta,
                                alpha=self.beta)
                out2 = torch.addmm(x[batch_size:], x[batch_size:], self.weight1.detach(), beta=1. - self.beta,
                                alpha=self.beta)
                out1 += torch.addmm(x_0[:batch_size], x_0[:batch_size], self.weight2, beta=1. - self.beta,
                                alpha=self.beta)
                out2 += torch.addmm(x_0[batch_size:], x_0[batch_size:], self.weight2.detach(), beta=1. - self.beta,
                                alpha=self.beta)
                out = torch.cat([out1,out2], dim=0)

        return out