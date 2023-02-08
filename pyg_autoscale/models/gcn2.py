from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d, LayerNorm
from torch_sparse import SparseTensor
from torch_geometric.nn import GCN2Conv

from pyg_autoscale.models import ScalableGNN
from pyg_autoscale.models.gcn import MaskLinear
from pyg_autoscale.layers import MaskGCN2Conv, DoubleBN, SharedBN, BNWapper

class GCN2(ScalableGNN):
    def __init__(self, num_nodes: int, in_channels, hidden_channels: int,
                 out_channels: int, num_layers: int,
                 alpha: float, theta: float, shared_weights: bool = True, bn_name: str = 'BatchNorm1d',
                 compensate: bool = True,
                 savegrad: bool=True, prehist: bool=False, random_dp: bool=False, beta: float=1,
                 dropout: float = 0.0, drop_input: bool = True,
                 batch_norm: bool = False, residual: bool = False,
                 pool_size: Optional[int] = None,
                 buffer_size_train: Optional[int] = None, buffer_size_eval: Optional[int] = None, device=None):
        super().__init__(num_nodes, compensate, hidden_channels, num_layers, savegrad, prehist, random_dp, beta, pool_size,
                         buffer_size_train, buffer_size_eval, device)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.dropout = dropout
        self.drop_input = drop_input
        self.batch_norm = batch_norm
        self.residual = residual

        self.lins = ModuleList()
        if self.in_channels == 1:
            self.lins.append(torch.nn.Embedding(num_nodes, embedding_dim=hidden_channels))
        else:
            self.lins.append(Linear(in_channels, hidden_channels))
        self.lins.append(MaskLinear(hidden_channels, out_channels))

        self.convs = ModuleList()
        for i in range(num_layers):
            conv = MaskGCN2Conv(hidden_channels, alpha=alpha, theta=theta,
                            layer=i + 1, shared_weights=shared_weights,
                            normalize=False)
            self.convs.append(conv)

        self.add_last_hist(hidden_channels)

        self.bns = ModuleList()
        for i in range(num_layers):
            bn = eval(bn_name)(hidden_channels)
            if bn_name not in ['DoubleBN', 'SharedBN']:
                bn = BNWapper(bn)
            self.bns.append(bn)

    @property
    def reg_modules(self):
        return ModuleList(list(self.convs) + list(self.bns))

    @property
    def nonreg_modules(self):
        return self.lins

    def reset_parameters(self):
        super().reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def hist_conv(self, emb: Tensor, hist: Tensor, lmc_params):
        return emb + (1-self.alpha_beta)*(hist - emb.detach())

    def forward(self, x: Tensor, adj_t: SparseTensor, *args) -> Tensor:
        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = x_0 = self.lins[0](x).relu_()
        x = F.dropout(x, p=self.dropout, training=self.training)

        if len(args)>0:
            batch_size = args[0]
            if self.compensate and batch_size is not None:
                self.alpha_beta = self.beta*args[4]['alpha'][:,None]
            else:
                self.alpha_beta = 1
        else:
            batch_size = None
        for conv, bn, emb_hist, grad_hist in zip(self.convs[:-1], self.bns[:-1],
                                  self.emb_histories, self.grad_histories):
            h = conv(x, x_0, adj_t, batch_size)
            if self.prehist:
                h = self.push_and_pull(emb_hist, grad_hist, h, *args)
                if self.batch_norm:
                    h = bn(h, batch_size)
                if self.residual:
                    h += x[:h.size(0)]
            else:
                if self.batch_norm:
                    h = bn(h, batch_size)
                if self.residual:
                    h += x[:h.size(0)]
                h = self.push_and_pull(emb_hist, grad_hist, h, *args)
            
            x = h.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.convs[-1](x, x_0, adj_t, batch_size)
        if self.compensate:
            h = self.push_and_pull_final(h, *args)
        if self.batch_norm:
            h = self.bns[-1](h, batch_size)
        if self.residual:
            h += x[:h.size(0)]
        x = h.relu_()
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[1](x, batch_size)
        return x

    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state, batch_size):
        if layer == 0:
            if self.drop_input:
                x = F.dropout(x, p=self.dropout, training=self.training)

            x = x_0 = self.lins[0](x).relu_()
            state['x_0'] = x_0[:adj_t.size(0)]
        else:
            if self.prehist:
                if self.batch_norm:
                    x = self.bns[layer-1](x, batch_size)
                if self.residual:
                    raise NotImplementedError
            x = x.relu_()
        x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.convs[layer](x, state['x_0'], adj_t)
        conv_h = h

        if (not self.prehist) or (layer == (self.num_layers - 1)):
            if self.batch_norm:
                h = self.bns[layer](h)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]
            h = h.relu_()

        if layer == self.num_layers - 1:
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.lins[1](h)

        if layer == self.num_layers - 1:
            return h, conv_h
        else:
            return h