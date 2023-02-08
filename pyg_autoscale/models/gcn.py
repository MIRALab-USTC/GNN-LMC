from typing import Optional

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Linear, BatchNorm1d, LayerNorm
from torch_sparse import SparseTensor
from torch_geometric.nn import GCNConv

from pyg_autoscale.models import ScalableGNN
from pyg_autoscale.layers import MaskGCNConv, DoubleBN, SharedBN, BNWapper

class MaskLinear(Linear):
    def forward(self, input: Tensor, batch_size = None) -> Tensor:
        if batch_size is None:
            return F.linear(input, self.weight, self.bias)
        else:
            out1 = F.linear(input[:batch_size], self.weight, self.bias)
            out2 = F.linear(input[batch_size:], self.weight.detach(), self.bias.detach())
            return torch.cat([out1,out2], dim=0)

class GCN(ScalableGNN):
    def __init__(self, num_nodes: int, in_channels: int, hidden_channels: int,
                 out_channels: int, num_layers: int, dropout: float = 0.0, bn_name: str = 'BatchNorm1d',
                 compensate: bool = True,
                 savegrad: bool=True, prehist: bool=False, random_dp: bool=False, beta: float=1,
                 drop_input: bool = True, batch_norm: bool = False,
                 residual: bool = False, linear: bool = False,
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
        self.linear = linear

        if self.in_channels == 1:
            self.emb_layer = torch.nn.Embedding(num_nodes, embedding_dim=hidden_channels)
            in_channels = hidden_channels

        self.lins = ModuleList()
        if linear:
            self.lins.append(Linear(in_channels, hidden_channels))
            self.lins.append(MaskLinear(hidden_channels, out_channels))

        self.convs = ModuleList()
        for i in range(num_layers):
            in_dim = out_dim = hidden_channels
            if i == 0 and not linear:
                in_dim = in_channels
            if i == num_layers - 1 and not linear:
                out_dim = out_channels
            conv = MaskGCNConv(in_dim, out_dim, normalize=False)
            self.convs.append(conv)
        
        if linear:
            self.add_last_hist(hidden_channels)
        else:
            self.add_last_hist(out_channels)

        self.bns = ModuleList()
        for i in range(num_layers):
            bn = eval(bn_name)(hidden_channels)
            if bn_name not in ['DoubleBN', 'SharedBN']:
                bn = BNWapper(bn)
            self.bns.append(bn)

    @property
    def reg_modules(self):
        if self.linear:
            return ModuleList(list(self.convs) + list(self.bns))
        else:
            return ModuleList(list(self.convs[:-1]) + list(self.bns))

    @property
    def nonreg_modules(self):
        return self.lins if self.linear else self.convs[-1:]

    def reset_parameters(self):
        super().reset_parameters()
        if self.in_channels == 1:
            self.emb_layer.reset_parameters()
        for lin in self.lins:
            lin.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def hist_conv(self, emb: Tensor, hist: Tensor, lmc_params):
        return emb + (1-self.alpha_beta)*(hist - emb.detach())

    def forward(self, x: Tensor, adj_t: SparseTensor, *args) -> Tensor:
        if self.in_channels == 1:
            x = self.emb_layer(x)

        if self.drop_input:
            x = F.dropout(x, p=self.dropout, training=self.training)

        if self.linear:
            x = self.lins[0](x).relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)

        if len(args)>0 and args[0] is not None:
            batch_size = args[0]
            if self.compensate:
                self.alpha_beta = self.beta*args[4]['alpha'][:,None]
        else:
            batch_size = None
        for conv, bn, emb_hist, grad_hist in zip(self.convs[:-1], self.bns, self.emb_histories, self.grad_histories):
            h = conv(x, adj_t, batch_size)
            if self.prehist:
                h = self.push_and_pull(emb_hist, grad_hist, h, *args)
                if self.batch_norm:
                    h = bn(h, batch_size)
                if self.residual and h.size(-1) == x.size(-1):
                    h += x[:h.size(0)]
            else:
                if self.batch_norm:
                    h = bn(h, batch_size)
                if self.residual and h.size(-1) == x.size(-1):
                    h += x[:h.size(0)]
                h = self.push_and_pull(emb_hist, grad_hist, h, *args)

            x = h.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)
        h = self.convs[-1](x, adj_t, batch_size)
        
        if self.compensate:
            h = self.push_and_pull_final(h, *args)

        if not self.linear:
            return h

        if self.batch_norm:
            h = self.bns[-1](h, batch_size)
        if self.residual and h.size(-1) == x.size(-1):
            h += x[:h.size(0)]
        h = h.relu_()
        h = F.dropout(h, p=self.dropout, training=self.training)
        return self.lins[1](h, batch_size)

    @torch.no_grad()
    def forward_layer(self, layer, x, adj_t, state, batch_size):
        if layer == 0:
            if self.in_channels == 1:
                x = self.emb_layer(x)
            if self.drop_input:
                x = F.dropout(x, p=self.dropout, training=self.training)
            if self.linear:
                x = self.lins[0](x).relu_()
                x = F.dropout(x, p=self.dropout, training=self.training)
        else:
            if self.prehist:
                if self.batch_norm:
                    x = self.bns[layer-1](x, batch_size)
                if self.residual:
                    raise NotImplementedError
            x = x.relu_()
            x = F.dropout(x, p=self.dropout, training=self.training)

        h = self.convs[layer](x, adj_t,)
        conv_h = h

        if layer < self.num_layers - 1 and (not self.prehist):
            if self.batch_norm:
                h = self.bns[layer](h, batch_size)
            if self.residual and h.size(-1) == x.size(-1):
                h += x[:h.size(0)]

        if layer == self.num_layers - 1 and self.linear:
            if self.batch_norm:
                h = self.bns[-1](h, batch_size)
            h = h.relu_()
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = self.lins[1](h)

        if layer == self.num_layers - 1:
            return h, conv_h
        else:
            return h