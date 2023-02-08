from typing import Optional, Callable, Dict, Any

import warnings

import time
import torch
from torch import Tensor
from torch_sparse import SparseTensor

from pyg_autoscale import History, AsyncIOPool
from pyg_autoscale import SubgraphLoader, EvalSubgraphLoader


class ScalableGNN(torch.nn.Module):
    def __init__(self, num_nodes: int, compensate: bool, hidden_channels: int, num_layers: int,
                 savegrad: bool, prehist: bool, random_dp: bool,beta :float,
                 pool_size: Optional[int] = None,
                 buffer_size_train: Optional[int] = None, buffer_size_eval: Optional[int] = None, device=None):
        super().__init__()

        self.num_nodes = num_nodes
        self.hidden_channels = hidden_channels
        self.num_layers = num_layers
        self.pool_size = num_layers - 1 if pool_size is None else pool_size
        self.buffer_size_train = buffer_size_train
        self.buffer_size_eval = buffer_size_eval
        self.compensate = compensate
        self.savegrad = savegrad
        self.prehist = prehist
        self.beta = beta
        self.random_dp = random_dp
        self.update_hist_grad = True

        self.emb_histories = torch.nn.ModuleList([
            History(num_nodes, hidden_channels, device)
            for _ in range(num_layers-1)
        ])
        self.grad_histories = torch.nn.ModuleList([
            History(num_nodes, hidden_channels, device)
            for _ in range(num_layers-1)
        ])

        self.pool: Optional[AsyncIOPool] = None
        self._async = False
        self._async_grad = False
        self.__out: Optional[Tensor] = None

        self.alpha_beta = 1

    def add_last_hist(self, out_channels):
        self.emb_histories_final = History(self.num_nodes, out_channels, self.emb_device)

    @property
    def emb_device(self):
        return self.emb_histories[0].emb.device

    @property
    def device(self):
        return self.emb_histories[0]._device

    def _apply(self, fn: Callable) -> None:
        super()._apply(fn)
        # We only initialize the AsyncIOPool in case histories are on CPU:
        if (str(self.emb_device) == 'cpu' and str(self.device)[:4] == 'cuda'
                and self.pool_size is not None
                and self.buffer_size_train is not None and self.buffer_size_eval is not None):
            self.pool = AsyncIOPool(self.pool_size, self.buffer_size_train, self.buffer_size_eval,
                                    self.emb_histories[0].embedding_dim)
            self.pool.to(self.device)
        return self

    def reset_parameters(self):
        for history in self.emb_histories:
            history.reset_parameters()
        for history in self.grad_histories:
            history.reset_parameters()
        self.emb_histories_final.reset_parameters()

    def __call__(
        self,
        x: Optional[Tensor] = None,
        adj_t: Optional[SparseTensor] = None,
        batch_size: Optional[int] = None,
        n_id: Optional[Tensor] = None,
        offset: Optional[Tensor] = None,
        count: Optional[Tensor] = None,
        lmc_params: Dict = {},
        loader: EvalSubgraphLoader = None,
        **kwargs,
    ) -> Tensor:
        r"""Enhances the call of forward propagation by immediately start
        pulling historical embeddings for all layers asynchronously.
        After forward propogation is completed, the push of node embeddings to
        the histories will be synchronized.

        For example, given a mini-batch with node indices
        :obj:`n_id = [0, 1, 5, 6, 7, 3, 4]`, where the first 5 nodes
        represent the mini-batched nodes, and nodes :obj:`3` and :obj:`4`
        denote out-of-mini-batched nodes (i.e. the 1-hop neighbors of the
        mini-batch that are not included in the current mini-batch), then
        other input arguments should be given as:

        .. code-block:: python

            batch_size = 5
            offset = [0, 2, 5]
            count = [2, 3]
        """

        if loader is not None:
            return self.mini_inference(loader)

        # We only perform asynchronous history transfer in case the following
        # conditions are met:
        self._async = (self.pool is not None and batch_size is not None
                       and n_id is not None and offset is not None
                       and count is not None)
        self._async_grad = self._async

        if (batch_size is not None and not self._async
                and str(self.emb_device) == 'cpu'
                and str(self.device)[:4] == 'cuda'):
            warnings.warn('Asynchronous I/O disabled, although history and '
                          'model sit on different devices.')

        self.synpull = False

        if self._async and not self.synpull:
            for emb_hist in self.emb_histories:
                self.pool.async_pull(emb_hist.emb, None, None, lmc_params['pull_nid'])
            if self.compensate:
                for grad_hist in self.grad_histories[::-1]:
                    self.pool.async_pull(grad_hist.emb, None, None, lmc_params['pull_nid'])

        out = self.forward(x, adj_t, batch_size, n_id, offset, count, lmc_params, **kwargs)

        self._async = False

        return out

    def hist_conv(self, emb: Tensor, hist: Tensor,):
        raise NotImplementedError

    def push_and_pull_grad(self, grad_hist: History, x: Tensor,
                      batch_size: Optional[int] = None,
                      n_id: Optional[Tensor] = None,
                      offset: Optional[Tensor] = None,
                      count: Optional[Tensor] = None,
                      lmc_params: Dict = {},
                    ):
        if batch_size is None:
            if self.compensate or self.savegrad:
                def backward_hook(grad):
                    grad_hist.push(grad)
                    return grad
                x.register_hook(backward_hook)
            return x

        if self.training and self.compensate:
            num_cluster = lmc_params['num_cluster']

            def backward_hook(grad):
                grad = grad/num_cluster
                
                if self._async_grad and not self.synpull:
                    compensate_grad = self.pool.synchronize_pull()[:lmc_params['pull_mask_id'].numel()]
                else:
                    compensate_grad = grad_hist.pull(lmc_params['pull_nid'])
                grad[lmc_params['pull_mask_id']] = (1-self.alpha_beta)*compensate_grad + self.alpha_beta*grad[lmc_params['pull_mask_id']]

                if self._async_grad:
                    self.pool.async_push(grad[:batch_size], offset, count, grad_hist.emb)
                else:
                    grad_hist.push(grad[:batch_size], n_id[:batch_size], offset, count)
                
                if self.random_dp:
                    grad[batch_size:] = 0

                if self._async_grad and not self.synpull:
                    self.pool.free_pull()

                return grad*(num_cluster) 

            x.register_hook(backward_hook)
        elif self.training and self.savegrad:
            if self.grad_histories[-1].emb is grad_hist.emb:
                num_cluster = lmc_params['num_cluster']
            else:
                num_cluster = self.num_nodes/batch_size
            def backward_hook(grad):
                assert grad.shape[0] == batch_size

                if self.update_hist_grad:
                    if self._async_grad:
                        self.pool.async_push(grad[:batch_size]/num_cluster, offset, count, grad_hist.emb)
                    else:
                        grad_hist.push(grad[:batch_size]/num_cluster, n_id[:batch_size], offset, count)

                return grad
            x.register_hook(backward_hook)
        return x

    def push_and_pull(self, history, history_grad, x: Tensor,
                      batch_size: Optional[int] = None,
                      n_id: Optional[Tensor] = None,
                      offset: Optional[Tensor] = None,
                      count: Optional[Tensor] = None,
                      lmc_params: Dict = {},) -> Tensor:
        r"""Pushes and pulls information from :obj:`x` to :obj:`history` and
        vice versa."""
        if n_id is None and x.size(0) != self.num_nodes:
            return x  # Do nothing...

        if n_id is None and x.size(0) == self.num_nodes:
            history.push(x)
            x = self.push_and_pull_grad(history_grad, x, batch_size, n_id, offset, count)
            return x

        assert n_id is not None

        if batch_size is None:
            history.push(x, n_id)
            return x

        if not self._async:
            h = history.pull(lmc_params['pull_nid'])
            history.push(x[:batch_size], n_id[:batch_size],)

            if self.compensate:
                x[lmc_params['pull_mask_id']] = self.hist_conv(x[lmc_params['pull_mask_id']], h, lmc_params)
                x = self.push_and_pull_grad(history_grad, x, batch_size, n_id, offset, count, lmc_params)
                return x
            else:
                x = self.push_and_pull_grad(history_grad, x, batch_size, n_id, offset, count, lmc_params)

                return torch.cat([x[:batch_size], h], dim=0)

        else:
            if not self.synpull:
                out = self.pool.synchronize_pull()
                out = out[:lmc_params['pull_mask_id'].numel()]
            else:
                out = history.pull(lmc_params['pull_nid'])
                
            self.pool.async_push(x[:batch_size].clone(), offset, count, history.emb)

            if self.compensate:
                x[lmc_params['pull_mask_id']] = self.hist_conv(x[lmc_params['pull_mask_id']], out, lmc_params)

                if not self.synpull:
                    self.pool.free_pull()
                x = self.push_and_pull_grad(history_grad, x, batch_size, n_id, offset, count, lmc_params)
                return x
            else:
                x = self.push_and_pull_grad(history_grad, x, batch_size, n_id, offset, count, lmc_params)
                out = torch.cat([x[:batch_size], out], dim=0)
                if not self.synpull:
                    self.pool.free_pull()
                return out

    def push_and_pull_final(self, x: Tensor,
                      batch_size: Optional[int] = None,
                      n_id: Optional[Tensor] = None,
                      offset: Optional[Tensor] = None,
                      count: Optional[Tensor] = None,
                      lmc_params: Dict = {},) -> Tensor:
        r"""Pushes and pulls information from :obj:`x` to :obj:`history` and
        vice versa."""
        if n_id is None and x.size(0) != self.num_nodes:
            return x  # Do nothing...

        if n_id is None and x.size(0) == self.num_nodes:
            self.emb_histories_final.push(x)
            return x

        assert n_id is not None

        if batch_size is None:
            self.emb_histories_final.push(x, n_id)
            return x
        
        h = self.emb_histories_final.pull(lmc_params['pull_nid'])
        if not self._async:
            self.emb_histories_final.push(x[:batch_size], n_id[:batch_size],)
        else:
            self.pool.async_push(x[:batch_size], offset, count, self.emb_histories_final.emb)

        x[lmc_params['pull_mask_id']] = self.hist_conv(x[lmc_params['pull_mask_id']], h, lmc_params)

        if self.random_dp:
            def backward_hook(grad):
                grad[batch_size:] =0
                return grad
            x.register_hook(backward_hook)
        return x

    @property
    def _out(self):
        if self.__out is None:
            self.__out = torch.empty(self.num_nodes, self.out_channels,
                                     pin_memory=True)
        return self.__out

    @torch.no_grad()
    def mini_inference_serial(self, loader: SubgraphLoader) -> Tensor:
        # We push the outputs of the first layer to the history:
        for data, batch_size, n_id, offset, count, lmc_params, state in loader:
            x = data.x.to(self.device)
            adj_t = data.adj_t.to(self.device)
            out = self.forward_layer(0, x, adj_t, state, batch_size)[:batch_size]
            self.emb_histories[0].push(out, n_id[:batch_size], offset, count)

        for i in range(1, len(self.emb_histories)):
            # Compute new output embeddings one-by-one and start pushing them
            # to the history.
            for batch, batch_size, n_id, offset, count, lmc_params, state in loader:
                adj_t = batch.adj_t.to(self.device)
                x = self.emb_histories[i - 1].pull(n_id)
                out = self.forward_layer(i, x, adj_t, state, batch_size)[:batch_size]
                self.emb_histories[i].push(out, n_id[:batch_size], offset, count)

        # And compute final output embeddings, which we write into a private
        # output embedding matrix:
        out_all = torch.zeros(self.num_nodes, self.out_channels, device=self.device)
        for batch, batch_size, n_id, offset, count, lmc_params, state in loader:
            adj_t = batch.adj_t.to(self.device)
            x = self.emb_histories[-1].pull(n_id)
            out, conv_h = self.forward_layer(self.num_layers - 1, x, adj_t,
                                     state, batch_size)[:batch_size]
            self.emb_histories_final.push(conv_h, n_id[:batch_size], offset, count)
            
            out_all[n_id[:batch_size]] = out

        return out_all.cpu()

    @torch.no_grad()
    def mini_inference(self, loader: SubgraphLoader) -> Tensor:
        r"""An implementation of layer-wise evaluation of GNNs.
        For each individual layer and mini-batch, :meth:`forward_layer` takes
        care of computing the next state of node embeddings.
        Additional state (such as residual connections) can be stored in
        a `state` directory."""

        # We iterate over the loader in a layer-wise fashsion.
        # In order to re-use some intermediate representations, we maintain a
        # `state` dictionary for each individual mini-batch.

        loader = [sub_data + ({}, ) for sub_data in loader]

        if self.pool is None:
            return self.mini_inference_serial(loader)

        # We push the outputs of the first layer to the history:
        for data, batch_size, n_id, offset, count, lmc_params, state in loader:
            x = data.x.to(self.device)
            adj_t = data.adj_t.to(self.device)
            out = self.forward_layer(0, x, adj_t, state, batch_size)[:batch_size]
            self.pool.async_push(out, offset, count, self.emb_histories[0].emb)
        self.pool.synchronize_push()

        for i in range(1, len(self.emb_histories)):
            # Pull the complete layer-wise history:
            for _, batch_size, n_id, offset, count, lmc_params, _ in loader:
                self.pool.async_pull(self.emb_histories[i - 1].emb, offset, count,
                                     n_id[batch_size:])

            # Compute new output embeddings one-by-one and start pushing them
            # to the history.
            for batch, batch_size, n_id, offset, count, lmc_params, state in loader:
                adj_t = batch.adj_t.to(self.device)
                x = self.pool.synchronize_pull()[:n_id.numel()]
                out = self.forward_layer(i, x, adj_t, state, batch_size)[:batch_size]
                self.pool.async_push(out, offset, count, self.emb_histories[i].emb)
                self.pool.free_pull()
            self.pool.synchronize_push()

        # We pull the histories from the last layer:
        for _, batch_size, n_id, offset, count, lmc_params, _ in loader:
            self.pool.async_pull(self.emb_histories[-1].emb, offset, count,
                                 n_id[batch_size:])

        # And compute final output embeddings, which we write into a private
        # output embedding matrix:
        for batch, batch_size, n_id, offset, count, lmc_params, state in loader:
            adj_t = batch.adj_t.to(self.device)
            x = self.pool.synchronize_pull()[:n_id.numel()]
            out, conv_h = self.forward_layer(self.num_layers - 1, x, adj_t,
                                     state, batch_size)[:batch_size]
            if self.compensate:
                self.pool.async_push(conv_h, offset, count, self.emb_histories_final.emb)
            self.pool.async_push(out, offset, count, self._out)
            self.pool.free_pull()
        self.pool.synchronize_push()

        return self._out

    @torch.no_grad()
    def forward_layer(self, layer: int, x: Tensor, adj_t: SparseTensor,
                      state: Dict[str, Any], batch_size) -> Tensor:
        raise NotImplementedError

    def synchronize_grad(self,):
        self._async_grad = False