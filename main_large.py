import time
import hydra
from omegaconf import OmegaConf
import yaml
import traceback
import os
import json

import numpy as np
import torch
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from pyg_autoscale import (get_data, metis, random_subgraph, permute,
                                       LMCLoader, SubgraphLoader, EvalSubgraphLoader,
                                       models, compute_micro_f1, dropout)
from pyg_autoscale.data import get_ppi
import pyg_autoscale.logger as logger

torch.manual_seed(123)


def get_grad(model, loader, criterion, optimizer, epoch):
    for bn in model.bns:
        if hasattr(bn, 'bn_ib'):
            bn.bn_ib.track_running_stats = False
            bn.bn_ob.track_running_stats = False
        else:
            bn.bn.track_running_stats = False
    optimizer.zero_grad()
    emb_hist_list = []
    for i in range(len(model.emb_histories)):
        emb_hist_list.append(model.emb_histories[i].emb.clone())
        model.emb_histories[i].reset_parameters()

    grad_hist_list = []
    for i in range(len(model.grad_histories)):
        grad_hist_list.append(model.grad_histories[i].emb.clone())
        model.grad_histories[i].reset_parameters()

    adj_t_all = loader.data.adj_t.to(model.device)
    input_x = loader.data.x.to(model.device)
    logits = model(input_x, adj_t_all,)
    train_mask = loader.data.train_mask.to(model.device)
    label = loader.data.y.to(model.device)

    loss = criterion(logits[train_mask], label[train_mask])
    loss.backward()

    exact_grad = []
    for i in range(len(model.convs)):
        exact_grad.append(model.convs[i].weight.grad.detach().clone())
    optimizer.zero_grad()

    emb_hist_exact_list = []
    for i in range(len(model.emb_histories)):
        emb_hist_exact_list.append(model.emb_histories[i].emb.clone())
        model.emb_histories[i].emb.copy_(emb_hist_list[i])
    
    grad_hist_exact_list = []
    for i in range(len(model.grad_histories)):
        grad_hist_exact_list.append(model.grad_histories[i].emb.clone())
        model.grad_histories[i].emb.copy_(grad_hist_list[i])

    optimizer.zero_grad()
    for bn in model.bns:
        if hasattr(bn, 'bn_ib'):
            bn.bn_ib.track_running_stats = True
            bn.bn_ob.track_running_stats = True
        else:
            bn.bn.track_running_stats = True
    return exact_grad, emb_hist_exact_list, grad_hist_exact_list

def mini_train(model, loader, criterion, optimizer, max_steps, num_alltrain, update_grad, epoch, accumulation_steps=1, random_dp_prob=1.0, grad_norm=None,
               edge_dropout=0.0):
    model.train()

    DEBUG = False
    if DEBUG:
        model.dropout = 0
        for bn in model.bns:
            bn.eval()

    if model.compensate and update_grad:
        optimizer.zero_grad()
        data = loader.data
        logits = model(data.x.to(model.device), data.adj_t.to(model.device), None)
        train_mask = data.train_mask.to(model.device)
        y = data.y.to(model.device)
        loss = criterion(logits[train_mask], y[train_mask])
        loss.backward()
        optimizer.zero_grad()
    
    total_loss = total_examples = 0
    t = time.perf_counter()
    
    if not DEBUG:
        optimizer.zero_grad()
    for i, (batch, batch_size, *args) in enumerate(loader):
        logger.logkv_mean('Train/sample_time', time.perf_counter() - t)
        
        if model.savegrad:
            exact_grad, emb_hist_exact_list, grad_hist_exact_list = get_grad(model, loader, criterion, optimizer, epoch)

        t = time.perf_counter()
        
        x = batch.x.to(model.device)
        adj_t = batch.adj_t.to(model.device)
        y = batch.y[:adj_t.size(0)].to(model.device)
        train_mask = batch.train_mask[:adj_t.size(0)].to(model.device)
        if 'alpha' in args[3]:
            args[3]['alpha'] = args[3]['alpha'].to(model.device)

        num_train = train_mask.sum()
        if train_mask.sum() == 0:
            logger.log('num_minitrain = 0')
            if not DEBUG:
                if (i+1) % accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
            continue

        # We make use of edge dropout on ogbn-products to avoid overfitting.
        adj_t = dropout(adj_t, p=edge_dropout)

        # weight = (num_train / num_alltrain) * (model.num_nodes / batch_size)
        if model.compensate:
            weight = (num_train / num_alltrain * args[3]['num_cluster']) # * (model.num_nodes / batch_size)
        else:
            weight = 1
        out = model(x, adj_t, batch_size, *args)
        loss = criterion(out[train_mask], y[train_mask])*weight

        logger.logkv_mean('Train/forward_time', time.perf_counter() - t)
        t = time.perf_counter()

        loss.backward()

        if model.savegrad:
            for i in range(len(model.convs)):
                logger.logkv_mean(f'Error/weight_{i}',torch.norm(model.convs[i].weight.grad.detach().clone()-exact_grad[i]).item()/torch.norm(exact_grad[i]).item())

            for i in range(len(model.emb_histories)):
                logger.logkv_mean(f'Error/emb_hist_relative_{i}', torch.abs(emb_hist_exact_list[i][args[0][:batch_size]] - model.emb_histories[i].emb[args[0][:batch_size]]).mean().item() / torch.abs(emb_hist_exact_list[i][args[0][:batch_size]]).mean().item())
                logger.logkv_mean(f'Error/emb_hist_{i}', torch.abs(emb_hist_exact_list[i][args[0][:batch_size]] - model.emb_histories[i].emb[args[0][:batch_size]]).mean().item())

            for i in range(len(model.grad_histories)):
                logger.logkv_mean(f'Error/grad_bias_{i}', torch.abs(grad_hist_exact_list[i][args[0][:batch_size]] - model.grad_histories[i].emb[args[0][:batch_size]]).mean().item() / torch.abs(grad_hist_exact_list[i][args[0][:batch_size]]).mean().item())

        if grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_norm)
        if not DEBUG:
            if (i+1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

        
        loss_all = criterion(out.detach()[:batch_size][train_mask[:batch_size]], y[:batch_size][train_mask[:batch_size]]) # If the log of loss is not important, removing this step can significantly accelerate LMC
        total_loss += float(loss_all) * int(train_mask[:batch_size].sum())
        total_examples += int(train_mask[:batch_size].sum())

        model.synchronize_grad()

        logger.logkv_mean('Train/backward_time', time.perf_counter() - t)
        t = time.perf_counter()

        # We may abort after a fixed number of steps to refresh histories...
        if (i + 1) >= max_steps and (i + 1) < len(loader):
            break
    return total_loss / total_examples


@torch.no_grad()
def full_test(model, data):
    model.eval()
    return model(data.x.to(model.device), data.adj_t.to(model.device)).cpu()


@torch.no_grad()
def mini_test(model, loader):
    model.eval()
    return model(loader=loader)

def logresults(name, results,):
    result_len = len(results[results>0])
    logger.logkv(name+'/mean', 0 if result_len == 0 else torch.mean(results[results>0]).item())
    logger.logkv(name+'/max', 0 if result_len == 0 else torch.max(results[results>0]).item())
    results_std = torch.std(results[results>0]).item()
    logger.logkv(name+'/std', results_std if not np.isnan(results_std) else 0)

@hydra.main(config_path='conf_large', config_name='config')
def main(conf):
    conf.model.params = conf.model.params[conf.dataset.name]

    if 'json' in conf.model and conf.model.json is not None:
        if os.path.exists(conf.model.json):
            args_beta = conf.model.params.architecture.beta
            args_bn = conf.model.params.architecture.batch_norm
            args_dropout = conf.model.params.architecture.dropout
            args_savegrad = conf.model.params.savegrad
            args_runs = conf.model.params.runs
            args_rm1hop = conf.model.rm1hop
            args_update_grad = conf.model.params.update_grad
            args_max_steps = conf.model.params.max_steps
            args_batch_size = conf.model.params.batch_size

            print('Load hyperparameters')
            with open(os.path.join(conf.model.json), 'r') as f:
                dict_conf = json.load(f, )
            conf['model'] = dict_conf['model']

            conf.model.params.max_steps = args_max_steps
            conf.model.params.update_grad = args_update_grad
            conf.model.rm1hop = args_rm1hop
            conf.model.params.runs = args_runs
            conf.model.params.savegrad = args_savegrad
            conf.model.params.architecture.dropout = args_dropout
            conf.model.params.architecture.batch_norm = args_bn
            if conf.log_every == 2:
                conf.model.params.batch_size = args_batch_size
                conf.model.params.architecture.beta = args_beta
        else:
            print('Json path does not exist!')
            exit()

    params = conf.model.params
    logger.configure(dir='./log_dir/', format_strs=['stdout','log','csv','tensorboard'])
    dict_conf = yaml.safe_load(OmegaConf.to_yaml(conf))
    logger.save_conf(dict_conf)
    try:
        edge_dropout = params.edge_dropout
    except:  # noqa
        edge_dropout = 0.0
    grad_norm = None if isinstance(params.grad_norm, str) else params.grad_norm

    device = f'cuda' if torch.cuda.is_available() else 'cpu'

    t = time.perf_counter()
    logger.log('Loading data...',)
    data, in_channels, out_channels = get_data(conf.dataset.name)
    logger.log(f'Done! [{time.perf_counter() - t:.2f}s]')

    perm, ptr = eval(conf.model.partition)(data.adj_t, num_parts=params.num_parts, log=True)
    data = permute(data, perm, log=True)

    num_train = data.train_mask.int().sum()

    if conf.model.loop:
        t = time.perf_counter()
        logger.log('Adding self-loops...',)
        data.adj_t = data.adj_t.set_diag()
        logger.log(f'Done! [{time.perf_counter() - t:.2f}s]')
    if conf.model.norm:
        t = time.perf_counter()
        logger.log('Normalizing data...',)
        data.adj_t = gcn_norm(data.adj_t, add_self_loops=False)
        logger.log(f'Done! [{time.perf_counter() - t:.2f}s]')

    if data.y.dim() == 1:
        criterion = torch.nn.CrossEntropyLoss()
    else:
        criterion = torch.nn.BCEWithLogitsLoss()

    if params.architecture.compensate == False:
        assert (params.update_grad == False)

    if params.architecture.compensate:
        train_loader = LMCLoader(data, ptr, batch_size=params.batch_size,
                                  compensate=params.architecture.compensate,
                                  merge_cluster=params.merge_cluster,
                                  score_func_name=params.score_func_name,
                                  shuffle=True, num_workers=params.num_workers,
                                  persistent_workers=params.num_workers > 0)
    else:
        train_loader = SubgraphLoader(data, ptr, batch_size=params.batch_size,
                                  compensate=params.architecture.compensate,
                                  merge_cluster=params.merge_cluster,
                                  score_func_name=params.score_func_name,
                                  rm1hop=conf.model.rm1hop,
                                  shuffle=True, num_workers=params.num_workers,
                                  persistent_workers=params.num_workers > 0)
    batch_size = params['batch_size'] if conf.dataset.name in ['products'] else int(0.25*params.num_parts)
    eval_loader = EvalSubgraphLoader(data, ptr, rm1hop=conf.model.rm1hop,
                                     batch_size=batch_size)

    if conf.dataset.name == 'ppi':
        val_data, _, _ = get_ppi(split='val')
        test_data, _, _ = get_ppi(split='test')
        if conf.model.loop:
            val_data.adj_t = val_data.adj_t.set_diag()
            test_data.adj_t = test_data.adj_t.set_diag() 
        if conf.model.norm:
            val_data.adj_t = gcn_norm(val_data.adj_t, add_self_loops=False)
            test_data.adj_t = gcn_norm(test_data.adj_t, add_self_loops=False)

    t = time.perf_counter()
    logger.log('Calculating buffer size...',)
    # We reserve a much larger buffer size than what is actually needed for
    # training in order to perform efficient history accesses during inference.
    buffer_size_eval = max([n_id.numel() for _, _, n_id, _, _, _ in eval_loader])
    if params.merge_cluster or params['batch_size'] == 1:
        buffer_size_train = max([lmc_params['pull_mask_id'].numel() for _, _, _, _, _, lmc_params in train_loader])
    else:
        buffer_size_train = buffer_size_eval
    logger.log(f'Done! [{time.perf_counter() - t:.2f}s] -> {buffer_size_train},{buffer_size_eval}')

    kwargs = {}
    if conf.model.name[:3] == 'PNA':
        kwargs['deg'] = data.adj_t.storage.rowcount()

    GNN = getattr(models, conf.model.name)
    model = GNN(
        num_nodes=data.num_nodes,
        in_channels=in_channels,
        out_channels=out_channels,
        pool_size=params.pool_size,
        buffer_size_train=buffer_size_train,
        buffer_size_eval=buffer_size_eval,
        savegrad = params.savegrad,
        prehist = params.prehist,
        random_dp = params.random_dp,
        **params.architecture,
        **kwargs,
    ).to(device)

    results = torch.zeros(params.runs)
    try:
        for run in range(params.runs):
            optimizer = eval('torch.optim.'+params.optimizer_name)([
                dict(params=model.reg_modules.parameters(),
                    weight_decay=params.reg_weight_decay),
                dict(params=model.nonreg_modules.parameters(),
                    weight_decay=params.nonreg_weight_decay)
            ], lr=params.lr)
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max',
                                factor=params['lr_reduce_factor'],
                                patience=params['lr_schedule_patience'],
                                verbose=True)

            t = time.perf_counter()
            logger.log('Fill history...',)
            mini_test(model, eval_loader)
            logger.log(f'Done! [{time.perf_counter() - t:.2f}s]')

            t_total = time.perf_counter()
            best_val_acc = test_acc = 0

            for epoch in range(1, int(params.epochs*conf.log_every) + 1):
                t_temp = time.time()
                loss = mini_train(model, train_loader, criterion, optimizer,
                                params.max_steps, num_train, params.update_grad, epoch, params.accumulation_steps, params.random_dp_prob, grad_norm, edge_dropout)
                logger.logkv('Train/epoch_time', time.time() - t_temp)
                
                out = mini_test(model, eval_loader)
                train_acc = compute_micro_f1(out, data.y, data.train_mask)

                if conf.dataset.name != 'ppi':
                    val_acc = compute_micro_f1(out, data.y, data.val_mask)
                    tmp_test_acc = compute_micro_f1(out, data.y, data.test_mask)
                else:
                    # We need to perform inference on a different graph as PPI is an
                    # inductive dataset.
                    val_acc = compute_micro_f1(full_test(model, val_data), val_data.y)
                    tmp_test_acc = compute_micro_f1(full_test(model, test_data),
                                                    test_data.y)

                scheduler.step(val_acc)

                if val_acc > best_val_acc:
                    best_val_acc = val_acc
                    test_acc = tmp_test_acc
                    results[run] = test_acc
                if epoch % conf.log_every == 0:
                    logger.logkv('Epoch', int(epoch // conf.log_every))
                    logger.logkv('Train/loss_train', loss)
                    logger.logkv('Train/f1_train', train_acc)
                    logger.logkv('Val/f1_val', val_acc)
                    logger.logkv('Test/f1_test', tmp_test_acc)
                    logger.logkv('Test/final_f1_test', test_acc)
                    logger.logkv('Total Time', time.perf_counter() - t_total)
                    logger.logkv('lr', optimizer.param_groups[0]['lr'])
                    logresults( 'results', results,)
                    logger.dumpkvs()
            
            model.reset_parameters()
        logger.log(torch.mean(results), torch.std(results))
    except Exception as e:
            logger.log(traceback.format_exc())
            # logger.log(e)



if __name__ == "__main__":
    main()
