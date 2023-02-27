import random
import warnings
import pdb
import numpy as np
import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, Fp16OptimizerHook,
                         OptimizerHook, build_optimizer, build_runner)
from mmdet.utils.Epoch_Based_Runner_Lambda import MyEpochBasedRunnerLambda
from mmdet.utils.Epoch_Based_Runner_SSD import MyEpochBasedRunnerSSD
from mmdet.utils.Epoch_Based_Runner_SSD_L import MyEpochBasedRunnerLSSD
from operator import itemgetter
from mmdet.core import DistEvalHook, EvalHook
from mmdet.datasets import (build_dataloader, build_dataset, replace_ImageToTensor)
from mmdet.utils import get_root_logger

def set_random_seed(seed, deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def train_detector_SSL(model, dataset, cfg, distributed=False, validate=False, timestamp=None, meta=None):
    if type(cfg) != list: # Supervised Setting
        logger = get_root_logger(cfg.log_level)
        dataset_L = dataset if isinstance(dataset, (list, tuple)) else [dataset]
        data_loaders_L = [build_dataloader(ds, cfg.data.samples_per_gpu, cfg.data.workers_per_gpu, len(cfg.gpu_ids),
                                         dist=distributed, seed=cfg.seed) for ds in dataset_L]
        data_loaders_U = None
    else: # Semi-Supervised Setting
        cfg = cfg[0]  # config used in this file are the same for cfg and cfg_u
        logger = get_root_logger(cfg.log_level)
        dataset_L, dataset_U = dataset[0], dataset[1]
        data_loaders_L = [build_dataloader(ds, cfg.data.samples_per_gpu, cfg.data.workers_per_gpu, len(cfg.gpu_ids),
                                           dist=distributed, seed=cfg.seed) for ds in dataset_L]
        data_loaders_U = [build_dataloader(ds, cfg.data.samples_per_gpu, cfg.data.workers_per_gpu, len(cfg.gpu_ids),
                                           dist=distributed, seed=cfg.seed) for ds in dataset_U]

    model = MMDataParallel(model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    optimizer = build_optimizer(model, cfg.optimizer)
    # RemoveParamFromOptim(optimizer, model.module, 'L_convs')
    runner = build_runner(cfg.runner, default_args=dict(model=model, optimizer=optimizer, work_dir=cfg.work_dir,
                                                        logger=logger, meta=meta))
    # runner.optimizer_L = torch.optim.SGD(list(model.module.bbox_head.L_convs.parameters()), lr = cfg.optimizer.lr,
    #                                      momentum=cfg.optimizer.momentum, weight_decay=cfg.optimizer.weight_decay)
    runner.timestamp = timestamp
    optimizer_config = cfg.optimizer_config
    # register hooks
    runner.register_training_hooks(cfg.lr_config, optimizer_config, cfg.checkpoint_config, cfg.log_config,
                                   cfg.get('momentum_config', None))
    # remove optimizer hook from runner.hooks
    for i, hook in enumerate(runner.hooks):
        if type(hook).__name__ == 'OptimizerHook':
            runner.hooks.pop(i)
            break
    # register eval hooks
    if validate:
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_dataloader(val_dataset, samples_per_gpu=val_samples_per_gpu,
                                          workers_per_gpu=cfg.data.workers_per_gpu, dist=distributed, shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = DistEvalHook if distributed else EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg))

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        runner.load_checkpoint(cfg.load_from)

    if data_loaders_U is None:
        eval_res = runner.run_SSL(data_loaders_L, cfg.workflow, cfg.total_epochs, onlyEval=cfg.onlyEval)
    else:
        eval_res = runner.run_SSL([data_loaders_L, data_loaders_U], cfg.workflow, cfg.total_epochs, onlyEval=cfg.onlyEval)

    return eval_res

def RemoveParamFromOptim(optimizer, model, param_name):
    targetIDs = []
    for name, param in model.named_parameters():
        if param_name in name:
            targetIDs.append(id(param))
            print(f'param {name} at {id(param)} is detected.')
    allIdx = set(range(len(optimizer.param_groups[0]['params'])))
    for idx, param in enumerate(optimizer.param_groups[0]['params']):
        if id(param) in targetIDs:
            allIdx = allIdx - {idx}
            # optimizer.param_groups[0]['params'].pop(idx)
            print(f'param at {id(param)} is removed from optimizer.')
    optimizer.param_groups[0]['params'] = list(itemgetter(*list(allIdx))(optimizer.param_groups[0]['params']))

