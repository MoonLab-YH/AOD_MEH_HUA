import argparse
import copy
import os
import os.path as osp
import time
import warnings
import pdb
import random
import torch
import math
import sys
import psutil
sys.path.append(os.getcwd())
sys.path.append('../')
from mmcv import Config
from mmcv.utils import get_git_hash
from mmdet import __version__
from mmdet.apis import calculate_uncertainty
from mmdet.apis.train_SSD_L import train_detector_SSL
from mmdet.datasets import build_dataset, build_dataloader
from mmdet.models import build_detector
from mmdet.utils import collect_env, get_root_logger
from mmdet.utils.active_datasets import *
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)
from mmdet.utils.functions import *
import wandb

onlyUnc = False
onlyEval = False
noEval = False
showEval = False
boolScaleUnc = False
ismini = False
load_cycle = -1 # set to -1 if you don't want to load checkpoint.
resume_cycle = -1
isSave = True
isWandB = False
# editCfg = {}
editCfg = {'uncertainty_pool2':'objectSum_scaleMax_classSum'}
clsW = False
zeroRate = 0.15
saveMaxConf = False
useMaxConf = 'False'
score_thr = 0.3
iou_thr = 0.9
print(f'clsW is {clsW}, zeroRate is {zeroRate} useMaxConf is {useMaxConf}')
print(f'score_thr is {score_thr}, iou_thr is {iou_thr}')
if load_cycle >= 0 or resume_cycle >= 0:
    print(f'Load param of {load_cycle}cycle, Resume from {resume_cycle}cycle')

def parse_args():
    base_dir = './' if 'configs' in os.listdir() else '../'
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('--config', help='train config file path',
                        default = base_dir+'configs/_base_/Config_SSD.py',
                        # default = base_dir+'configs/_base_/Config_L_ReWeightST.py',
                        # default = base_dir+'configs/_base_/Entropy_ALL_Retina.py',
                        )
    parser.add_argument('--work-dir', help='the dir to save logs and models', default='SSD_L_OsSaCs_wBG') # No workdir --> No saved model!
    parser.add_argument('--uncertainty', help='uncertainty type')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--load-from', help='the checkpoint file to load from')
    parser.add_argument('--bbox-head') # default='Lambda_L2Net_ablation
    parser.add_argument('--no-validate', default=False, help='whether not to evaluate during training')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int, default=1,
        help='number of gpus to use (only applicable to non-distributed training)')
    group_gpus.add_argument('--gpu-ids', type=int, default=[0], nargs='+',
        help='ids of gpus to use (only applicable to non-distributed training)')
    parser.add_argument('--deterministic',action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--Unc-type', type=str)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args

def main():
    torch.set_num_threads(2)
    args = parse_args()
    cfg = Config.fromfile(args.config)
    random_seed = 20
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)
    cfg.seed = random_seed
    cfg.onlyEval = onlyEval
    if noEval: args.no_validate = True
    if args.bbox_head: cfg.model.bbox_head.type = args.bbox_head
    str2unc = {'SACA':'scaleAvg_classAvg', 'SSCS':'scaleSum_classSum',
               'SACS':'scaleAvg_classSum', 'SSCA':'scaleSum_classAvg'}
    if args.Unc_type:
        cfg.uncertainty_pool2 = str2unc[args.Unc_type]
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    base_dir = './' if 'configs' in os.listdir() else '../'
    if args.work_dir is not None:
        cfg.work_dir = osp.join(base_dir + './work_dirs', args.work_dir)
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(base_dir + './work_dirs', osp.splitext(osp.basename(args.config))[0])
    if not os.path.exists(cfg.work_dir):
        os.mkdir(cfg.work_dir)
    if args.work_dir and isWandB:
        api = wandb.Api()
        runs = api.runs('dnffkf369/active_object_detection_SSD')
        for run in runs:
            if run.name == args.work_dir:
                if run.state == 'running':
                    print(f'can"t delete the run {run}, since it is still running')
                else:
                    run.delete()
        wandb.init(project="active_object_detection_SSD", name=args.work_dir)
        print(f'wandb.run.dir => {wandb.run.dir}')
        SaveCode(base_dir, wandb.run.dir)
        RemoveEmptyDir(wandb.run.dir)
    if 'save_dir' not in args:
        cfg.save_dir = osp.join(cfg.work_dir, 'model_save')
        if not os.path.exists(cfg.save_dir):
            os.mkdir(cfg.save_dir)
    if ismini:
        cfg.data.test = ConfigDatasetAL(ismini)
        cfg.data.val = ConfigDatasetTEST(ismini)
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    if args.gpu_ids is not None:
        cfg.gpu_ids = args.gpu_ids
    else:
        cfg.gpu_ids = range(1) if args.gpus is None else range(args.gpus)
    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    if showEval:
        cfg.evaluation.show = True
        cfg.evaluation.out_dir = 'show_dir'
    if editCfg: EditCfg(cfg, editCfg)
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)
    meta = dict()
    env_info_dict = collect_env()
    env_info = '\n'.join([(f'{k}: {v}') for k, v in env_info_dict.items()])
    # logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)
    # logger.info(f'Distributed training: {distributed}')
    # logger.info(f'Config:\n{cfg.pretty_text}')
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    meta['exp_name'] = osp.basename(args.config)
    ppid = os.getppid()
    cfg.data.workers_per_gpu = 8 if psutil.Process(ppid).name() == 'bash' else 0
    print(f'cfg.data.workers_per_gpu is set to {cfg.data.workers_per_gpu}')
    # changes for active learning
    X_L, X_U, X_all, all_anns = get_X_L_0_prev(cfg)
    np.save(cfg.work_dir + '/X_L_' + '0' + '.npy', X_L)
    np.save(cfg.work_dir + '/X_U_' + '0' + '.npy', X_U)
    notResumed = True
    for cycle in cfg.cycles:
        if resume_cycle >= 0 and notResumed:
            X_L, X_U = ResumeCycle(cfg, cycle, resume_cycle)
            if not isinstance(X_L, np.ndarray): continue
            notResumed = False

        logger.info(f'Current cycle is {cycle} cycle.')
        print(f'len of X_U:{len(X_U)}, X_L:{len(X_L)}')
        cfg = create_X_L_file(cfg, X_L, all_anns, cycle)
        model = build_detector(cfg.model)
        model.init_weights()
        if 'bias' in cfg.model.train_cfg:
            if cfg.model.train_cfg.bias == 'uniform':
                for sIdx, cls_conv in enumerate(model.bbox_head.cls_convs):
                    torch.nn.init.uniform_(cls_conv[-1].bias, -math.sqrt(1 / 2000), math.sqrt(1 / 2000))
            else:
                model.bbox_head.init_cfg['override'] = \
                    {'type': 'Normal', 'name': 'retina_cls', 'std': 0.01, 'bias': cfg.model.train_cfg.bias}
        if load_cycle >= 0:
            cfg_name = osp.splitext(osp.basename(args.config))[0]
            load_name = f'{cfg.save_dir}/{cfg_name}_Cycle{load_cycle}_Epoch{cfg.runner.max_epochs}_mycode.pth'
            checkpoint = load_checkpoint(model, load_name, map_location='cpu')
            print(f'model is loaded from {load_name}')
        datasets = [build_dataset(cfg.data.train)]
        if cfg.checkpoint_config is not None:
            cfg.checkpoint_config.meta = dict(mmdet_version=__version__ + get_git_hash()[:7], CLASSES=datasets[0].CLASSES)
        model.CLASSES = datasets[0].CLASSES

        for epoch in range(cfg.outer_epoch):
            if onlyUnc: break
            if epoch == cfg.outer_epoch - 1:
                # cfg.lr_config.step = initial_step
                cfg.lr_config.step = [1000]
            else:
                cfg.lr_config.step = [1000]
                cfg.evaluation.interval = 100

            cfg.optimizer['lr'] = 0.001
            if epoch == 0:
                logger.info(f'Epoch = {epoch}, First Label Set Training')
                cfg = create_X_L_file(cfg, X_L, all_anns, cycle) # reflect results of uncertainty sampling
                datasets = [build_dataset(cfg.data.train)]
                cfg.total_epochs = cfg.epoch_ratio[0]
                cfg_bak = cfg.deepcopy()
                train_detector_SSL(model, datasets, cfg, distributed=distributed,
                                  validate=(not args.no_validate), timestamp=timestamp, meta=meta)
                cfg = cfg_bak
                torch.cuda.empty_cache()

            cfg.evaluation.interval = 100
            if epoch == cfg.outer_epoch - 1:
                cfg.lr_config.step = [2]
                cfg.evaluation.interval = cfg.epoch_ratio[0]
                # cfg.evaluation.interval = 100
            # ---------- Label Set Training ---------- #
            logger.info(f'Epoch = {epoch}, Fully-Supervised Learning')
            cfg = create_X_L_file(cfg, X_L, all_anns, cycle)
            datasets = [build_dataset(cfg.data.train)]
            cfg.total_epochs = cfg.epoch_ratio[0]
            cfg_bak = cfg.deepcopy()
            eval_res = train_detector_SSL(model, datasets, cfg, distributed=distributed,
                                          validate=(not args.no_validate), timestamp=timestamp, meta=meta)
            cfg = cfg_bak
            torch.cuda.empty_cache()

            # SSL_epoch = 2
            # if epoch == cfg.outer_epoch - 1:
            #     cfg.evaluation.interval = SSL_epoch
            #     cfg.optimizer['lr'] = 0.1 * cfg.optimizer['lr']
            # # ---------- Label + UnLabel Set Training ----------
            # logger.info(f'Epoch = {epoch}, Semi-Supervised Learning')
            # cfg_u = create_X_U_file(cfg.deepcopy(), X_U, all_anns, cycle)
            # cfg = create_X_L_file(cfg, X_L, all_anns, cycle)
            # datasets, datasets_u = [build_dataset(cfg.data.train)], [build_dataset(cfg_u.data.train)]
            # cfg.total_epochs, cfg_u.total_epochs = SSL_epoch, SSL_epoch
            # cfg_u_bak = cfg_u.deepcopy()
            # cfg_bak = cfg.deepcopy()
            # train_detector_SSL(model, [datasets, datasets_u], [cfg, cfg_u],
            #                    distributed=distributed, validate=(not args.no_validate), timestamp=timestamp, meta=meta)
            # cfg_u, cfg = cfg_u_bak, cfg_bak
            # torch.cuda.empty_cache()
            if args.work_dir and isWandB and eval_res:
                wandb.log(eval_res)

        if isSave:
            for file in os.listdir(cfg.save_dir): # To save memory
                if not '_mycode' in file:
                    os.remove(os.path.join(cfg.save_dir, file))
            cfg_name = osp.splitext(osp.basename(args.config))[0]
            save_name = f'{cfg.save_dir}/{cfg_name}_Cycle{cycle}_Epoch{cfg.runner.max_epochs}_mycode.pth'
            torch.save(model.state_dict(), save_name)

        if cycle != cfg.cycles[-1]:
            # get new labeled data
            dataset_al = build_dataset(cfg.data.test)
            data_loader = build_dataloader(dataset_al, samples_per_gpu=cfg.data.samples_per_gpu,
                                           workers_per_gpu=cfg.data.workers_per_gpu, dist=False, shuffle=False)
            if not distributed:
                poolModel = MMDataParallel(model, device_ids=cfg.gpu_ids)
            else:
                poolModel = MMDistributedDataParallel(model.cuda(), device_ids=[torch.cuda.current_device()],
                            broadcast_buffers=False)
            torch.cuda.empty_cache()
            with torch.no_grad():
                # uncOuts = calculate_uncertainty(cfg, poolModel, data_loader, return_box=False, clsW=clsW,
                #                                 scaleUnc=boolScaleUnc)
                uncOuts = calculate_uncertainty(cfg, poolModel, data_loader, return_box=False, showNMS = False,
                                                saveUnc=False, saveMaxConf=saveMaxConf, clsW=clsW, scaleUnc=boolScaleUnc,
                                                score_thr = score_thr, iou_thr = iou_thr)
            if saveMaxConf:
                maxconf = uncOuts[1]
                uncertainty = uncOuts[0]
            else:
                maxconf = None
                uncertainty = uncOuts
            if torch.is_tensor(uncertainty):
                uncertainty = uncertainty.numpy()
            elif not isinstance(uncertainty, np.ndarray):
                uncertainty = torch.stack(uncertainty).numpy()
            X_L, X_U = update_X_L(uncertainty, X_all, X_L, cfg.X_S_size, zeroRate=zeroRate,
                                  maxconf = maxconf, useMaxConf=useMaxConf)
            # save set and model
            np.save(cfg.work_dir + '/X_L_' + str(cycle + 1) + '.npy', X_L)
            np.save(cfg.work_dir + '/X_U_' + str(cycle + 1) + '.npy', X_U)
            np.save(cfg.work_dir + '/Unc_' + str(cycle + 1) + '.npy', uncertainty)
            torch.cuda.empty_cache()
        DelJunkSave(cfg.work_dir)
if __name__ == '__main__':
    main()
