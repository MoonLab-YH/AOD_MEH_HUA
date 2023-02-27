import os.path as osp
import pickle
import shutil
import tempfile
import time
import pdb
import mmcv
import torch
import torch.distributed as dist
import torch.nn as nn
import numpy as np
import cv2
from mmcv.image import tensor2imgs
from mmcv.runner import get_dist_info
from mmdet.core import encode_mask_results
from mmdet.utils.functions import *
import time

class Uncertainty_fns:
    @staticmethod
    def Random(cfg, *args, **kwargs):
        data_loader = args[1]
        dataset = data_loader.dataset
        print('>>> Computing Random Uncertainty...')
        uncertainty = torch.randperm(len(dataset)).numpy()
        return uncertainty

    @staticmethod
    @torch.no_grad()
    def Entropy_ALL(cfg, *args, **kwargs):
        model, dataloader = args
        model.eval()
        print('>>> Computing Entropy_ALL Uncertainty...')
        uType = cfg.uncertainty_type
        uPool = cfg.uncertainty_pool
        uPool2 = cfg.uncertainty_pool2
        uncertainties = single_gpu_uncertainty(model, dataloader, isUnc=uType, uPool=uPool, uPool2 = uPool2, **kwargs)
        # return uncertainties.cpu()
        return list(map(lambda obj:obj.cpu() if torch.is_tensor(obj) else obj, uncertainties))

    @staticmethod
    @torch.no_grad()
    def Entropy_NoNMS(cfg, *args, **kwargs):
        model, dataloader = args
        model.eval()
        print('>>> Computing Entropy_NoNMS Uncertainty...')
        uType = cfg.uncertainty_type
        uPool = cfg.uncertainty_pool
        uncertainties = single_gpu_uncertainty(model, dataloader, isUnc=uType, uPool=uPool)
        return uncertainties.cpu()

    @staticmethod
    @torch.no_grad()
    def Entropy_NMS(cfg, *args, **kwargs):
        model, dataloader = args
        model.eval()
        print('>>> Computing Entropy_NMS Uncertainty...')
        uType = cfg.uncertainty_type
        uPool = cfg.uncertainty_pool
        uPool2 = cfg.uncertainty_pool2
        uncertainties = single_gpu_uncertainty(model, dataloader, isUnc=uType, uPool=uPool, uPool2 = uPool2, **kwargs)
        # return uncertainties.cpu()
        return list(map(lambda obj:obj.cpu() if torch.is_tensor(obj) else obj, uncertainties))

    @staticmethod
    @torch.no_grad()
    def Entropy_Avg(cfg, *args, **kwargs):
        model, dataloader = args
        model.eval()
        print('>>> Computing Entropy_NMS Uncertainty...')
        uType = cfg.uncertainty_type
        uPool = cfg.uncertainty_pool
        uPool2 = cfg.uncertainty_pool2
        uncertainties = single_gpu_uncertainty(model, dataloader, isUnc=uType, uPool=uPool, uPool2 = uPool2, **kwargs)
        # return uncertainties.cpu()
        return list(map(lambda obj:obj.cpu() if torch.is_tensor(obj) else obj, uncertainties))


def calculate_uncertainty(cfg, *args, **kwargs):
    if 'showNMS' in kwargs and kwargs['showNMS']:
        DeleteImgs('visualization2')
    uncertainty_pool = cfg.uncertainty_pool
    uncertainty_fn = getattr(Uncertainty_fns, uncertainty_pool)
    return uncertainty_fn(cfg, *args, **kwargs)

def Ensemble_uncertainty(cfg, m1, m2, m3, data_loader, **kwargs):
    uncertainties = Ensemble_MI(m1, m2, m3, data_loader, **kwargs)
    return uncertainties

def single_gpu_uncertainty(model, data_loader, **kwargs):
    model.eval()
    uncertainties = []
    scaleUncs, maxconfs = [], []
    img_metas = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data['img'] = data['img'].data # Resolve mismatch between single_gpu_test and train_pipeline
            data['img_metas'] = data['img_metas'].data
        def DrawLoss():
            DeleteContent('visualization')
            data['img_metas'] = data['img_metas'][0]
            data['img'] = data['img'][0]
            loss, head_out, feat_out, prev_loss = model.train_step(data, **kwargs, Labeled=True, Pseudo=False)
            for sIdx, (loss, feat) in enumerate(zip(prev_loss, feat_out)):
                B,C,H,W = feat.shape
                _loss = loss.reshape(B,H,W,9)
                for iIdx in range(B):
                    visualize(data['img'][iIdx], f'visualization/{iIdx}_img.jpg')
                    iLoss = _loss[iIdx].max(dim=-1)[0]
                    # iLoss = _loss[iIdx].sum(dim=-1)
                    visualize(iLoss, f'visualization/{iIdx}_loss_{sIdx}.jpg', size=(H*(2**sIdx),W*(2**sIdx)), heatmap=True)
        # DrawLoss()
        result, *UncOuts = model(return_loss=False, rescale=True, isEval=False, batchIdx=i, **data, **kwargs)
        batch_size = len(result)
        if len(UncOuts) > 1:
            others = UncOuts[1:]
            UncOuts = UncOuts[0]
        for _ in range(batch_size):
            prog_bar.update()
        try:
            while isinstance(UncOuts[0], list):
                UncOuts = UncOuts[0]
            uncertainties.extend(UncOuts)
        except:
            pass
            # print('Error Raised..!')
        if kwargs['scaleUnc']: scaleUncs.extend(UncOuts[1])
        if 'saveMaxConf' in kwargs and kwargs['saveMaxConf']: maxconfs.extend(others[0])
        img_metas.extend(data['img_metas'][0])
    uncertainties = torch.tensor(uncertainties)
    if kwargs['scaleUnc']: return uncertainties, scaleUncs
    if 'saveMaxConf' in kwargs and kwargs['saveMaxConf']: return uncertainties, maxconfs
    return uncertainties

def Ensemble_MI(m1, m2, m3, data_loader, **kwargs):
    # return torch.arange(16551)
    m1.eval(); m2.eval(); m3.eval()
    uncertainties = []
    latencies = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            data['img'] = data['img'].data # Resolve mismatch between single_gpu_test and train_pipeline
            data['img_metas'] = data['img_metas'].data
        start = time.time()
        m1Out = m1(return_loss=False, rescale=True, isEval=True, justOut=True, batchIdx=i, **data, **kwargs)
        m2Out = m2(return_loss=False, rescale=True, isEval=True, justOut=True, batchIdx=i, **data, **kwargs)
        m3Out = m3(return_loss=False, rescale=True, isEval=True, justOut=True, batchIdx=i, **data, **kwargs)
        batch_size = len(m1Out[0])
        for _ in range(batch_size):
            prog_bar.update()
        UncOuts = ComputeMI(m1Out, m2Out, m3Out, nCls=20)
        end = time.time()
        uncertainties.extend(UncOuts)
        latencies.append(end-start)
    torch.cuda.empty_cache()
    uncertainties = torch.tensor(uncertainties)
    print(f'latency is {torch.tensor(latencies).mean()/2}')
    return uncertainties

def ComputeMI(m1Out, m2Out, m3Out, nCls = 20):
    S, B = len(m1Out), len(m1Out[0])
    buffer = torch.zeros(B,S)
    for sIdx, (O1, O2, O3) in enumerate(zip(m1Out, m2Out, m3Out)):
        for bIdx, (o1, o2, o3) in enumerate(zip(O1,O2,O3)):
            o1 = F.sigmoid(o1).permute([1,2,0]).reshape(-1,nCls)
            o2 = F.sigmoid(o2).permute([1,2,0]).reshape(-1,nCls)
            o3 = F.sigmoid(o3).permute([1,2,0]).reshape(-1,nCls)
            preds = torch.cat((o1[None],o2[None],o3[None]), dim=0)
            avg = preds.mean(dim=0)
            total = (-avg * avg.log()).sum(dim=1)
            ent = (-preds * preds.log()).sum(dim=-1)
            aleatoric = ent.mean(dim=0)
            epistemic = total - aleatoric
            buffer[bIdx,sIdx] = epistemic.mean()
    out =  buffer.mean(dim=-1).tolist()
    return out

