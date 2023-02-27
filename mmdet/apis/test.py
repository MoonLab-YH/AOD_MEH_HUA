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

def calculate_uncertainty(cfg, *args, **kwargs):
    if 'showNMS' in kwargs and kwargs['showNMS']:
        DeleteImgs('visualization2')
    uncertainty_pool = cfg.uncertainty_pool
    uncertainty_fn = getattr(Uncertainty_fns, uncertainty_pool)
    return uncertainty_fn(cfg, *args, **kwargs)

@torch.no_grad()
def cal_numObj(cfg, *args, **kwargs):
    model, dataloader = args
    model.eval()
    print('>>> Computing Entropy_NMS Uncertainty...')
    uncertainties = []
    dataset = dataloader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for bIdx, data in enumerate(dataloader):
        data['img'] = data['img'].data
        batch_size = len(data['img'][0])
        for i in range(batch_size):
            prog_bar.update()
            numObj = len(data['gt_bboxes'].data[0][i])
            uncertainties.append(numObj)
    uncertainties = torch.tensor(uncertainties)
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
            a=2
        except:
            print('Error Raised..!')
        if kwargs['scaleUnc']: scaleUncs.extend(UncOuts[1])
        if 'saveMaxConf' in kwargs and kwargs['saveMaxConf']: maxconfs.extend(others[0])
        img_metas.extend(data['img_metas'][0])
    uncertainties = torch.tensor(uncertainties)
    if kwargs['scaleUnc']: return uncertainties, scaleUncs
    if 'saveMaxConf' in kwargs and kwargs['saveMaxConf']: return uncertainties, maxconfs
    return uncertainties


def single_gpu_test(model,
                    data_loader,
                    show=False,
                    out_dir=None,
                    show_score_thr=0.3,
                    **kwargs):
    model.eval()
    results = []
    dataset = data_loader.dataset
    prog_bar = mmcv.ProgressBar(len(dataset))
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            # normed = data['img'][0][0].permute([1,2,0])
            # normnp = np.array(normed)
            # fliped = torch.tensor(cv2.flip(normnp, 0)).permute([2,0,1])
            # data['img'][0][0] = fliped
            # result = model(return_loss=False, rescale=True, isEval=True, isUnc = kwargs['isUnc'], **data)
            result = model(return_loss=False, rescale=True, isEval=True, isUnc = None, **data)
        # Check if sampled images change every time.
        batch_size = len(result)
        if show or out_dir:
            if batch_size == 1 and isinstance(data['img'][0], torch.Tensor):
                img_tensor = data['img'][0]
            else:
                img_tensor = data['img'][0].data[0]
            img_metas = data['img_metas'][0].data[0]
            imgs = tensor2imgs(img_tensor, **img_metas[0]['img_norm_cfg'])
            assert len(imgs) == len(img_metas)

            for i, (img, img_meta) in enumerate(zip(imgs, img_metas)):
                h, w, _ = img_meta['img_shape']
                img_show = img[:h, :w, :]

                ori_h, ori_w = img_meta['ori_shape'][:-1]
                img_show = mmcv.imresize(img_show, (ori_w, ori_h))

                if out_dir:
                    out_file = osp.join(out_dir, img_meta['ori_filename'])
                else:
                    out_file = None

                model.module.show_result(
                    img_show,
                    result[i],
                    show=show,
                    out_file=out_file,
                    score_thr=show_score_thr,
                    **kwargs)

        # encode mask results
        if isinstance(result[0], tuple):
            result = [(bbox_results, encode_mask_results(mask_results))
                      for bbox_results, mask_results in result]
        results.extend(result)

        for _ in range(batch_size):
            prog_bar.update()
    return results

def multi_gpu_test(model, data_loader, tmpdir=None, gpu_collect=False):
    """Test model with multiple gpus.
    This method tests model with multiple gpus and collects the results
    under two different modes: gpu and cpu modes. By setting 'gpu_collect=True'
    it encodes results to gpu tensors and use gpu communication for results
    collection. On cpu mode it saves the results on different gpus to 'tmpdir'
    and collects them by the rank 0 worker.
    Args:
        model (nn.Module): Model to be tested.
        data_loader (nn.Dataloader): Pytorch data loader.
        tmpdir (str): Path of directory to save the temporary results from
            different gpus under cpu mode.
        gpu_collect (bool): Option to use either gpu or cpu to collect results.
    Returns:
        list: The prediction results.
    """
    model.eval()
    results = []
    dataset = data_loader.dataset
    rank, world_size = get_dist_info()
    if rank == 0:
        prog_bar = mmcv.ProgressBar(len(dataset))
    time.sleep(2)  # This line can prevent deadlock problem in some cases.
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            result = model(return_loss=False, rescale=True, **data)
            # encode mask results
            if isinstance(result[0], tuple):
                result = [(bbox_results, encode_mask_results(mask_results))
                          for bbox_results, mask_results in result]
        results.extend(result)

        if rank == 0:
            batch_size = len(result)
            for _ in range(batch_size * world_size):
                prog_bar.update()

    # collect results from all ranks
    if gpu_collect:
        results = collect_results_gpu(results, len(dataset))
    else:
        results = collect_results_cpu(results, len(dataset), tmpdir)
    return results


def collect_results_cpu(result_part, size, tmpdir=None):
    rank, world_size = get_dist_info()
    # create a tmp dir if it is not specified
    if tmpdir is None:
        MAX_LEN = 512
        # 32 is whitespace
        dir_tensor = torch.full((MAX_LEN, ),
                                32,
                                dtype=torch.uint8,
                                device='cuda')
        if rank == 0:
            mmcv.mkdir_or_exist('.dist_test')
            tmpdir = tempfile.mkdtemp(dir='.dist_test')
            tmpdir = torch.tensor(
                bytearray(tmpdir.encode()), dtype=torch.uint8, device='cuda')
            dir_tensor[:len(tmpdir)] = tmpdir
        dist.broadcast(dir_tensor, 0)
        tmpdir = dir_tensor.cpu().numpy().tobytes().decode().rstrip()
    else:
        mmcv.mkdir_or_exist(tmpdir)
    # dump the part result to the dir
    mmcv.dump(result_part, osp.join(tmpdir, f'part_{rank}.pkl'))
    dist.barrier()
    # collect all parts
    if rank != 0:
        return None
    else:
        # load results of all parts from tmp dir
        part_list = []
        for i in range(world_size):
            part_file = osp.join(tmpdir, f'part_{i}.pkl')
            part_list.append(mmcv.load(part_file))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        # remove tmp dir
        shutil.rmtree(tmpdir)
        return ordered_results


def collect_results_gpu(result_part, size):
    rank, world_size = get_dist_info()
    # dump result part to tensor with pickle
    part_tensor = torch.tensor(
        bytearray(pickle.dumps(result_part)), dtype=torch.uint8, device='cuda')
    # gather all result part tensor shape
    shape_tensor = torch.tensor(part_tensor.shape, device='cuda')
    shape_list = [shape_tensor.clone() for _ in range(world_size)]
    dist.all_gather(shape_list, shape_tensor)
    # padding result part tensor to max length
    shape_max = torch.tensor(shape_list).max()
    part_send = torch.zeros(shape_max, dtype=torch.uint8, device='cuda')
    part_send[:shape_tensor[0]] = part_tensor
    part_recv_list = [
        part_tensor.new_zeros(shape_max) for _ in range(world_size)
    ]
    # gather all result part
    dist.all_gather(part_recv_list, part_send)

    if rank == 0:
        part_list = []
        for recv, shape in zip(part_recv_list, shape_list):
            part_list.append(
                pickle.loads(recv[:shape[0]].cpu().numpy().tobytes()))
        # sort the results
        ordered_results = []
        for res in zip(*part_list):
            ordered_results.extend(list(res))
        # the dataloader may pad some samples
        ordered_results = ordered_results[:size]
        return ordered_results