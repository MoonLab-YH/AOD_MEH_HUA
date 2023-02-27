from PIL import Image
import numpy as np
import shutil
import torch
import torch.nn.functional as F
import torch.nn as nn
import os
import time
import random
from tqdm import tqdm
import cv2
from mmcv.runner import (get_dist_info, init_dist, load_checkpoint, wrap_fp16_model)

def visualize(imgten, path, color=True, threshold = False, size=None, reverse = False, heatmap=False):
    imgten = imgten.detach().float()
    if color: # input should be [C,W,H]
        if imgten.size(0) == 3:
            if size != None:
                imgten = F.interpolate(imgten.unsqueeze(dim=0), size=size, mode='bilinear', align_corners=True)
                imgnp = imgten[0].detach().cpu().numpy().transpose([1, 2, 0])
            else:
                imgnp = imgten.permute([1,2,0]).cpu().numpy()
        elif len(imgten.shape) == 2:
            if size != None:
                imgten = F.interpolate(imgten[None,None,...], size=size, mode='bilinear', align_corners=True)
                imgnp = imgten[0].detach().cpu().numpy().transpose([1, 2, 0])
            else:
                imgnp = imgten.cpu().numpy()
        imgnp = np.interp(imgnp, (imgnp.min(), imgnp.max()), (0,255)).astype(np.uint8)
        if heatmap:
            imgnp = cv2.applyColorMap(imgnp, cv2.COLORMAP_JET)
            cv2.imwrite(path, imgnp)
            return
        img = Image.fromarray(imgnp)
        img.save(path)
    else: #grayscale, input should be [W,H]
        imgten = imgten.unsqueeze(dim=0).unsqueeze(dim=0).float()
        if size!= None:
            imgten = F.interpolate(imgten, size=size, mode='bilinear', align_corners=True)
        imgnp = imgten[0,0].detach().cpu().numpy()
        imgnp = np.interp(imgnp, (imgnp.min(), imgnp.max()), (0,255)).astype(np.uint8)
        if threshold:
            imgnp[imgnp<threshold] = 0; imgnp[imgnp>=threshold] = 255
        if reverse:
            imgnp = 255 - imgnp
        img = Image.fromarray(imgnp)
        img.save(path)

def DeleteContent(path):
    if not os.path.exists(path):
        os.mkdir(path)
    eval_list = os.listdir(path)
    for i in eval_list:
        if os.path.isdir(os.path.join(path,i)):
            shutil.rmtree(os.path.join(path,i))
        else:
            os.remove(os.path.join(path,i))

def FlipGT(GTs, H, W):
    Out = []
    for GT in GTs:
        out = []
        for gt in GT:
            _x1, _y1, _x2, _y2 = gt.tolist()
            out.append(torch.tensor([W-_x2, _y1, W-_x1, _y2]))
        Out.append(torch.stack(out).reshape(-1,4))
    return Out


def RotateGT(degree, GTs, H, W):
    Out = []
    for GT in GTs:
        out = []
        for gt in GT:
            _x1, _y1, _x2, _y2 = gt.tolist()
            if degree == 0:
                out.append(torch.tensor([_x1, _y1, _x2, _y2]))
            elif degree == 1:
                x1, y1, x2, y2 = [_y1, W-_x2, _y2, W-_x1]
                out.append(torch.tensor([x1, y1, x2, y2]))
            elif degree == 2:
                x1, y1, x2, y2 = [W-_x2, H-_y2, W-_x1, H-_y1]
                out.append(torch.tensor([x1, y1, x2, y2]))
            elif degree == 3:
                x1, y1, x2, y2 = [H - _y2, _x1, H - _y1, _x2]
                out.append(torch.tensor([x1, y1, x2, y2]))
        Out.append(torch.stack(out).reshape(-1,4))
    return Out

def RotateMeta(degree, img_metas):
    Out = []
    for img_meta in img_metas:
        tmp = {}
        for key in ['img_shape', 'pad_shape']:
            H,W,_ = img_meta[key]
            if degree == 0 or degree == 2:
                tmp[key] = (H,W,3)
            else:
                tmp[key] = (W,H,3)
        Out.append(tmp)
    return Out

def SaveCode(base_path, des_path):
    if (not os.path.isdir(base_path)):
        if '.py' in base_path:
            save_path = os.path.join(des_path, base_path.replace('../', ''))
            shutil.copy(base_path, save_path)
        return
    if 'wandb' in base_path: return
    filenames = os.listdir(base_path)
    for file in filenames:
        longPath = os.path.join(base_path, file)
        save_path = os.path.join(des_path, longPath.replace('../',''))
        if os.path.isdir(longPath):
            os.mkdir(save_path)
        SaveCode(longPath, des_path)

def RemoveEmptyDir(path):
    filenames = os.listdir(path)
    for file in filenames:
        longPath = os.path.join(path, file)
        if os.path.isdir(longPath):
            if len(os.listdir(longPath)) == 0:
                os.rmdir(longPath)
                continue
            RemoveEmptyDir(longPath)

def DrawGT(img, GT, path, GT_labels=None, cls_color=False):
    Colors = {0:(255,0,0), 1:(255,128,0), 2:(255,255,0), 3:(128,255,0), 4:(0,255,0),
              5:(0,255,128),6:(0,255,255),7:(0,128,255),8:(0,0,255),9:(128,0,255),
              10:(255,0,255),11:(255,0,128),12:(128,128,128),13:(255,255,255),14:(102,0,102),
              15:(123,113,52),16:(244,49,156),17:(79,32,154),18:(211,45,219),19:(105,50,177)}
    img = img.clone()
    width = 1
    for i, gt in enumerate(GT):
        x1,y1,x2,y2 = list(map(int, gt.tolist()))
        if x1 < width:
            x1 = width
        if y1 < width:
            y1 = width
        if GT_labels is not None:
            colors = {cls.cpu().item():idx+1 for idx,cls in enumerate(GT_labels.unique())}
            color = (colors[GT_labels[i].cpu().item()] / float(len(GT_labels.unique())))**2
        else:
            color = (0.8 ** i)
        img[:,y1:y2,x1-width:x1+width] = img.max() * color
        img[:,y1:y2,x2-width:x2+width] = img.max() * color
        img[:,y1-width:y1+width,x1:x2] = img.max() * color
        img[:,y2-width:y2+width,x1:x2] = img.max() * color
        if cls_color:
            r,g,b = Colors[GT_labels[i].item()]
            img[0, y1:y2, x1 - width:x1 + width] = img.max() * r / 255
            img[1, y1:y2, x1 - width:x1 + width] = img.max() * g / 255
            img[2, y1:y2, x1 - width:x1 + width] = img.max() * b / 255
            img[0, y1:y2, x2 - width:x2 + width] = img.max() * r / 255
            img[1, y1:y2, x2 - width:x2 + width] = img.max() * g / 255
            img[2, y1:y2, x2 - width:x2 + width] = img.max() * b / 255
            img[0, y1 - width:y1 + width, x1:x2] = img.max() * r / 255
            img[1, y1 - width:y1 + width, x1:x2] = img.max() * g / 255
            img[2, y1 - width:y1 + width, x1:x2] = img.max() * b / 255
            img[0, y2 - width:y2 + width, x1:x2] = img.max() * r / 255
            img[1, y2 - width:y2 + width, x1:x2] = img.max() * g / 255
            img[2, y2 - width:y2 + width, x1:x2] = img.max() * b / 255

    imgten = img.detach().float()
    imgnp = imgten.permute([1, 2, 0]).cpu().numpy()
    imgnp = np.interp(imgnp, (imgnp.min(), imgnp.max()), (0, 255)).astype(np.uint8)
    img = Image.fromarray(imgnp)
    img.save(path)

    for i, (gt, gt_label) in enumerate(zip(GT, GT_labels)):
        x1, y1, x2, y2 = list(map(int, gt.tolist()))
        cv2.putText(imgnp, str(i), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)

    img = Image.fromarray(imgnp)
    img.save(path[:-4] + '_number.jpg')
    a=2


def DrawGT_backup(img, GT, path, GT_labels=None):
    img = img.clone()
    width = 1
    for i, gt in enumerate(GT):
        x1, y1, x2, y2 = list(map(int, gt.tolist()))
        if x1 < width:
            x1 = width
        if y1 < width:
            y1 = width
        if GT_labels is not None:
            colors = {cls.cpu().item(): idx + 1 for idx, cls in enumerate(GT_labels.unique())}
            color = (colors[GT_labels[i].cpu().item()] / float(len(GT_labels.unique()))) ** 2
        else:
            color = (0.8 ** i)
        img[:, y1:y2, x1 - width:x1 + width] = img.max() * color
        img[:, y1:y2, x2 - width:x2 + width] = img.max() * color
        img[:, y1 - width:y1 + width, x1:x2] = img.max() * color
        img[:, y2 - width:y2 + width, x1:x2] = img.max() * color

    visualize(img, path)

def FuseList(list_of_tensor, selectFirst=False):
    totalLen = 0
    for _tensor in list_of_tensor:
        if selectFirst: totalLen += len(_tensor[0])
        else: totalLen += len(_tensor)

    firstTen = list_of_tensor[0]
    if selectFirst: firstTen = firstTen[0]
    if firstTen.dim() == 1:
        rst = firstTen.new_full((totalLen,), fill_value=0)
    else:
        rst = firstTen.new_full((totalLen,) + firstTen.size()[1:], fill_value=0)
    start ,end = 0, 0
    for _tensor in list_of_tensor:
        if selectFirst:
            end += len(_tensor[0])
            rst[start:end] = _tensor[0]
        else:
            end += len(_tensor)
            rst[start:end] = _tensor
        start = end
    return rst

def MakeWeights(featmap_size, pad_shape, sIdx, device):
    strides = [8, 16, 32, 64, 128]
    feat_h, feat_w = featmap_size
    h, w = pad_shape[:2]
    anchor_stride = strides[sIdx]
    valid_h = min(int(np.ceil(h / anchor_stride)), feat_h)
    valid_w = min(int(np.ceil(w / anchor_stride)), feat_w)
    assert valid_h <= feat_h and valid_w <= feat_w
    valid_x = torch.zeros(feat_w, dtype=torch.bool, device=device)
    valid_y = torch.zeros(feat_h, dtype=torch.bool, device=device)
    valid_x[:valid_w] = 1
    valid_y[:valid_h] = 1
    valid_xx, valid_yy = _meshgrid(valid_x, valid_y)
    valid = valid_xx & valid_yy
    num_base_anchors = 9
    valid = valid[:, None].expand(valid.size(0), num_base_anchors).contiguous().view(-1)
    return valid

def _meshgrid(x, y, row_major=True):
    xx = x.repeat(y.shape[0])
    yy = y.view(-1, 1).repeat(1, x.shape[0]).view(-1)
    if row_major:
        return xx, yy
    else:
        return yy, xx

def VisualizeData(data, model, cfg, **kwargs):
    torch.cuda.empty_cache()
    device = list(model.parameters())[0].device
    # Resolve mismatch from direct sampling to tset_pipeline
    data['img_metas'] = data['img_metas'].data
    data['gt_bboxes'] = data['gt_bboxes'].data
    data['gt_labels'] = data['gt_labels'].data
    if not kwargs['fromLoader']:
        data['img'] = data['img'].data.to(device)
        data['img'] = [data['img'][None,:]]
        data['img_metas'] = [[data['img_metas']]]
        data['gt_bboxes'] = [data['gt_bboxes']]
        data['gt_labels'] = [data['gt_labels']]
    else:
        data['img'] = data['img'].data
        data['img'][0] = data['img'][0].to(device)
    uType = cfg.uncertainty_type
    uPool = cfg.uncertainty_pool
    uPool2 = cfg.uncertainty_pool2
    model.eval()
    loss, *UncOuts = model(**data, return_loss=False, rescale=True, isEval=False, isUnc=uType, uPool=uPool,
                           uPool2 = uPool2, Labeled=True, Pseudo=False, draw=False, saveUnc=True, **kwargs)
    return (loss, *UncOuts)

dataset_type = 'VOCDataset'
data_root = '/drive1/YH/datasets/VOCdevkit/VOCdevkit/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1000, 600), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1000, 600),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

def ConfigDatasetAL(ismini=False):
    if ismini:
        return dict(
               type=dataset_type,
               ann_file=data_root + 'VOC2007/ImageSets/Main/mini_test.txt',
               img_prefix=data_root + 'VOC2007/',
               pipeline=train_pipeline)
    else:
        return dict(
        type=dataset_type,
        ann_file=[data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                  data_root + 'VOC2012/ImageSets/Main/trainval.txt'],
        img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
        pipeline=train_pipeline)

def ConfigDatasetALCustom(custom=False):
    if custom:
        return dict(
           type=dataset_type,
           ann_file=[data_root + 'VOC2007/ImageSets/Main/custom_test.txt',
                     data_root + 'VOC2012/ImageSets/Main/custom_test.txt'],
           img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
           pipeline=train_pipeline)
    else:
        return dict(
        type=dataset_type,
        ann_file=[data_root + 'VOC2007/ImageSets/Main/trainval.txt',
                  data_root + 'VOC2012/ImageSets/Main/trainval.txt'],
        img_prefix=[data_root + 'VOC2007/', data_root + 'VOC2012/'],
        pipeline=train_pipeline)

def ConfigDatasetTEST(ismini=False):
    if ismini:
        return dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/mini_test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline)
    else:
        return dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline)

def ConfigDatasetTESTCustom(custom=False):
    if custom:
        return dict(
        type=dataset_type,
        ann_file=[data_root + 'VOC2007/ImageSets/Main/custom_test.txt',
                  data_root + 'VOC2012/ImageSets/Main/custom_test.txt'],
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline)
    else:
        return dict(
        type=dataset_type,
        ann_file=data_root + 'VOC2007/ImageSets/Main/test.txt',
        img_prefix=data_root + 'VOC2007/',
        pipeline=test_pipeline)

def DelJunkSave(work_dir):
    for file in os.listdir(work_dir):
        if '.pth' in file:
            os.remove(os.path.join(work_dir, file))

def SamplingViaLoader(data_loader, index):
    for i, data in enumerate(data_loader):
        if i != index:
            continue
        print(f'The loader is returning {i}st batch.')

        return data

def mmm(tensor):
    return tensor.max().item(), tensor.mean().item(), tensor.min().item()

def ShowSelectedImg(work_dir, cycle, dataset_al, model, cfg):
    fileName = os.path.join(work_dir, f'Unc_{cycle}.npy')
    idxName = os.path.join(work_dir, f'X_L_{cycle}.npy')
    prevName = os.path.join(work_dir, f'X_L_{cycle-1}.npy')
    dirName = os.path.split(work_dir)[1]
    if not os.path.exists(os.path.join('visualization',dirName)):
        os.mkdir(os.path.join('visualization',dirName))
    else:
        DeleteContent(os.path.join('visualization',dirName))
    uncertainty = np.load(fileName)
    argSorted = (uncertainty).argsort()
    X_L = np.load(idxName)
    X_L_prev = np.load(prevName)
    X_L_cur = np.setdiff1d(X_L, X_L_prev)
    for i in tqdm(range(len(X_L_cur))):
        idx = X_L_cur[i]
        torch.cuda.empty_cache()
        data = dataset_al[idx]
        device = list(model.parameters())[0].device
        data['img_metas'] = data['img_metas'].data
        data['gt_bboxes'] = data['gt_bboxes'].data
        data['gt_labels'] = data['gt_labels'].data
        data['img'] = data['img'].data.to(device)
        data['img'] = [data['img'][None, :]]
        data['img_metas'] = [[data['img_metas']]]
        data['gt_bboxes'] = [data['gt_bboxes']]
        data['gt_labels'] = [data['gt_labels']]
        uType = cfg.uncertainty_type
        uPool = cfg.uncertainty_pool
        uPool2 = cfg.uncertainty_pool2
        model.eval()
        loss, *UncOuts = model(**data, return_loss=False, rescale=True, isEval=False, isUnc=uType, uPool=uPool,
                               uPool2=uPool2, Labeled=True, Pseudo=False, saveUnc=True, name=f'{i}', dirName=dirName)


def EditCfg(cfg, new_dict):
    print(' ===== EditCfg is ... ===== ')
    print(new_dict)
    for key, val in new_dict.items():
        if key in cfg:
            cfg[key] = val

def Shape(container):
    return [i.shape for i in container]

def ExtractAggFunc(type):
    splitTypes = type.split('_')
    funcDict = {'Sum': torch.sum, 'Avg': torch.mean, 'Max': torch.max}
    names = ['object', 'scale', 'class']
    output = {}
    for name in names:
        for splitType in splitTypes:
            if name in splitType:
                funcName = splitType.replace(name,'')
                func = funcDict[funcName]
                output[name] = func
    return output

def StartEnd(mlvl, sIdx):
    start, end = 0, 0
    for si, slvl in enumerate(mlvl):
        end = end + slvl.size(1)
        if si == sIdx:
            return start, end
        start = end

def DeleteImgs(path):
    if not os.path.exists(path):
        os.mkdir(path)
    eval_list = os.listdir(path)
    for i in eval_list:
        if '.jpg' in i:
            os.remove(os.path.join(path,i))

def ShowUncZero(Uncs, dataset_al, model, cfg, **kwargs):
    DeleteImgs('visualization')
    zeroIdx = (Uncs == 0).nonzero()[0]
    for idx in tqdm(zeroIdx):
        EasyOne = dataset_al[idx]
        VisualizeData(EasyOne, model, cfg, name=f'zero_{idx}', fromLoader=False, **kwargs)

def FindRunDir(name, path='/drive2/YH/[MX32]Active_Tracking_Project/[MyCodes]/MMdet_study/MIAOD_based_AOD/wandb/'):
    dirList = os.listdir(path)
    for dir in dirList:
        if name in dir:
            return dir

def getMaxConf(mlvl_cls_scores, nCls):
    B = mlvl_cls_scores[0].size(0)
    nScale = len(mlvl_cls_scores)
    device = mlvl_cls_scores[0].device
    output = torch.zeros(B, nScale).to(device)
    for sIdx, cls_scores in enumerate(mlvl_cls_scores):
        bar = cls_scores.permute([0,2,3,1]).reshape(B, -1, nCls)
        maxprob = bar.softmax(dim=-1).reshape(B,-1).max(dim=-1)[0]
        output[:,sIdx] = maxprob
    return output.max(dim=-1)[0].tolist(), output

def ResumeCycle(cfg, currentCycle, fromStartCycle):
    if currentCycle < fromStartCycle:
        return (False, False)
    X_L = np.load(cfg.work_dir + '/X_L_' + str(fromStartCycle) + '.npy')
    X_U = np.load(cfg.work_dir + '/X_U_' + str(fromStartCycle) + '.npy')
    return (X_L, X_U)

def ResumeCycle_WorkDir(work_dir, currentCycle, fromStartCycle):
    if currentCycle < fromStartCycle:
        return (False, False)
    X_L = np.load(work_dir + '/X_L_' + str(fromStartCycle) + '.npy')
    X_U = np.load(work_dir + '/X_U_' + str(fromStartCycle) + '.npy')
    return (X_L, X_U)

def append_dropout(model, rate=0.1):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module)
        if isinstance(module, nn.ReLU):
            new = nn.Sequential(module, nn.Dropout2d(p=rate))
            setattr(model, name, new)

def activate_dropout(model):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module)
        if isinstance(module, nn.Dropout2d):
            module.train()

def FlattenNcls(ten, idx = 0, nCls=20):
    out = ten[0][idx].permute(1,2,0).reshape(-1,nCls)
    if len(ten) > 1:
        for j in range(1,len(ten)):
            tmp = ten[j][idx].permute(1,2,0).reshape(-1,nCls)
            out = torch.cat((out,tmp), dim=0)
    return out