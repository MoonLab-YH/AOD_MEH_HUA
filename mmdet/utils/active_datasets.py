import mmcv
import numpy as np
import pdb
import torch


def get_X_L_ALL(cfg):
    anns = load_ann_list(cfg.data.train.dataset.ann_file)
    X_all = np.arange(len(anns[0]) + len(anns[1]))
    np.random.shuffle(X_all)
    X_L, X_U = X_all.copy(), X_all[len(X_all)-1:]
    X_L.sort()
    return X_L, X_U, X_all, anns

def get_X_L_0_Double(cfg):
    # load dataset anns
    anns = load_ann_list(cfg.data.train.dataset.ann_file)
    # get all indexes
    X_all = np.arange(len(anns[0]) + len(anns[1]))
    # randomly select labeled sets
    np.random.shuffle(X_all)
    X_L, X_U = X_all[:2*cfg.X_L_0_size].copy(), X_all[2*cfg.X_L_0_size:].copy()
    X_L.sort(); X_U.sort()
    return X_L, X_U, X_all, anns

def get_X_L_0(cfg):
    # load dataset anns
    anns = load_ann_list(cfg.data.train.dataset.ann_file)
    # get all indexes
    X_all = np.arange(len(anns[0]) + len(anns[1]))
    # randomly select labeled sets
    np.random.shuffle(X_all)
    X_L, X_U = X_all[:cfg.X_L_0_size].copy(), X_all[cfg.X_L_0_size:].copy()
    X_L.sort(); X_U.sort()
    return X_L, X_U, X_all, anns

def get_X_L_0_prev(cfg):
    # load dataset anns
    anns = load_ann_list(cfg.data.train.dataset.ann_file)
    # get all indexes
    X_all = np.arange(len(anns[0]) + len(anns[1]))
    # randomly select labeled sets
    np.random.shuffle(X_all)
    X_L = X_all[:cfg.X_L_0_size].copy()
    X_U = X_all[cfg.X_L_0_size:cfg.X_L_0_size*2].copy()
    X_L.sort()
    X_U.sort()
    return X_L, X_U, X_all, anns

def create_X_L_file(cfg, X_L, anns, cycle):
    # split labeled set into 2007 and 2012
    X_L = [X_L[X_L < len(anns[0])], X_L[X_L >= len(anns[0])] - len(anns[0])]
    # create labeled ann files
    X_L_path = []
    for ann, X_L_single, year in zip(anns, X_L, ['07', '12']):
        save_folder = cfg.work_dir + '/cycle' + str(cycle)
        mmcv.mkdir_or_exist(save_folder)
        save_path = save_folder + '/trainval_X_L_' + year + '.txt'
        np.savetxt(save_path, ann[X_L_single], fmt='%s')
        X_L_path.append(save_path)
    # update cfg
    cfg.data.train.dataset.ann_file = X_L_path
    cfg.data.train.times = cfg.X_L_repeat
    return cfg


def create_X_U_file(cfg, X_U, anns, cycle):
    # split unlabeled set into 2007 and 2012
    X_U = [X_U[X_U < len(anns[0])], X_U[X_U >= len(anns[0])] - len(anns[0])]
    # create labeled ann files
    X_U_path = []
    for ann, X_U_single, year in zip(anns, X_U, ['07', '12']):
        save_folder = cfg.work_dir + '/cycle' + str(cycle)
        mmcv.mkdir_or_exist(save_folder)
        save_path = save_folder + '/trainval_X_U_' + year + '.txt'
        np.savetxt(save_path, ann[X_U_single], fmt='%s')
        X_U_path.append(save_path)
    # update cfg
    cfg.data.train.dataset.ann_file = X_U_path
    cfg.data.train.times = cfg.X_U_repeat
    return cfg


def load_ann_list(paths):
    anns = []
    for path in paths:
        anns.append(np.loadtxt(path, dtype='str'))
    return anns


def update_X_L2(uncertainty, X_all, X_L, X_S_size):
    uncertainty = uncertainty.cpu().numpy()
    all_X_U = np.array(list(set(X_all)))
    arg = uncertainty.argsort()
    X_L_next = all_X_U[arg[-(X_S_size+len(X_L)):]]
    X_U_next = np.array(list(set(X_all) - set(X_L_next)))
    np.random.shuffle(X_U_next)
    X_L_next.sort()
    X_U_next.sort()
    return X_L_next, X_U_next

def update_X_L(uncertainty, X_all, X_L, X_S_size, **kwargs):
    if torch.is_tensor(uncertainty):
        uncertainty = uncertainty.cpu().numpy()
    all_X_U = np.array(list(set(X_all) - set(X_L)))
    uncertainty_X_U = uncertainty[all_X_U]
    arg = uncertainty_X_U.argsort()
    if 'zeroRate' in kwargs and kwargs['zeroRate']:
        zeros = (uncertainty_X_U == 0).nonzero()[0]
        zeroSize = int(X_S_size * kwargs['zeroRate'])
        nonZeroSize = X_S_size - zeroSize
        if len(zeros) < zeroSize:
            zeroSize = len(zeros)
        if 'useMaxConf' in kwargs and kwargs['useMaxConf'] != 'False':
            maxConf = np.array(kwargs['maxconf'])[all_X_U]
            maxConfArg = maxConf.argsort()
            if kwargs['useMaxConf'] == 'min':
                zeroIdx = maxConfArg[:zeroSize]
            elif kwargs['useMaxConf'] == 'max':
                zeroIdx = maxConfArg[-zeroSize:]
        else:
            zeroIdx = np.random.choice(zeros, zeroSize)
        nonZeroIdx = arg[-nonZeroSize:]
        X_zero = all_X_U[zeroIdx]
        X_nonzero = all_X_U[nonZeroIdx]
        X_S = np.concatenate((X_zero, X_nonzero))
    else:
        X_S = all_X_U[arg[-X_S_size:]]
    X_L_next = np.concatenate((X_L, X_S))
    all_X_U_next = np.array(list(set(X_all) - set(X_L_next)))
    np.random.shuffle(all_X_U_next)
    X_U_next = all_X_U_next[:X_L_next.shape[0]]
    X_L_next.sort()
    X_U_next.sort()
    return X_L_next, X_U_next
#
# def update_X_L(uncertainty, X_all, X_L, X_S_size, **kwargs):
#     # uncertainty = uncertainty.cpu().numpy()
#     all_X_U = np.array(list(set(X_all) - set(X_L)))
#     uncertainty_X_U = uncertainty[all_X_U]
#     arg = uncertainty_X_U.argsort()
#     X_S = all_X_U[arg[-X_S_size:]]
#     X_L_next = np.concatenate((X_L, X_S))
#     all_X_U_next = np.array(list(set(X_all) - set(X_L_next)))
#     np.random.shuffle(all_X_U_next)
#     X_U_next = all_X_U_next[:X_L_next.shape[0]]
#     X_L_next.sort()
#     X_U_next.sort()
#     return X_L_next, X_U_next

def update_X_L_filter(uncertainty, X_all, X_L, X_S_size, ratio = None):
    uncertainty = uncertainty.cpu().numpy()
    all_X_U = np.array(list(set(X_all) - set(X_L)))
    uncertainty_X_U = uncertainty[all_X_U]
    arg = uncertainty_X_U.argsort()
    X_S = all_X_U[arg[-X_S_size:]]
    X_L_next = np.concatenate((X_L, X_S))
    all_X_U_next = np.array(list(set(X_all) - set(X_L_next)))
    np.random.shuffle(all_X_U_next)
    X_U_next = all_X_U_next[:X_L_next.shape[0]]
    X_L_next.sort()
    X_U_next.sort()
    return X_L_next, X_U_next
