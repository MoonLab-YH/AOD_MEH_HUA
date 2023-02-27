# Copyright (c) Open-MMLab. All rights reserved.
import os.path as osp
import platform
import shutil
import time
import warnings
import pdb
import torch
import mmcv
import wandb

from mmdet.utils.functions import *
from mmcv.runner.base_runner import BaseRunner
from mmcv.runner.builder import RUNNERS
from mmcv.runner.checkpoint import save_checkpoint
from mmcv.runner.utils import get_host_info

@RUNNERS.register_module()
class MyEpochBasedRunnerSSD(BaseRunner):
    def run_iter(self, data_batch, train_mode, **kwargs):
        if self.batch_processor is not None:
            outputs = self.batch_processor(self.model, data_batch, train_mode=train_mode, **kwargs)
        elif train_mode:
            loss = self.model.train_step(data_batch, None, **kwargs)
            self.optimizer.zero_grad()
            loss['loss'].backward()
            self.optimizer.step()
            outputs = loss
        else:
            outputs = self.model.val_step(data_batch, self.optimizer, **kwargs)
        if 'log_vars' in outputs:
            self.log_buffer.update(outputs['log_vars'], outputs['num_samples'])
        self.outputs = outputs

    def train(self, data_loader, **kwargs):
        if type(data_loader) != list:
            self.model.train()
            self.mode = 'train'
            self.data_loader = data_loader
            self._max_iters = self._max_epochs * len(data_loader)
            self.call_hook('before_train_epoch')
            for i, data_batch_L in enumerate(data_loader):
                if kwargs['onlyEval']: break
                self._inner_iter = i
                self.call_hook('before_train_iter')
                self.run_iter(data_batch_L, train_mode=True, Labeled = True, Pseudo = False, **kwargs)
                self.call_hook('after_train_iter')
                self._iter += 1
            eval_res = self.call_hook('after_train_epoch')
            self._epoch += 1
        else:
            self.model.train()
            self.mode = 'train'
            self.data_loader = data_loader[0]
            self._max_iters = self._max_epochs * len(data_loader[0])
            self.call_hook('before_train_epoch')
            time.sleep(1)  # Prevent possible deadlock during epoch transition
            unlabeled_data_iter = iter(data_loader[1])
            for i, data_batch_L in enumerate(data_loader[0]):
                if kwargs['onlyEval']: break
                self._inner_iter = i
                self.call_hook('before_train_iter')
                self.run_iter(data_batch_L, train_mode=True, Labeled = True, Pseudo = False, **kwargs)
                data_batch_U = unlabeled_data_iter.next()
                self.run_iter(data_batch_U, train_mode=True, Labeled = False, Pseudo = True, datas=data_batch_U, **kwargs)
                self.call_hook('after_train_iter')
                self._iter += 1
            eval_res = self.call_hook('after_train_epoch')
            self._epoch += 1
        return eval_res
        if eval_res: wandb.log(eval_res)


    @torch.no_grad()
    def val(self, data_loader, **kwargs):
        self.model.eval()
        self.mode = 'val'
        self.data_loader = data_loader
        self.call_hook('before_val_epoch')
        time.sleep(2)  # Prevent possible deadlock during epoch transition
        for i, data_batch in enumerate(self.data_loader):
            self._inner_iter = i
            self.call_hook('before_val_iter')
            self.run_iter(data_batch, train_mode=False)
            self.call_hook('after_val_iter')

        self.call_hook('after_val_epoch')

    def run(self, data_loaders, workflow, max_epochs=None, **kwargs):
        self._max_epochs = max_epochs
        self._max_iters = self._max_epochs * len(data_loaders[0])
        work_dir = self.work_dir if self.work_dir is not None else 'NONE'
        self.logger.info('Start running, host: %s, work_dir: %s', get_host_info(), work_dir)
        self.logger.info('workflow: %s, max: %d epochs', workflow, self._max_epochs)
        self.call_hook('before_run')
        eval_res = None
        while self.epoch < self._max_epochs:
            for i, flow in enumerate(workflow):
                mode, epochs = flow
                epoch_runner = getattr(self, mode)
                for _ in range(epochs):
                    if mode == 'train' and self.epoch >= self._max_epochs:
                        break
                    eval_res = epoch_runner(data_loaders[i], **kwargs)

        time.sleep(1)
        self.call_hook('after_run')
        return eval_res

    def run_SSL(self, data_loaders, workflow, max_epochs = None, **kwargs):
        self._max_epochs = max_epochs
        self._max_iters = self._max_epochs * len(data_loaders[0])
        self.logger.info('workflow: %s, max: %d epochs', workflow, self._max_epochs)
        self.call_hook('before_run')
        if type(data_loaders[0]) != list:
            while self.epoch < self._max_epochs:
                for i, flow in enumerate(workflow):
                    mode, epochs = flow
                    epoch_runner = getattr(self, mode)
                    for _ in range(epochs):
                        if mode == 'train' and self.epoch >= self._max_epochs:
                            break
                        eval_res = epoch_runner(data_loaders[i], **kwargs)
        else:
            data_loaders_u = data_loaders[1]
            data_loaders = data_loaders[0]
            while self.epoch < self._max_epochs:
                for i, flow in enumerate(workflow):
                    mode, epochs = flow
                    epoch_runner = getattr(self, mode)
                    for _ in range(epochs):
                        if mode == 'train' and self.epoch >= self._max_epochs:
                            break
                        eval_res = epoch_runner([data_loaders[i], data_loaders_u[i]], **kwargs)
        time.sleep(0.5)
        self.call_hook('after_run')
        return eval_res

    def save_checkpoint(self,
                        out_dir,
                        filename_tmpl='epoch_{}.pth',
                        save_optimizer=True,
                        meta=None,
                        create_symlink=True):
        if meta is None:
            meta = dict(epoch=self.epoch + 1, iter=self.iter)
        elif isinstance(meta, dict):
            meta.update(epoch=self.epoch + 1, iter=self.iter)
        else:
            raise TypeError(
                f'meta should be a dict or None, but got {type(meta)}')
        if self.meta is not None:
            meta.update(self.meta)

        filename = filename_tmpl.format(self.epoch + 1)
        filepath = osp.join(out_dir, filename)
        optimizer = self.optimizer if save_optimizer else None
        save_checkpoint(self.model, filepath, optimizer=optimizer, meta=meta)
        if create_symlink:
            dst_file = osp.join(out_dir, 'latest.pth')
            if platform.system() != 'Windows':
                mmcv.symlink(filename, dst_file)
            else:
                shutil.copy(filepath, dst_file)
