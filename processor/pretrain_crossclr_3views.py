#!/usr/bin/env python
# pylint: disable=W0201
import sys
import argparse
import yaml
import math
import numpy as np

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# torchlight
import torchlight
from torchlight import str2bool
from torchlight import DictAction
from torchlight import import_class

from .processor import Processor
from .pretrain import PT_Processor


class CrosSCLR_3views_Processor(PT_Processor):
    """
        Processor for 3view-CrosSCLR Pretraining.
    """

    def train(self, epoch):
        self.model.train()
        self.adjust_lr()
        loader = self.data_loader['train']
        loss_value = []
        loss_motion_value = []
        loss_bone_value = []

        for [data1, data2], label in loader:
            self.global_step += 1
            # get data
            data1 = data1.float().to(self.dev, non_blocking=True)
            data2 = data2.float().to(self.dev, non_blocking=True)
            label = label.long().to(self.dev, non_blocking=True)

            # forward
            if epoch <= self.arg.cross_epoch:
                output, output_motion, output_bone, target = self.model(data1, data2)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output.size(0))
                else:
                    self.model.update_ptr(output.size(0))
                loss = self.loss(output, target)
                loss_motion = self.loss(output_motion, target)
                loss_bone = self.loss(output_bone, target)

                self.iter_info['loss'] = loss.data.item()
                self.iter_info['loss_motion'] = loss_motion.data.item()
                self.iter_info['loss_bone'] = loss_bone.data.item()
                loss_value.append(self.iter_info['loss'])
                loss_motion_value.append(self.iter_info['loss_motion'])
                loss_bone_value.append(self.iter_info['loss_bone'])
                loss = loss + loss_motion + loss_bone
            else:
                output_jm, output_jb, output_mj, output_mb, output_bj, output_bm, mask_jm, mask_jb, mask_mj, mask_mb, mask_bj, mask_bm = self.model(data1, data2, cross=True, topk=self.arg.topk, context=self.arg.context)
                if hasattr(self.model, 'module'):
                    self.model.module.update_ptr(output_jm.size(0))
                else:
                    self.model.update_ptr(output_jm.size(0))
                loss_jm = - (F.log_softmax(output_jm, dim=1) * mask_jm).sum(1) / mask_jm.sum(1)
                loss_jb = - (F.log_softmax(output_jb, dim=1) * mask_jb).sum(1) / mask_jb.sum(1)
                loss_mj = - (F.log_softmax(output_mj, dim=1) * mask_mj).sum(1) / mask_mj.sum(1)
                loss_mb = - (F.log_softmax(output_mb, dim=1) * mask_mb).sum(1) / mask_mb.sum(1)
                loss_bj = - (F.log_softmax(output_bj, dim=1) * mask_bj).sum(1) / mask_bj.sum(1)
                loss_bm = - (F.log_softmax(output_bm, dim=1) * mask_bm).sum(1) / mask_bm.sum(1)
                loss = (loss_jm + loss_jb) / 2.
                loss_motion = (loss_mj + loss_mb) / 2.
                loss_bone = (loss_bj + loss_bm) / 2.
                loss = loss.mean()
                loss_motion = loss_motion.mean()
                loss_bone = loss_bone.mean()

                self.iter_info['loss'] = loss.data.item()
                self.iter_info['loss_motion'] = loss_motion.data.item()
                self.iter_info['loss_bone'] = loss_bone.data.item()
                loss_value.append(self.iter_info['loss'])
                loss_motion_value.append(self.iter_info['loss_motion'])
                loss_bone_value.append(self.iter_info['loss_bone'])
                loss = loss + loss_motion + loss_bone

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # statistics
            self.iter_info['lr'] = '{:.6f}'.format(self.lr)
            self.show_iter_info()
            self.meta_info['iter'] += 1
            self.train_log_writer(epoch)
            self.train_writer.add_scalar('batch_loss_motion', self.iter_info['loss_motion'], self.global_step)
            self.train_writer.add_scalar('batch_loss_bone', self.iter_info['loss_bone'], self.global_step)

        self.epoch_info['train_mean_loss']= np.mean(loss_value)
        self.epoch_info['train_mean_loss_motion']= np.mean(loss_motion_value)
        self.epoch_info['train_mean_loss_bone']= np.mean(loss_bone_value)
        self.train_writer.add_scalar('loss', self.epoch_info['train_mean_loss'], epoch)
        self.train_writer.add_scalar('loss_motion', self.epoch_info['train_mean_loss_motion'], epoch)
        self.train_writer.add_scalar('loss_bone', self.epoch_info['train_mean_loss_bone'], epoch)
        self.show_epoch_info()

    @staticmethod
    def get_parser(add_help=False):

        # parameter priority: command line > config > default
        parent_parser = Processor.get_parser(add_help=False)
        parser = argparse.ArgumentParser(
            add_help=add_help,
            parents=[parent_parser],
            description='Spatial Temporal Graph Convolution Network')

        # region arguments yapf: disable
        parser.add_argument('--base_lr', type=float, default=0.01, help='initial learning rate')
        parser.add_argument('--step', type=int, default=[], nargs='+', help='the epoch where optimizer reduce the learning rate')
        parser.add_argument('--optimizer', default='SGD', help='type of optimizer')
        parser.add_argument('--nesterov', type=str2bool, default=True, help='use nesterov or not')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay for optimizer')
        parser.add_argument('--view', type=str, default='joint', help='the view of input')
        parser.add_argument('--cross_epoch', type=int, default=1e6, help='the starting epoch of cross-view training')
        parser.add_argument('--context', type=str2bool, default=True, help='using context knowledge')
        parser.add_argument('--topk', type=int, default=1, help='topk samples in cross-view training')
        # endregion yapf: enable

        return parser
