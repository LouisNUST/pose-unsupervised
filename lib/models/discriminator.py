# ------------------------------------------------------------------------------
# multiview.pose3d.torch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import numpy as np

import torch
import torch.nn as nn

class Permute(torch.nn.Module):
    def __init__(self, *perm):
        super().__init__()
        self.perm = perm

    def forward(self, input):
        return input.permute(*self.perm)


class MI1x1ConvNet(nn.Module):
    def __init__(self, n_input, n_units):
        """
        Args:
            n_input: Number of input units.
            n_units: Number of output units.
        """
        super(MI1x1ConvNet, self).__init__()
        self.block_nonlinear = nn.Sequential(
            nn.Conv2d(n_input, n_units, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(n_units),
            nn.ReLU(),
            nn.Conv2d(n_units, n_units, kernel_size=1, stride=1, padding=0, bias=True),
        )

        self.block_ln = nn.Sequential(
            Permute(0, 2, 3, 1),
            nn.LayerNorm(n_units),
            Permute(0, 3, 1, 2)
        )

        self.linear_shortcut = nn.Conv2d(n_input, n_units, kernel_size=1,
                                         stride=1, padding=0, bias=False)
        # initialize shortcut to be like identity (if possible)
        if n_units >= n_input:
            eye_mask = np.zeros((n_units, n_input, 1, 1), dtype=np.uint8)
            for i in range(n_input):
                eye_mask[i, i, 0, 0] = 1
            self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
            self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)

    def forward(self, x):
        """
        Output: [N, C, H, W]
        """
        h = self.block_ln(self.block_nonlinear(x) + self.linear_shortcut(x))
        return h


class MIFCNet(nn.Module):
    """Simple custom network for computing MI.

    """
    def __init__(self, n_input, n_units, bn =False):
        super(MIFCNet, self).__init__()
        self.bn = bn
        assert(n_units >= n_input)
        self.linear_shortcut = nn.Linear(n_input, n_units)
        self.block_nonlinear = nn.Sequential(
            nn.Linear(n_input, n_units, bias=False),
            nn.BatchNorm1d(n_units),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(n_units, n_units)
        )

        # initialize the initial projection to a sort of noisy copy
        eye_mask = np.zeros((n_units, n_input), dtype=np.uint8)
        for i in range(n_input):
            eye_mask[i, i] = 1

        self.linear_shortcut.weight.data.uniform_(-0.01, 0.01)
        self.linear_shortcut.weight.data.masked_fill_(torch.tensor(eye_mask), 1.)

        self.block_ln = nn.LayerNorm(n_units)

    def forward(self, x):
        h = self.block_nonlinear(x) + self.linear_shortcut(x)

        if self.bn:
            h = self.block_ln(h)
        return h


class GlobalDiscriminator(nn.Module):
    def __init__(self, cfg):
        super(GlobalDiscriminator, self).__init__()


    def forward(self, x):
        return


class LocalDiscriminator(nn.Module):
    def __init__(self, cfg):
        super(LocalDiscriminator, self).__init__()
        c_low = cfg.LOCAL_DISCRIMINATOR.LOW_FEATURES_CHANNELS
        c_high = cfg.LOCAL_DISCRIMINATOR.HIGH_FEATURES_CHANNELS
        c_out = cfg.LOCAL_DISCRIMINATOR.OUTPUT_CHANNELS
        # self.use_low_features_preprocess = cfg.LOSS.USE_LOW_FEATURES_PREPROCESS
        self.low_features_net = MI1x1ConvNet(c_low, c_out)
        self.high_features_net = MI1x1ConvNet(c_high, c_out)
        # self.low_features_pre_process = nn.Sequential(
        #     nn.Conv2d(c_low, c_low,
        #      kernel_size=3, stride=1, padding=0, bias=False),
        #     nn.BatchNorm2d(c_low),
        #     nn.ReLU()
        #     )

    def forward(self, low_features, high_features):
        """
        low_features: [N, C, H, W] or [N, C, L] or [C, L]
        high_features: [N, C, H, W] or [N, C, L] or [C, L], same shape as low_features
        Return: [N, H, W] or [N, L] or [C, L], same shape as input
        """
        org_dim = 4
        if low_features.dim() == 3:
            low_features = low_features[..., None]  # [N, C, L, 1]
            high_features = high_features[..., None]
            org_dim = 3
        elif low_features.dim() == 2:
            low_features = low_features[None, :, :, None]  # [1, C, L, 1]
            high_features = high_features[None, :, :, None]  # [1, C, L, 1]
            org_dim = 2
        else:
            assert low_features.dim() == 3, 'unsupported feature dim {}'.format(low_features.dim())

        # if self.use_low_features_preprocess:
        #     low_features = self.low_features_pre_process(low_features)
        low_features_embd = self.low_features_net(low_features)  # [N, C, H, W]
        high_features_embd = self.high_features_net(high_features)  # [N, C, H, W]
        scores = torch.sum(low_features_embd * high_features_embd, dim=1)  # [N, H, W]
        if org_dim == 3:
            scores = scores.squeeze(-1)
        elif org_dim == 2:
            scores = scores.squeeze(0).squeeze(-1)
        return scores


class DomainDiscriminator(nn.Module):
    def __init__(self, cfg):
        super(DomainDiscriminator, self).__init__()
        # input is [N, 2048, 8, 8]
        self.input_c = cfg.DOMAIN_DISCRIMINATOR.FEATURES_CHANNELS
        self.main = nn.Sequential(
            nn.Conv2d(self.input_c, 256, 1, 1, 0, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # [N, 256, 8, 8]
            nn.Conv2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # [N, 128, 4, 4]
            nn.Conv2d(128, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
            )

    def forward(self, input):
        return self.main(input)


class ViewDiscriminator(nn.Module):
    def __init__(self, cfg):
        super(ViewDiscriminator, self).__init__()
        # two random variables [N, 16*K, 64, 64] as input
        view1_num = cfg.VIEW_DISCRIMINATOR.VIEW_ONE_NUM
        view2_num = cfg.VIEW_DISCRIMINATOR.VIEW_TWO_NUM
        c_out = cfg.VIEW_DISCRIMINATOR.OUTPUT_CHANNELS
        self.num_joints = cfg.NETWORK.NUM_JOINTS
        self.view1_net = MIFCNet(view1_num*self.num_joints*2, c_out, bn=True)
        self.view2_net = MIFCNet(view2_num*self.num_joints*2, c_out, bn=True)

    def forward(self, joints_2d_view1, joints_2d_view2):
        """
        # [N, 16, 2]
        # [N, 3*16, 2]
        """
        batch_size = joints_2d_view1.shape[0]
        joints_2d_view1 = joints_2d_view1.view(batch_size, -1)  # [N, 16*2]
        joints_2d_view2 = joints_2d_view2.view(batch_size, -1)  # [N, 3*16*2]
        scores_view1 = self.view1_net(joints_2d_view1)  # [N, c_out]
        scores_view2 = self.view2_net(joints_2d_view2)  # [N, c_out]
        return scores_view1, scores_view2


class JointsDiscriminator(nn.Module):
    def __init__(self, cfg):
        super(JointsDiscriminator, self).__init__()
        # two random variables [N, 16*K, 64, 64] as input
        var1_num = cfg.JOINTS_DISCRIMINATOR.VAR_ONE_NUM
        var2_num = cfg.JOINTS_DISCRIMINATOR.VAR_TWO_NUM
        c_out = cfg.JOINTS_DISCRIMINATOR.OUTPUT_CHANNELS
        self.var1_net = MIFCNet(var1_num*2, c_out, bn=True)
        self.var2_net = MIFCNet(var2_num*2, c_out, bn=True)

    def forward(self, joints_2d_var1, joints_2d_var2):
        """
        # [N, var1_num, 2]
        # [N, var2_num, 2]
        """
        batch_size = joints_2d_var1.shape[0]
        joints_2d_var1 = joints_2d_var1.view(batch_size, -1)  # [N, var1_num*2]
        joints_2d_var2 = joints_2d_var2.view(batch_size, -1)  # [N, var2_num*2]
        scores_var1 = self.var1_net(joints_2d_var1)  # [N, c_out]
        scores_var2 = self.var2_net(joints_2d_var2)  # [N, c_out]
        return scores_var1, scores_var2


class HeatmapDiscriminator(nn.Module):
    def __init__(self, cfg):
        super(HeatmapDiscriminator, self).__init__()
        c_in = cfg.HEATMAP_DISCRIMINATOR.INPUT_CHANNELS
        c_m = cfg.HEATMAP_DISCRIMINATOR.INTER_CHANNELS

        self.block_nonlinear = nn.Sequential(
            nn.Linear(c_in, c_m, bias=False),
            nn.BatchNorm1d(c_m),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(c_m, c_m//4),
            nn.BatchNorm1d(c_m//4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(c_m//4, 1)
        )

    def forward(self, embd):
        return self.block_nonlinear(embd)