# ------------------------------------------------------------------------------
# multiview.pose3d.torch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn


class ChannelWiseFC(nn.Module):

    def __init__(self, size):
        super(ChannelWiseFC, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(size, size))
        self.weight.data.uniform_(0, 0.1)

    def forward(self, input):
        N, C, H, W = input.size()
        input_reshape = input.reshape(N * C, H * W)
        output = torch.matmul(input_reshape, self.weight)
        output_reshape = output.reshape(N, C, H, W)
        return output_reshape


class Aggregation(nn.Module):

    def __init__(self, cfg, weights=[0.4, 0.2, 0.2, 0.2]):
        super(Aggregation, self).__init__()
        NUM_NETS = 12
        size = int(cfg.NETWORK.HEATMAP_SIZE[0])
        self.weights = weights
        self.aggre = nn.ModuleList()
        for i in range(NUM_NETS):
            self.aggre.append(ChannelWiseFC(size * size))

    def forward(self, inputs):
        """
        Warped views don't add to the cardinal view.
        """
        fc_idx = 0
        outputs = []
        nviews = len(inputs)
        for i in range(nviews):
            others_indices = [k for k in range(nviews) if k != i]
            warped = torch.zeros_like(inputs[0])
            for others_idx in others_indices:
                fc = self.aggre[fc_idx]
                fc_output = fc(inputs[others_idx])
                warped += fc_output / (nviews - 1)
                fc_idx += 1
            outputs.append(warped)
        return outputs


class MultiViewPose(nn.Module):

    def __init__(self, PoseResNet, Aggre, CFG):
        super(MultiViewPose, self).__init__()
        self.config = CFG
        self.resnet = PoseResNet
        self.aggre_layer = Aggre

    def forward(self, views):
        if isinstance(views, list):
            single_views = []
            low_features = []
            high_features = []
            for view in views:
                heatmaps, pre_f, latter_f = self.resnet(view)
                single_views.append(heatmaps)
                low_features.append(pre_f)
                high_features.append(latter_f)
            multi_views = []
            if self.config.NETWORK.AGGRE:
                multi_views = self.aggre_layer(single_views)
            return single_views, multi_views, low_features, high_features
        else:
            return self.resnet(views)


def get_multiview_pose_net(PoseResNet, CFG):
    if CFG.NETWORK.AGGRE:
        Aggre = Aggregation(CFG)
    else:
        Aggre = None
    model = MultiViewPose(PoseResNet, Aggre, CFG)
    return model
