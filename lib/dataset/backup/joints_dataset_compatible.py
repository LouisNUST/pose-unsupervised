# ------------------------------------------------------------------------------
# multiview.pose3d.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import cv2
import copy
import random
import numpy as np
import os.path as osp

import torch
from torch.utils.data import Dataset

from utils.transforms import get_affine_transform
from utils.transforms import affine_transform
from utils.transforms import fliplr_joints


class JointsDatasetCompatible(Dataset):

    def __init__(self, cfg, subset, is_train, transform=None):
        self.is_train = is_train
        self.subset = subset

        self.root = cfg.DATASET.ROOT
        self.data_format = cfg.DATASET.DATA_FORMAT
        self.image_size = cfg.NETWORK.IMAGE_SIZE
        self.heatmap_size = cfg.NETWORK.HEATMAP_SIZE
        self.sigma = cfg.NETWORK.SIGMA
        self.transform = transform
        self.db = []

        self.num_joints = 16
        # mpii naming
        self.union_joints = {
            0: 'rank',
            1: 'rkne',
            2: 'rhip',
            3: 'lhip',
            4: 'lkne',
            5: 'lank',
            6: 'root',
            7: 'thorax',
            8: 'upper neck',
            9: 'head top',
            10: 'rwri',
            11: 'relb',
            12: 'rsho',
            13: 'lsho',
            14: 'lelb',
            15: 'lwri'
        }
        self.actual_joints = {}
        self.u2a_mapping = {}

    def do_mapping(self):
        union_indices = [k for k, v in self.u2a_mapping.items() if v != '*']
        actual_indices = [v for v in self.u2a_mapping.values() if v != '*']
        for item in self.db:
            joints = item['joints_2d']
            joints_vis = item['joints_vis']

            joints_union = np.zeros(shape=(self.num_joints, 2))
            joints_union_vis = np.zeros(shape=(self.num_joints, 3))

            joints_union[union_indices] = joints[actual_indices]
            joints_union_vis[union_indices] = joints_vis[actual_indices]

            item['joints_2d'] = joints_union
            item['joints_vis'] = joints_union_vis

    def _get_db(self):
        raise NotImplementedError

    def evaluate(self, cfg, preds, output_dir, *args, **kwargs):
        raise NotImplementedError

    def __len__(self,):
        return len(self.db)

    def __getitem__(self, idx):
        db_rec = copy.deepcopy(self.db[idx])

        image_dir = 'images.zip@' if self.data_format == 'zip' else ''
        image_file = osp.join(self.root, db_rec['source'], image_dir, 'images',
                              db_rec['image'])
        if self.data_format == 'zip':
            from utils import zipreader
            data_numpy = zipreader.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
        else:
            data_numpy = cv2.imread(
                image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

        joints = db_rec['joints_2d'].copy()  # [union_joints, 2]
        joints_vis = db_rec['joints_vis'].copy()[:, :2]  # [union_joints, 2]
        assert len(joints) == self.num_joints
        assert len(joints_vis) == self.num_joints

        # crop and scale according to ground truth
        center = np.array(db_rec['center']).copy()
        scale = np.array(db_rec['scale']).copy()
        rotation = 0

        if self.is_train and db_rec['source'] == 'mpii':
            sf = self.mpii_scale_factor
            rf = self.mpii_rotation_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
            rotation = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) \
                if random.random() <= 0.6 else 0

            if self.mpii_flip and random.random() <= 0.5:
                data_numpy = data_numpy[:, ::-1, :]
                joints, joints_vis = fliplr_joints(
                    joints, joints_vis, data_numpy.shape[1], self.mpii_flip_pairs)
                center[0] = data_numpy.shape[1] - center[0] - 1

        trans = get_affine_transform(center, scale, rotation, self.image_size)
        input = cv2.warpAffine(
            data_numpy,
            trans, (int(self.image_size[0]), int(self.image_size[1])),
            flags=cv2.INTER_LINEAR)

        if self.transform:
            input = self.transform(input)

        visible_joints = joints_vis[:, 0] > 0
        if np.any(visible_joints):
            joints[visible_joints, :2] = affine_transform(joints[visible_joints, :2], trans)
            # zero_indices = np.any(
            #         np.concatenate((joints[:, :2]<0, 
            #         joints[:, [0]] >= self.image_size[0],
            #         joints[:, [1]] >= self.image_size[1]), 
            #         axis=1), 
            #         axis=1)
            # joints_vis[zero_indices, :] = 0

        target, target_weight = self.generate_target(joints, joints_vis, db_rec['source'])

        target = torch.from_numpy(target)
        target_weight = torch.from_numpy(target_weight)

        meta = {
            'scale': scale,
            'center': center,
            'rotation': rotation,
            'joints_2d': db_rec['joints_2d'],
            'joints_2d_transformed': joints,
            'joints_vis': joints_vis,
            'source': db_rec['source']
        }
        return input, target, target_weight, meta

    def generate_target(self, joints_2d, joints_vis, source):
        target, weight = self.generate_heatmap(joints_2d, joints_vis, source)
        return target, weight

    def generate_heatmap(self, joints, joints_vis, source):
        '''
        :param joints:  [num_joints, 2]
        :param joints_vis: [num_joints, 3]
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        target = np.zeros(
            (self.num_joints, self.heatmap_size[1], self.heatmap_size[0]),
            dtype=np.float32)

        target_weight = np.ones((self.num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_vis[:, 0]

        tmp_size = self.sigma * 3

        for joint_id in range(self.num_joints):
            feat_stride = self.image_size / self.heatmap_size
            mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
            mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if ul[0] >= self.heatmap_size[0] or ul[1] >= self.heatmap_size[1] \
                    or br[0] < 0 or br[1] < 0:
                target_weight[joint_id] = 0
                continue

            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            g = np.exp(-((x - x0)**2 + (y - y0)**2) / (2 * self.sigma**2))

            g_x = max(0, -ul[0]), min(br[0], self.heatmap_size[0]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self.heatmap_size[1]) - ul[1]
            img_x = max(0, ul[0]), min(br[0], self.heatmap_size[0])
            img_y = max(0, ul[1]), min(br[1], self.heatmap_size[1])

            v = target_weight[joint_id]
            if v > 0.5:
                target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                    g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        if source == 'h36m':
            target_weight = np.zeros((self.num_joints, 1), dtype=np.float32)

        return target, target_weight
