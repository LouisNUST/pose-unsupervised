# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import os.path as osp
import numpy as np
import json_tricks as json
import collections
from scipy.io import loadmat

from dataset.joints_dataset_compatible import JointsDatasetCompatible
from utils.vis import save_all_preds

class MPIIDatasetCompatible(JointsDatasetCompatible):

    def __init__(self, cfg, image_set, is_train, transform=None, pseudo_label_path='', no_distortion=False):
        super().__init__(cfg, image_set, is_train, transform)
        self.actual_joints = {
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
        self.pseudo_label = False
        self.no_distortion = False
        self.db = self._get_db()

        self.u2a_mapping = self.get_mapping()
        super().do_mapping()

        self.grouping = self.get_group(self.db)
        self.group_size = len(self.grouping)
        self.dataset_type = 'mpii'

        # Data Augmentation
        self.aug_param_dict = {'mpii':{'scale_factor': cfg.DATASET.MPII_SCALE_FACTOR,
                                       'rotation_factor': cfg.DATASET.MPII_ROT_FACTOR,
                                       'flip': cfg.DATASET.MPII_FLIP}}

    def get_mapping(self):
        union_keys = list(self.union_joints.keys())
        union_values = list(self.union_joints.values())

        mapping = {k: '*' for k in union_keys}
        for k, v in self.actual_joints.items():
            idx = union_values.index(v)
            key = union_keys[idx]
            mapping[key] = k
        return mapping

    def _get_db(self):
        file_name = os.path.join(self.root, 'mpii', 'annot',
                                 self.subset + '.json')
        with open(file_name) as anno_file:
            anno = json.load(anno_file)

        gt_db = []
        for a in anno:
            image_name = a['image']

            c = np.array(a['center'], dtype=np.float)
            s = np.array([a['scale'], a['scale']], dtype=np.float)

            # Adjust center/scale slightly to avoid cropping limbs
            if c[0] != -1:
                c[1] = c[1] + 15 * s[1]
                s = s * 1.25

            # MPII uses matlab format, index is based 1,
            # we should first convert to 0-based index
            c = c - 1

            joints_vis = np.zeros((16, 3), dtype=np.float)
            if self.subset != 'test':
                joints = np.array(a['joints'])
                joints[:, 0:2] = joints[:, 0:2] - 1
                vis = np.array(a['joints_vis'])

                joints_vis[:, 0] = vis[:]
                joints_vis[:, 1] = vis[:]

            gt_db.append({
                'image': image_name,
                'center': c,
                'scale': s,
                'joints_2d': joints,
                'joints_3d': np.zeros((16, 3)),
                'joints_vis': joints_vis,
                'source': 'mpii'
            })

        return gt_db

    def get_group(self, db):
        mpii_grouping = []
        mpii_length = len(db)
        for i in range(mpii_length // 4):
            mini_group = []
            for j in range(4):
                index = i * 4 + j
                mini_group.append(index)
            mpii_grouping.append(mini_group)
        return mpii_grouping

    def __getitem__(self, idx):
        input, target, weight, meta = [], [], [], []
        items = self.grouping[idx]
        for item in items:
            i, t, w, m = super().__getitem__(item)
            input.append(i)
            target.append(t)
            weight.append(w)
            meta.append(m)
        return input, target, weight, meta

    def __len__(self):
        return self.group_size

    def evaluate(self, pred, output_dir=None):
        pred = pred.copy()

        sc_bias = 0.6
        threshold = 0.5

        gt_file = os.path.join(self.root, 'mpii', 'annot', 'gt_%s.mat' % self.subset)
        gt_dict = loadmat(gt_file)
        headboxes_src = gt_dict['headboxes_src']
        headsizes = headboxes_src[1, :, :] - headboxes_src[0, :, :]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes = headsizes * sc_bias  # [2958]

        u2a = self.u2a_mapping
        u2a = {k:v  for k, v in u2a.items() if v != '*'}
        # sorted by union index
        sorted_u2a = sorted(u2a.items(), key=lambda x: x[0])
        u = np.array([mapping[0] for mapping in sorted_u2a])
        a = np.array([mapping[1] for mapping in sorted_u2a])

        gt = []
        joints_vis = []
        scale = []
        for items in self.grouping:
            for item in items:
                gt.append(self.db[item]['joints_2d'])
                joints_vis.append(self.db[item]['joints_vis'])
                scale.append(headsizes[item])
        gt = np.array(gt)[:, u, :2]
        pred = pred[:, :, :2]
        joints_vis = np.array(joints_vis)[:, u, 0]  # [N, njoints]
        scale = np.array(scale)[:, np.newaxis]  # [N, 1]
        print('selected images', len(scale))

        distance = np.sqrt(np.sum((gt - pred)**2, axis=2))  # [N, njoints]
        scaled_distance = np.divide(distance, scale)  # [N, njoints]
        detected = (scaled_distance <= threshold)

        if output_dir is not None:
            image_names = []
            for items in self.grouping:
                for item in items:
                    image_names.append(self.db[item]['image'])
            save_all_preds(gt, pred, detected, image_names, 'mpii', output_dir)

        considered_detected = detected * joints_vis  # [N, njoints]
        joint_detection_rate = np.sum(considered_detected, axis=0) / np.sum(joints_vis, axis=0).astype(np.float32)

        name_values = collections.OrderedDict()
        joint_names = self.actual_joints
        for i in range(len(u2a)):
            name_values[joint_names[a[i]]] = joint_detection_rate[i]
        joint_ratio = np.sum(joints_vis, axis=0) / np.sum(joints_vis).astype(np.float32)
        name_values['mean'] = np.sum(joint_ratio*joint_detection_rate)
        return name_values, name_values['mean']
