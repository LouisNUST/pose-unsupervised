# ------------------------------------------------------------------------------
# multiview.pose3d.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil
import pickle

import h5py
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint, load_checkpoint
from utils.utils import create_logger
import dataset
import models
from multiviews.body import HumanBody
import collections


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Data For RPSM')
    parser.add_argument(
        '--cfg', help='configuration file name', required=True, type=str)
    parser.add_argument(
        '--heatmap', help='heatmap file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def main():
    args = parse_args()
    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False)  # multiview
    grouping = test_dataset.grouping
    h5set = h5py.File(args.heatmap)
    heatmaps = h5set['heatmaps']  # [N, 16, h, w]
    stored_joint_orders = h5set['joint_names_order']  # u of u2a, order indices in union(mpii) datasets

    union_joints = {0:'rank', 1:'rkne', 2:'rhip', 3:'lhip', 4:'lkne', 5:'lank', 
            6:'root', 7:'thorax', 8:'upper neck', 9:'head top', 10:'rwri',
            11:'relb', 12:'rsho', 13:'lsho', 14:'lelb', 15:'lwri'
        }
    stored_joint_names = [union_joints[idx] for idx in stored_joint_orders]

    # mapping stored order to predefined HumanBody order (mpii naming)
    body = HumanBody()
    assert heatmaps.shape[1] == len(body.skeleton)
    store2body_indices = [stored_joint_names.index(joint_dict['name']) for joint_dict in body.skeleton]
    heatmaps = heatmaps[:, store2body_indices, :, :]

    # mapping gt 3d pose (h36m order) to HumanBody order (mpii naming)
    h36m_joints = [(0, 'root'),
                   (1, 'rhip'),
                   (2, 'rkne'),
                   (3, 'rank'),
                   (4, 'lhip'),
                   (5, 'lkne'),
                   (6, 'lank'),
                   (7, 'belly'),
                   (8, 'neck'),
                   (9, 'nose'),
                   (10, 'head'),
                   (11, 'lsho'),
                   (12, 'lelb'),
                   (13, 'lwri'),
                   (14, 'rsho'),
                   (15, 'relb'),
                   (16, 'rwri')]
    h36m_joints = collections.OrderedDict(h36m_joints)
    # replace with mpii name
    h36m_joints[8] = 'thorax'
    h36m_joints[9] = 'upper neck'
    h36m_joints[10] = 'head top'
    h36m_names = list(h36m_joints.values())
    h36m2body_indices = [h36m_names.index(joint_dict['name']) for joint_dict in body.skeleton]

    rpsm_db = []
    cnt = 0
    for group_idx, items in enumerate(grouping):
        for idx in items:
            datum = test_dataset.db[idx]
            hm = heatmaps[cnt]
            cnt += 1

            rpsm_datum = {
                'heatmap': hm,
                'cam_params': datum['camera'],
                'joints_3d_cam': datum['joints_3d'][h36m2body_indices, :], # body order.
                'scale': datum['scale'],
                'center': datum['center']
            }
            rpsm_db.append(rpsm_datum)
        if group_idx % 1000 == 0:
            print(group_idx)

    with open('data/testdata/rpsm_testdata_b16.pkl', 'wb') as f:
        pickle.dump(rpsm_db, f)


if __name__ == '__main__':
    main()
