# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import pickle
import numpy as np
import h5py

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from multiviews.pictorial import rpsm
from multiviews.cameras import camera_to_world_frame
from multiviews.body import HumanBody
import collections
import dataset
import os



def parse_args():
    parser = argparse.ArgumentParser(description='Generate Data For RPSM')
    parser.add_argument(
        '--cfg', help='configuration file name', required=True, type=str)
    parser.add_argument(
        '--heatmap', help='heatmap file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    parser.add_argument(
        '--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument(
        '--dataDir', help='data directory', type=str, default='')
    parser.add_argument(
        '--no-distortion', help='wheter use no distortion data', action='store_true')
    args = parser.parse_args()
    update_dir(args.modelDir, args.logDir, args.dataDir)
    return args


def compute_limb_length(body, pose):
    # gt pose order is same as body order
    limb_length = {}
    skeleton = body.skeleton
    for node in skeleton:
        idx = node['idx']
        children = node['children']

        for child in children:
            length = np.linalg.norm(pose[idx] - pose[child])
            limb_length[(idx, child)] = length
    return limb_length


def generate_data_for_rpsm():
    args = parse_args()
    no_distortion = True if args.no_distortion else False
    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False, pseudo_label_path='', no_distortion=no_distortion)  # multiview
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

    return rpsm_db


def load_rpsm_testdata_all(db):

    assert len(db) % 4 == 0
    all_heatmaps = []
    all_cameras = []
    all_boxes = []
    all_grid_centers = []
    all_limb_lengths = []
    all_gts = []

    body = HumanBody()

    for idx in range(0,len(db),4):
        group_hms = []
        group_cameras = []
        group_boxes = []
        group_gts = []

        for inner_idx in range(idx, idx+4):
            heatmap = db[inner_idx]['heatmap']  # [16, h, w], body order
            group_hms.append(heatmap)

            camera = db[inner_idx]['cam_params']
            group_cameras.append(camera)

            pose_camera = db[inner_idx]['joints_3d_cam']  # [16,3] body order
            pose_world = camera_to_world_frame(pose_camera, camera['R'],
                                               camera['T'])
            group_gts.append(pose_world)

            box = {}
            box['scale'] = np.array(db[inner_idx]['scale'])
            box['center'] = np.array(db[inner_idx]['center'])
            group_boxes.append(box)
        group_hms = np.array(group_hms)

        all_heatmaps.append(group_hms)
        all_cameras.append(group_cameras)
        all_boxes.append(group_boxes)
        all_grid_centers.append(group_gts[0][body.root_idx])
        all_limb_lengths.append(compute_limb_length(body, group_gts[0]))
        all_gts.append(group_gts[0]) 

    return all_cameras, all_heatmaps, all_boxes, all_grid_centers, all_limb_lengths, all_gts


def main():
    # load heatmaps .etc
    rpsm_db = generate_data_for_rpsm()

    # load pairwise constraints
    pairwise_file = os.path.join(config.DATASET.ROOT, 'testdata', 'pairwise_b16.pkl') 
    with open(pairwise_file, 'rb') as f:
        pairwise = pickle.load(f)
        pairwise = pairwise['pairwise_constrain']

    all_cameras, all_hms, all_boxes, all_grid_centers, all_limb_lengths, all_gts = load_rpsm_testdata_all(
        rpsm_db)

    res = []
    for idx, (cameras, hms, boxes, grid_center, limb_length, gt) in enumerate(zip(all_cameras,
        all_hms, all_boxes, all_grid_centers,all_limb_lengths, all_gts)):

        pose = rpsm(cameras, hms, boxes, grid_center, limb_length, pairwise, config)
        # print('GroundTruth Pose: ', gt)
        # print('Recovered Pose by RPSM: ', pose)
        mpjpe = np.mean(np.sqrt(np.sum((pose - gt)**2, axis=1)))
        # print('MPJPE: ', mpjpe)
        res.append(mpjpe)
        if idx % 500 == 0:
            print('%d:%.2f' % (idx, mpjpe))
    print('MPJPE: ', np.mean(res))


if __name__ == '__main__':
    main()
