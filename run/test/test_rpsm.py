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

import _init_paths
from core.config import config
from core.config import update_config
from multiviews.pictorial import rpsm
from multiviews.cameras import camera_to_world_frame
from multiviews.body import HumanBody


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test Recursive Pictorial Structure Model')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
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


def load_rpsm_testdata(testdata):
    with open(testdata, 'rb') as f:
        db = pickle.load(f)

    heatmaps = []
    cameras = []
    boxes = []
    poses = []
    for i in range(4):
        heatmap = db[i]['heatmap']  # [k, h, w]
        heatmaps.append(heatmap)

        camera = db[i]['cam_params']
        cameras.append(camera)

        pose_camera = db[i]['joints_3d_cam']
        pose_world = camera_to_world_frame(pose_camera, camera['R'],
                                           camera['T'])
        poses.append(pose_world)

        box = {}
        box['scale'] = np.array(db[i]['scale'])
        box['center'] = np.array(db[i]['center'])
        boxes.append(box)
    hms = np.array(heatmaps)

    grid_center = poses[0][0]
    body = HumanBody()
    limb_length = compute_limb_length(body, poses[0])

    return cameras, hms, boxes, grid_center, limb_length, poses[0]


def load_rpsm_testdata_all(testdata):
    with open(testdata, 'rb') as f:
        db = pickle.load(f)

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
    parse_args()
    test_data_file = 'data/testdata/rpsm_testdata_b16.pkl'
    pairwise_file = 'data/testdata/pairwise_b16.pkl'
    with open(pairwise_file, 'rb') as f:
        pairwise = pickle.load(f)
        pairwise = pairwise['pairwise_constrain']
    all_cameras, all_hms, all_boxes, all_grid_centers, all_limb_lengths, all_gts = load_rpsm_testdata_all(
        test_data_file)

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
