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
import numpy as np
import pickle
import h5py

import _init_paths
import dataset
from core.config import config
from core.config import update_config
from core.config import update_dir
from multiviews.cameras import camera_to_world_frame, project_pose
from multiviews.triangulate import triangulate_poses, ransac


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test Recursive Pictorial Structure Model')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        '--heatmap', help='heatmap file name', default='', type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    parser.add_argument(
        '--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument(
        '--dataDir', help='data directory', type=str, default='')
    parser.add_argument(
        '--no-distortion', help='wheter use no distortion data', action='store_true')
    parser.add_argument(
        '--inliers', help='min support num of inliers', type=int)
    parser.add_argument(
        '--reproj-thre', help='reprojection threshold to accept inliers', type=int)
    args = parser.parse_args()
    update_dir(args.modelDir, args.logDir, args.dataDir)
    return args


def reset_config(config, args):
    if args.inliers:
        config.PSEUDO_LABEL.NUM_INLIERS = args.inliers
    if args.reproj_thre:
        config.PSEUDO_LABEL.REPROJ_THRE = args.reproj_thre
    return


def main():
    args = parse_args()
    reset_config(config, args)
    no_distortion = True if args.no_distortion else False
    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False, pseudo_label_path='', no_distortion=no_distortion)
    grouping = test_dataset.grouping

    if args.heatmap == '':
        flag_test_gt = True
    else:
        flag_test_gt = False
    
    if not flag_test_gt:
        test_data_file = args.heatmap
        db = h5py.File(test_data_file, 'r')
        pred2d = np.array(db['locations'])[:, :, :2]  # [8860, 16, 2]

    if flag_test_gt:
        pred2d = []
    cameras = []
    gt3d = []
    for items in grouping:
        for item in items:
            cam = test_dataset.db[item]['camera']
            cameras.append(cam)
            if flag_test_gt:
                pred2d.append(test_dataset.db[item]['joints_2d'])  # [N, k_union, 2], already mapped
        gt = test_dataset.db[items[-1]]['joints_3d']
        gt3d.append(camera_to_world_frame(gt, cameras[-1]['R'], cameras[-1]['T']))

    if flag_test_gt:
        pred2d = np.array(pred2d)
    gt3d = np.array(gt3d)

    joints_vis = np.ones(pred2d.shape[:2])
    joints_vis = ransac(camera_params=cameras, poses2d=pred2d, joints_vis=joints_vis, config=config)  # [N, 16, 3]
    pred3d = triangulate_poses(cameras, pred2d, joints_vis=joints_vis, no_distortion=no_distortion)  # [N, 16, 3]
    assert len(gt3d) == len(pred3d)
    
    u2a = test_dataset.u2a_mapping
    u2a = {k:v  for k, v in u2a.items() if v != '*'}
    sorted_u2a = sorted(u2a.items(), key=lambda x: x[0])
    u = np.array([mapping[0] for mapping in sorted_u2a])
    a = np.array([mapping[1] for mapping in sorted_u2a])

    if flag_test_gt:
        compatible_pred = pred3d[:, u, :]  # [N, 16, 3]
    else:
        compatible_pred = pred3d
    # compatible_pred = pred3d[:, a, :]  # for multivew mixed resutls

    compatible_gt = gt3d[:, a, :]  # [N, 16, 3]

    norm = np.linalg.norm(compatible_pred - compatible_gt, axis=2)  # [N, 16]
    print('Mean Error:', np.mean(norm))
    print('Std Error:', np.std(norm))
    print('Max Error:', np.amax(norm))
    print('Larger than Mean+Std Error: {:.1%}'.format(np.sum(norm>np.mean(norm)+np.std(norm))/norm.size))
    thre_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 500, 1000, 2000, 5000, 10000]
    print('| ' +  ' | '.join(str(thre) for thre in thre_list) + ' |')
    print(''.join('| {:.1%} '.format(np.sum(norm<thre)/norm.size) for thre in thre_list) + '|')


if __name__ == '__main__':
    main()
