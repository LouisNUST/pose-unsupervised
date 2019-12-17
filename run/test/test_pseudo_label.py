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
import collections
import logging
import pprint
from pathlib import Path
import os, time

import _init_paths
import dataset
from core.config import config
from core.config import update_config
from core.config import update_dir
from multiviews.cameras import camera_to_world_frame, project_pose
from multiviews.triangulate import triangulate_poses, ransac, reproject_poses


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test Pseudo Labels')
    # Required Param
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        '--heatmap', help='heatmap file name', default='', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    parser.add_argument(
        '--confidence-thre', help='max pixel error to accept inliers', type=float)
    parser.add_argument(
        '--ransac', help='use ransac', action='store_true')
    parser.add_argument(
        '--inliers', help='min support num of inliers', type=int)
    parser.add_argument(
        '--reproj-thre', help='reprojection threshold to accept inliers', type=int)
    parser.add_argument(
        '--use-reproj', help='use reprojected 2d as label', action='store_true')
    parser.add_argument(
        '--loop', help='whether in loop training, only generate one pseudo_label', action='store_true')
    parser.add_argument(
        '--no-distortion', help='whether use no distortion data', action='store_true')
    parser.add_argument(
        '--net-layers', help='network layers', type=int, default=50, choices=[50, 152])

    parser.add_argument(
        '--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument(
        '--dataDir', help='data directory', type=str, default='')
    args = parser.parse_args()
    update_dir(args.modelDir, args.logDir, args.dataDir)
    return args


def reset_config(config, args):
    if args.confidence_thre:
        config.PSEUDO_LABEL.CONFIDENCE_THRE = args.confidence_thre
    if args.ransac:
        config.PSEUDO_LABEL.IF_RANSAC = args.ransac
    if args.inliers:
        config.PSEUDO_LABEL.NUM_INLIERS = args.inliers
    if args.reproj_thre:
        config.PSEUDO_LABEL.REPROJ_THRE = args.reproj_thre
    if args.use_reproj:
        config.PSEUDO_LABEL.USE_REPROJ = args.use_reproj
    if args.loop:
        config.PSEUDO_LABEL.IF_LOOP = args.loop
    if args.no_distortion:
        config.DATASET.NO_DISTORTION = args.no_distortion
    if args.net_layers:
        config.POSE_RESNET.NUM_LAYERS = args.net_layers
    return


def my_eval(pred2d, gt2d, joints_vis, headsizes, threshold=0.5):
    """
    pred2d: [N, 16, 2]
    gt2d: [N, 16, 2]
    joints_vis: [N, 16]
    headsizes: [N, 1]
    """
    distance = np.sqrt(np.sum((gt2d - pred2d)**2, axis=2))  # [N, njoints]
    scaled_distance = np.divide(distance, headsizes)  # [N, njoints]
    detected = (scaled_distance <= threshold)

    considered_detected = detected * joints_vis  # [N, njoints]
    joint_detection_rate = np.sum(considered_detected, axis=0) / np.sum(joints_vis, axis=0).astype(np.float32)

    joint_ratio = np.sum(joints_vis, axis=0) / np.sum(joints_vis).astype(np.float32)
    mean_pckh = np.sum(joint_ratio*joint_detection_rate)
    return mean_pckh


def create_logger(cfg, cfg_name):
    root_output_dir = Path(cfg.OUTPUT_DIR)  # console dir
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    cfg_name_list = os.path.basename(cfg_name).split('.')[:-1]
    cfg_name = '.'.join(cfg_name_list)

    # pth/test_pseudo_label_152/
    if cfg.POSE_RESNET.NUM_LAYERS != 50:
        cfg_name+='_{:d}'.format(cfg.POSE_RESNET.NUM_LAYERS)

    final_output_dir = root_output_dir / 'test' / cfg_name / '{}_{}'.format(cfg.PSEUDO_LABEL.NUM_INLIERS, config.PSEUDO_LABEL.REPROJ_THRE)

    if not final_output_dir.exists():
        print('=> creating {}'.format(final_output_dir))
        final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg.PSEUDO_LABEL.NUM_INLIERS,
        config.PSEUDO_LABEL.REPROJ_THRE, time_str)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file), format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    return logger, final_output_dir


def main():
    args = parse_args()
    reset_config(config, args)
    logger, final_output_dir = create_logger(config, args.cfg)
    logger.info(pprint.pformat(config.PSEUDO_LABEL))

    test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, 'train', True, pseudo_label_path='', no_distortion=config.DATASET.NO_DISTORTION)
    grouping = test_dataset.grouping
    logger.info('=> Dataset: {}_{}_{}'.format(config.DATASET.TEST_DATASET, 'train', 
        'nodistortion' if config.DATASET.NO_DISTORTION else 'distortion'))
    
    test_data_file = args.heatmap
    db = h5py.File(test_data_file, 'r')
    pred2d = np.array(db['locations'])[:, :, :2]  # [8860, 16, 2]
    confidence =np.array(db['locations'][:, :, 2])  # [8860, 16]
    assert len(pred2d) == len(grouping) * len(grouping[0])

    u2a = test_dataset.u2a_mapping
    u2a = {k:v  for k, v in u2a.items() if v != '*'}
    sorted_u2a = sorted(u2a.items(), key=lambda x: x[0])
    u = np.array([mapping[0] for mapping in sorted_u2a])

    cameras = []
    gt2d = []
    scales = []
    for items in grouping:
        for item in items:
            cam = test_dataset.db[item]['camera']
            cameras.append(cam)
            gt2d.append(test_dataset.db[item]['joints_2d'])
            scales.append(test_dataset.db[item]['scale'])
    gt2d = np.array(gt2d)[:, u, :]  # [8860, 16, 2]
    assert len(gt2d) == len(pred2d)
    headsizes = np.amax(np.array(scales), axis=1, keepdims=True) * 200 / 10.0  # [8860, 1]

    # Baseline
    # resutls = h5py.File(final_output_dir / '..' / 'baseline.h5', 'w')
    # resutls['pseudo_2d'] = pred2d
    # resutls['joints_vis'] = np.ones_like(confidence)
    # resutls.close()

    # RANSAC
    # confidence:
    #     max: 1.12, min:0.04, mean:0.83, std:0.12
    # thre: 0.6 or 0.7
    names = []
    acc = []
    num = []
    thre_list = [0.6, 0.7, 0.8, 0.9] if not config.PSEUDO_LABEL.IF_LOOP else [config.PSEUDO_LABEL.CONFIDENCE_THRE]
    # thre_list = [0, 0.1, 0.2, 0.3, 0.4, 0.5] if not config.PSEUDO_LABEL.IF_LOOP else [config.PSEUDO_LABEL.CONFIDENCE_THRE]
    for conf_thre in thre_list:
        joints_vis = confidence > conf_thre
        pckh = my_eval(pred2d, gt2d, joints_vis, headsizes, threshold=0.5)
        num_vis = np.sum(joints_vis) / joints_vis.size
        joints_group_num = np.sum(np.reshape(joints_vis, (-1, 4, 16)), axis=1)
        logger.info('----- thre %f -------' % conf_thre)
        logger.info('PCKh@0.5: %.3f' % pckh)
        logger.info('Vis: %.2f' % num_vis)
        logger.info('Joints@4: %.2f' % (np.sum(joints_group_num == 4) / joints_group_num.size))
        logger.info('Joints@3: %.2f' % (np.sum(joints_group_num == 3) / joints_group_num.size))
        logger.info('Joints@2: %.2f' % (np.sum(joints_group_num == 2) / joints_group_num.size))
        logger.info('Joints@1: %.2f' % (np.sum(joints_group_num == 1) / joints_group_num.size))
        logger.info('Joints@0: %.2f' % (np.sum(joints_group_num == 0) / joints_group_num.size))

        acc.append(pckh)
        num.append(num_vis)
        name = '{}_{}'.format(conf_thre, 0)
        names.append(name)

        if not (config.PSEUDO_LABEL.IF_LOOP and config.PSEUDO_LABEL.IF_RANSAC):
            resutls = h5py.File(final_output_dir / (name + '_pseudo_label.h5'), 'w')
            resutls['pseudo_2d'] = pred2d
            resutls['joints_vis'] = joints_vis
            resutls.close()
            logger.info('=> Save to: ' + str(final_output_dir / (name + '_pseudo_label.h5')))

        # After RANSAC
        if config.PSEUDO_LABEL.IF_RANSAC:
            joints_vis = ransac(pred2d, cameras, joints_vis, config)
            pckh = my_eval(pred2d, gt2d, joints_vis, headsizes, threshold=0.5)
            num_vis = np.sum(joints_vis) / joints_vis.size
            joints_group_num = np.sum(np.reshape(joints_vis, (-1, 4, 16)), axis=1)
            logger.info('-- After RANSAC --')
            logger.info('PCKh@0.5: %.3f' % pckh)
            logger.info('Vis: %.2f' % num_vis)
            logger.info('Joints@4: %.2f' % (np.sum(joints_group_num == 4) / joints_group_num.size))
            logger.info('Joints@3: %.2f' % (np.sum(joints_group_num == 3) / joints_group_num.size))
            logger.info('Joints@2: %.2f' % (np.sum(joints_group_num == 2) / joints_group_num.size))
            logger.info('Joints@1: %.2f' % (np.sum(joints_group_num == 1) / joints_group_num.size))
            logger.info('Joints@0: %.2f' % (np.sum(joints_group_num == 0) / joints_group_num.size))
            # only log, don't use these pseudo labels

        # After Reprojection
        if config.PSEUDO_LABEL.USE_REPROJ:
            proj2d, joints_vis = reproject_poses(pred2d, cameras, joints_vis, config.DATASET.NO_DISTORTION)
            pckh = my_eval(proj2d, gt2d, joints_vis, headsizes, threshold=0.5)
            num_vis = np.sum(joints_vis) / joints_vis.size
            joints_group_num = np.sum(np.reshape(joints_vis, (-1, 4, 16)), axis=1)
            logger.info('-- After Reprojection --')
            logger.info('PCKh@0.5: %.3f' % pckh)
            logger.info('Vis: %.2f' % num_vis)
            logger.info('Joints@4: %.2f' % (np.sum(joints_group_num == 4) / joints_group_num.size))
            logger.info('Joints@3: %.2f' % (np.sum(joints_group_num == 3) / joints_group_num.size))
            logger.info('Joints@2: %.2f' % (np.sum(joints_group_num == 2) / joints_group_num.size))
            logger.info('Joints@1: %.2f' % (np.sum(joints_group_num == 1) / joints_group_num.size))
            logger.info('Joints@0: %.2f' % (np.sum(joints_group_num == 0) / joints_group_num.size))

            acc.append(pckh)
            num.append(num_vis)
            name = '{}_{}'.format(conf_thre, 1)
            names.append(name)

            resutls = h5py.File(final_output_dir / (name + '_pseudo_label.h5'), 'w')
            resutls['pseudo_2d'] = proj2d
            resutls['joints_vis'] = joints_vis
            resutls.close()
            logger.info('=> Save to: ' + str(final_output_dir / (name + '_pseudo_label.h5')))

    if not config.PSEUDO_LABEL.IF_LOOP:
        _, acc_order = np.unique(acc, return_inverse=True)
        _, num_order = np.unique(num, return_inverse=True)

        sum_order = list(np.argsort(acc_order + num_order))  # ascent order
        final_indices = []
        while sum_order:
            ref_idx = sum_order.pop()
            final_indices.append(ref_idx)
            remove_list = []
            for rest_idx in sum_order:
                if acc_order[rest_idx] <= acc_order[ref_idx] and num_order[rest_idx] <= num_order[ref_idx]:
                    remove_list.append(rest_idx)
            sum_order = [idx for idx in sum_order if idx not in remove_list]

        # save selected files and deleted files
        with open(final_output_dir / 'select.txt', 'w') as f:
            for idx in final_indices:
                file_path = final_output_dir / (names[idx] + '_pseudo_label.h5')
                f.write(str(file_path) + '\n')

        remove_indices = [k for k in range(len(names)) if k not in final_indices]
        with open(final_output_dir / 'delete.txt', 'w') as f:
            for idx in remove_indices:
                file_path = final_output_dir / (names[idx] + '_pseudo_label.h5')
                f.write(str(file_path) + '\n')


if __name__ == '__main__':
    main()
