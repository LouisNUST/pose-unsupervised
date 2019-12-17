
import pickle
import numpy as np
import cv2
# from lib.multiviews.cameras import project_pose, camera_to_world_frame, unfold_camera_param
import argparse
# from pymvg.multi_camera_system import build_example_system
import numpy as np
import itertools
import logging
from pathlib import Path
import yaml
import os
import h5py

my_file_path = r"D:\Dataset\h36m\h36m_validation.pkl"
# my_file_path = r'D:\Code\pose_multiview\data\h36m\annot\h36m_train.pkl'
old_file_path = r'D:\Code\pose_unsupervised\h36m_zero_center.pkl'

# with open(my_file_path, 'rb') as f:
#     my_dataset = pickle.load(f)

# for idx, item in enumerate(my_dataset):
#     pose3d_camera = item['joints_3d']
#     camera = item['camera']
#     R, T, f, c, k, p = unfold_camera_param(camera, avg_f=False)
#     camera_matrix = np.array(
#             [[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1]], dtype=float)
#     pose2d, _ = cv2.projectPoints(pose3d_camera, (0, 0, 0), (0, 0, 0), camera_matrix, np.array((k[0],k[1],p[0],p[1],k[2])))
#     pose2d = pose2d.squeeze()  # [17, 2]
#     item['joints_2d'] = pose2d  # replace original pose2d

# with open(my_file_path, 'wb') as f:
#     pickle.dump(my_dataset, f)

import _init_paths
import dataset
from core.config import config
from core.config import update_config
from core.inference import get_final_preds, get_max_preds
from utils.transforms import transform_preds

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test Pseudo Labels')
    # Required Param
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    parser.add_argument(
        '--heatmap', help='heatmap file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    args = parser.parse_args()
    return args
args = parse_args()

heatmap_path = args.heatmap
h5_dataset = h5py.File(heatmap_path, 'r')
# all_preds = np.array(h5_dataset['locations'])[:, :, :2]  # [8860, 16, 2]
heatmaps = np.array(h5_dataset['heatmaps'])  # [8860, 16, h, w]

# get integral locations
heatmaps = heatmaps / np.sum(heatmaps, axis=(2, 3), keepdims=True)
coordinates = np.arange(64).reshape((1, 1, 64))
accu_w = np.sum(heatmaps, axis=2)  # [8860, 16, 64]
accu_h = np.sum(heatmaps, axis=3)  # [8860, 16, 64]
w_coordinates = np.sum(accu_w * coordinates, axis=2)  # [8860, 16]
h_coordinates = np.sum(accu_h * coordinates, axis=2)  # [8860, 16]
intergral_preds = np.stack((w_coordinates, h_coordinates), axis=2)  # [8860, 16, 2]

test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
    config, config.DATASET.TEST_SUBSET, False)

# get center and scale
center = []
scale = []
for items in test_dataset.grouping:
    for item in items:
        center.append(np.array(test_dataset.db[item]['center']))
        scale.append(np.array(test_dataset.db[item]['scale']))
assert len(center) == len(intergral_preds)

all_preds = np.zeros_like(intergral_preds)  # [8860, 16, 2]
# Transform back
for i in range(all_preds.shape[0]):
    all_preds[i] = transform_preds(intergral_preds[i], center[i], scale[i],
                               [heatmaps.shape[3], heatmaps.shape[2]])

name_value, perf_indicator = test_dataset.evaluate(all_preds, None)
names = name_value.keys()
values = name_value.values()
num_values = len(name_value)
print('| Arch ' +
            ' '.join(['| {}'.format(name) for name in names]) + ' |')
print('|---' * (num_values + 1) + '|')
print('| ' + 'multiview_pose_resnet50X256' + ' ' +
            ' '.join(['| {:.3f}'.format(value) for value in values]) +
            ' |')
