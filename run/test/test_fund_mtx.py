
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


test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False, no_distortion=False)

write_path = './data/testdata/fundamental_matrix.pkl'

with open(write_path, 'rb') as f:
    fundamental_matrix = pickle.load(f)

# print(list(fundamental_matrix.items())[0])

test_data_file = args.heatmap
db = h5py.File(test_data_file, 'r')
pred2d = np.array(db['locations'])[:, :, :2]  # [8860, 16, 2]
pred2d = np.reshape(pred2d, (len(pred2d)//4, 4, 16, 2))  # [8860/4, 4, 16, 2]
assert len(pred2d) == len(test_dataset.grouping), '{}, {}'.format(len(pred2d), len(test_dataset.grouping))

pairs = list(itertools.permutations([0, 1, 2, 3], 2))
res = []
for items, batch in zip(test_dataset.grouping, pred2d):
    subj = test_dataset.db[items[0]]['subject']
    for pair in pairs:
        pts1 = batch[pair[0]]
        pts2 = batch[pair[1]]
        new_pts1 = np.concatenate((pts1, np.ones((len(pts1), 1))), axis=1)
        new_pts2 = np.concatenate((pts2, np.ones((len(pts2), 1))), axis=1)
        F = fundamental_matrix[(subj, pair[0], pair[1])]
        res_mean = np.sum((new_pts2 @ F) * new_pts1, axis=1)
        res.append(res_mean)

res = np.abs(np.array(res))
print('mean: {}'.format(np.mean(res)))
print('max: {}'.format(np.amax(res)))

