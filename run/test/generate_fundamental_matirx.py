
import pickle
import numpy as np
import cv2
import argparse
import numpy as np
import h5py
import os
import itertools

import _init_paths
import dataset
from core.config import config
from core.config import update_config

write_path = './data/testdata/fundamental_matrix.pkl'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test Pseudo Labels')
    # Required Param
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    # parser.add_argument(
    #     '--heatmap', help='heatmap file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    args = parser.parse_args()
    return args
args = parse_args()

test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False, no_distortion=True)

test_fund_dict = {}  # {(subj, view1, view2): fund_matrix}
subj_list = []
pairs = list(itertools.permutations([0, 1, 2, 3], 2))
res = []
for items in test_dataset.grouping:
    subj = test_dataset.db[items[0]]['subject']
    if subj  not in subj_list:
        subj_list.append(subj)
        for pair in pairs:
            pts1 = test_dataset.db[items[pair[0]]]['joints_2d']
            pts2 = test_dataset.db[items[pair[1]]]['joints_2d']
            F, mask = cv2.findFundamentalMat(pts1,pts2, cv2.FM_LMEDS)
            assert np.sum(mask) >= 8
            test_fund_dict[(subj, pair[0], pair[1])] = F
    else:
        for pair in pairs:
            pts1 = test_dataset.db[items[pair[0]]]['joints_2d']
            pts2 = test_dataset.db[items[pair[1]]]['joints_2d']
            new_pts1 = np.concatenate((pts1, np.ones((len(pts1), 1))), axis=1)
            new_pts2 = np.concatenate((pts2, np.ones((len(pts2), 1))), axis=1)
            F = test_fund_dict[(subj, pair[0], pair[1])]
            res_mean = np.sum((new_pts2 @ F) * new_pts1, axis=1)
            res.append(res_mean)

print(len(test_fund_dict))
res = np.abs(np.array(res))
print('mean: {}'.format(np.mean(res)))
print('max: {}'.format(np.amax(res)))



train_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, 'train', False, no_distortion=True)

train_fund_dict = {}  # {(subj, view1, view2): fund_matrix}
subj_list = []
pairs = list(itertools.permutations([0, 1, 2, 3], 2))
res = []
for items in train_dataset.grouping:
    subj = train_dataset.db[items[0]]['subject']
    if subj  not in subj_list:
        subj_list.append(subj)
        for pair in pairs:
            pts1 = train_dataset.db[items[pair[0]]]['joints_2d']
            pts2 = train_dataset.db[items[pair[1]]]['joints_2d']
            F, mask = cv2.findFundamentalMat(pts1,pts2, cv2.FM_LMEDS)
            assert np.sum(mask) >= 8
            train_fund_dict[(subj, pair[0], pair[1])] = F
    else:
        for pair in pairs:
            pts1 = train_dataset.db[items[pair[0]]]['joints_2d']
            pts2 = train_dataset.db[items[pair[1]]]['joints_2d']
            new_pts1 = np.concatenate((pts1, np.ones((len(pts1), 1))), axis=1)
            new_pts2 = np.concatenate((pts2, np.ones((len(pts2), 1))), axis=1)
            F = train_fund_dict[(subj, pair[0], pair[1])]
            res_mean = np.sum((new_pts2 @ F) * new_pts1, axis=1)
            res.append(res_mean)

print(len(train_fund_dict))
res = np.abs(np.array(res))
print('mean: {}'.format(np.mean(res)))
print('max: {}'.format(np.amax(res)))

test_fund_dict.update(train_fund_dict)
assert len(test_fund_dict) == 12 * 7

with open(write_path, 'wb') as f:
    pickle.dump(test_fund_dict, f)

# item = test_dataset.grouping[0]
# pts1 = test_dataset.db[item[0]]['joints_2d']  # [16, 2]
# pts2 = test_dataset.db[item[1]]['joints_2d']  # [16, 2]

# F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.FM_LMEDS)
# print(F)

# F2, mask2 = cv2.findFundamentalMat(pts2,pts1,cv2.FM_LMEDS)
# print(F2)
# print(F2.T-F)

# # We select only inlier points
# pts1 = pts1[mask.ravel()==1]
# pts2 = pts2[mask.ravel()==1]

# new_pts1 = np.concatenate((pts1, np.ones((len(pts1), 1))), axis=1)
# new_pts2 = np.concatenate((pts2, np.ones((len(pts2), 1))), axis=1)
# res = np.sum((new_pts2 @ F) * new_pts1, axis=1)
# print(res)

