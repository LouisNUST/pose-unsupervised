# ------------------------------------------------------------------------------
# multiview.pose3d.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

import time
import pickle
import argparse
import numpy as np
from scipy.sparse import lil_matrix

import _init_paths
from core.config import config
from core.config import update_config
from multiviews.body import HumanBody
import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Generate Pairwise Constraint for RPSM')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)
    return args


def compute_avg_limb_length(dataset):
    avg_limb_length = {}
    humanbody = HumanBody()
    skeleton = humanbody.skeleton

    for i, db in enumerate(dataset.db):
        joints3d = db['joints_3d']

        for node in skeleton:
            current = node['idx']
            children = node['children']

            for child in children:
                limb_length = np.linalg.norm(joints3d[current] -
                                             joints3d[child])

                if (current, child) not in avg_limb_length:
                    avg_limb_length[(current, child)] = [limb_length]
                else:
                    avg_limb_length[(current, child)].append(limb_length)

    for k, v in avg_limb_length.items():
        avg_limb_length[k] = np.mean(v)

    with open('data/testdata/avg_limb_length.pkl', 'wb') as f:
        pickle.dump(avg_limb_length, f)
    return avg_limb_length


def compute_pairwise_constraint(boxSize, avg_limb_length, nbins=16):

    grid1D = np.linspace(-boxSize / 2, boxSize / 2, nbins)
    Gridx, Gridy, Gridz = np.meshgrid(
        grid1D,
        grid1D,
        grid1D,
    )

    dimensions = Gridx.shape[0] * Gridx.shape[1] * Gridx.shape[2]
    Gridx, Gridy, Gridz = np.reshape(Gridx, (dimensions, -1)), np.reshape(
        Gridy, (dimensions, -1)), np.reshape(Gridz, (dimensions, -1))
    xyz = np.concatenate((Gridx, Gridy, Gridz), axis=1)

    humanbody = HumanBody()
    skeleton = humanbody.skeleton

    size = xyz.shape[0]
    pairwise_constrain = {}

    for node in skeleton:
        current = node['idx']
        children = node['children']

        for child in children:
            expect_length = avg_limb_length[(current, child)]
            constrain_array = lil_matrix((size, size), dtype=np.int8)

            for i in range(size):
                for j in range(size):
                    actual_length = np.linalg.norm(xyz[i] - xyz[j])
                    offset = np.abs(actual_length - expect_length)
                    if offset < 0.4 * expect_length:
                        constrain_array[i, j] = 1
            pairwise_constrain[(current, child)] = constrain_array
    return pairwise_constrain


def main():
    parse_args()
    train_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TRAIN_SUBSET, False)

    avg_limb_length = compute_avg_limb_length(train_dataset)

    nbins = 16
    pairwise_constrain = compute_pairwise_constraint(
        config.DATASET.BBOX, avg_limb_length, nbins=nbins)

    d = dict(limb_length=avg_limb_length, pairwise_constrain=pairwise_constrain)
    with open('data/testdata/pairwise_b{}.pkl'.format(nbins), 'wb') as f:
        pickle.dump(d, f)


if __name__ == '__main__':
    main()