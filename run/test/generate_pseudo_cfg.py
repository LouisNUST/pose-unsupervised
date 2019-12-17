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
import yaml

import _init_paths
import dataset
from core.config import config
from core.config import update_config
from multiviews.cameras import camera_to_world_frame, project_pose
from multiviews.triangulate import triangulate_poses, ransac, reproject_poses


def parse_args():
    parser = argparse.ArgumentParser(
        description='Generate a batch of training cfgs accroding to a template cfg')
    # Required Param
    parser.add_argument(
        '--cfg', help='template configure file name', required=True, type=str)
    parser.add_argument(
        '--no-distortion', help='wheter use no distortion data', action='store_true')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    if args.no_distortion:
        root_dir = os.path.join('output', 'test', 'test_pseudo_label_nodistortion')
    else:
        root_dir = os.path.join('output', 'test', 'test_pseudo_label')

    sub_dirs = ['3_12', '4_12']

    with open(args.cfg, 'r') as f:
        ref_yaml = yaml.load(f)

    cfg_prefix = '256_nofusion_resume_pseudo_'
    cfg_output_dir = 'experiments/mixed/resnet50/pseudo_label/'

    if not os.path.exists(cfg_output_dir):
        os.makedirs(cfg_output_dir)

    for dir_name in sub_dirs:
        dir_path = os.path.join(root_dir, dir_name)
        selected_file = os.path.join(dir_path, 'select.txt')
        if os.path.exists(selected_file):
            with open(selected_file, 'r') as f:
                all_paths = f.readlines()
            all_paths = [path.strip('\n') for path in all_paths]

            for pseudo_label_path in all_paths:
                pseudo_label_name = os.path.basename(pseudo_label_path)[:-3]  # strip .h5
                new_cfg_basename = cfg_prefix + dir_name + '_' + pseudo_label_name
                new_cfg_basename = new_cfg_basename if not args.no_distortion else new_cfg_basename + '_nodistortion' 
                new_cfg_name = new_cfg_basename + '.yaml'

                ref_yaml['DATASET']['PSEUDO_LABEL_PATH'] = pseudo_label_path

                final_output_path = os.path.join(cfg_output_dir, new_cfg_name)
                print('=> writing {}'.format(final_output_path))
                with open(final_output_path, 'w') as f:
                    yaml.dump(ref_yaml, f)

    for dir_name in sub_dirs:
        dir_path = os.path.join(root_dir, dir_name)
        selected_file = os.path.join(dir_path, 'delete.txt')
        if os.path.exists(selected_file):
            with open(selected_file, 'r') as f:
                all_paths = f.readlines()
            all_paths = [path.strip('\n') for path in all_paths]

            for pseudo_label_path in all_paths:
                pseudo_label_name = os.path.basename(pseudo_label_path)[:-3]  # strip .h5
                new_cfg_basename = cfg_prefix + dir_name + '_' + pseudo_label_name
                new_cfg_basename = new_cfg_basename if not args.no_distortion else new_cfg_basename + '_nodistortion' 
                new_cfg_name = new_cfg_basename + '.yaml'

                ref_yaml['DATASET']['PSEUDO_LABEL_PATH'] = pseudo_label_path

                final_output_path = os.path.join(cfg_output_dir, new_cfg_name)
                print('=> writing {}'.format(final_output_path))
                with open(final_output_path, 'w') as f:
                    yaml.dump(ref_yaml, f)


if __name__ == '__main__':
    main()
