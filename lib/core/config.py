# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import yaml

import numpy as np
from easydict import EasyDict as edict


config = edict()

config.OUTPUT_DIR = 'output'
config.LOG_DIR = 'log'
config.DATA_DIR = ''
config.BACKBONE_MODEL = 'pose_resnet'
config.MODEL = 'multiview_pose_resnet'
config.GPUS = '0,1'
config.WORKERS = 8
config.PRINT_FREQ = 100

# Cudnn related params
config.CUDNN = edict()
config.CUDNN.BENCHMARK = True
config.CUDNN.DETERMINISTIC = False
config.CUDNN.ENABLED = True

# common params for NETWORK
config.NETWORK = edict()
config.NETWORK.PRETRAINED = 'models/pytorch/imagenet/resnet50-19c8e357.pth'
config.NETWORK.NUM_JOINTS = 16
config.NETWORK.HEATMAP_SIZE = np.array([80, 80])
config.NETWORK.IMAGE_SIZE = np.array([320, 320])
config.NETWORK.SIGMA = 2
config.NETWORK.TARGET_TYPE = 'gaussian'
config.NETWORK.AGGRE = True

# pose_resnet related params
config.POSE_RESNET = edict()
config.POSE_RESNET.NUM_LAYERS = 50
config.POSE_RESNET.DECONV_WITH_BIAS = False
config.POSE_RESNET.NUM_DECONV_LAYERS = 3
config.POSE_RESNET.NUM_DECONV_FILTERS = [256, 256, 256]
config.POSE_RESNET.NUM_DECONV_KERNELS = [4, 4, 4]
config.POSE_RESNET.FINAL_CONV_KERNEL = 1

config.LOCAL_DISCRIMINATOR = edict()
config.LOCAL_DISCRIMINATOR.LOW_FEATURES_CHANNELS = 256
config.LOCAL_DISCRIMINATOR.HIGH_FEATURES_CHANNELS = 256
config.LOCAL_DISCRIMINATOR.OUTPUT_CHANNELS = 2048

config.DOMAIN_DISCRIMINATOR = edict()
config.DOMAIN_DISCRIMINATOR.FEATURES_CHANNELS = 2048

config.VIEW_DISCRIMINATOR = edict()
config.VIEW_DISCRIMINATOR.VIEW_ONE_NUM = 1
config.VIEW_DISCRIMINATOR.VIEW_TWO_NUM = 3
config.VIEW_DISCRIMINATOR.OUTPUT_CHANNELS = 256

config.JOINTS_DISCRIMINATOR = edict()
config.JOINTS_DISCRIMINATOR.VAR_ONE_NUM = 4
config.JOINTS_DISCRIMINATOR.VAR_TWO_NUM = 12
config.JOINTS_DISCRIMINATOR.OUTPUT_CHANNELS = 128
config.JOINTS_DISCRIMINATOR.VAR_ONE_IDX = np.array([0, 5, 10, 15], dtype=np.int)  # union dataset order

config.HEATMAP_DISCRIMINATOR = edict()
config.HEATMAP_DISCRIMINATOR.INPUT_CHANNELS = 257
config.HEATMAP_DISCRIMINATOR.INTER_CHANNELS = 64
config.HEATMAP_DISCRIMINATOR.JOINT_IDX = 0

# config.GLOBAL_DISCRIMINATOR = edict()
# config.GLOBAL_DISCRIMINATOR.

config.LOSS = edict()
config.LOSS.USE_TARGET_WEIGHT = True
config.LOSS.USE_TARGET_WEIGHT_FUND = True

# loss
config.LOSS.USE_CONSISTENT_LOSS = False
config.LOSS.USE_FUNDAMENTAL_LOSS = False
config.LOSS.USE_GLOBAL_MI_LOSS = False
config.LOSS.USE_LOCAL_MI_LOSS = False
config.LOSS.USE_GRADIENT_CLIP = False
config.LOSS.USE_LOW_FEATURES_PREPROCESS = False
config.LOSS.USE_DOMAIN_TRANSFER_LOSS = False
config.LOSS.USE_VIEW_MI_LOSS = False
config.LOSS.USE_JOINTS_MI_LOSS = False
config.LOSS.USE_HEATMAP_MI_LOSS = False
config.LOSS.WATCH_GRAD_NORM = False


config.LOSS.MSE_LOSS_WEIGHT = 1
config.LOSS.CONSISTENT_LOSS_WEIGHT = 0.01
config.LOSS.FUNDAMENTAL_LOSS_WEIGHT = 1
config.LOSS.GLOBAL_MI_LOSS_WEIGHT = 1
config.LOSS.LOCAL_MI_LOSS_WEIGHT = 1
config.LOSS.DOMAIN_LOSS_WEIGHT = 1
config.LOSS.VIEW_MI_LOSS_WEIGHT = 1
config.LOSS.JOINTS_MI_LOSS_WEIGHT = 1
config.LOSS.HEATMAP_MI_LOSS_WEIGHT = 1


config.LOSS.MI_MEASURE = 'JSD'
config.LOSS.MI_NEG_POS_RATIO = 2
config.LOSS.MI_POSITIVE_NUM = 16
config.LOSS.SPECIFIC = 'org'
config.LOSS.VIEW_MI_MEASURE = 'NCE'
config.LOSS.JOINTS_MI_MEASURE = 'NCE'
config.LOSS.HEATMAP_MI_MEASURE = 'NCE'


# DATASET related params
config.DATASET = edict()
config.DATASET.ROOT = 'data/'
config.DATASET.TRAIN_DATASET = 'mixed_dataset'
config.DATASET.TEST_DATASET = 'multiview_h36m'
config.DATASET.TRAIN_SUBSET = 'train'
config.DATASET.TEST_SUBSET = 'validation'
config.DATASET.PSEUDO_LABEL_PATH = ''
config.DATASET.NO_DISTORTION = False
config.DATASET.ROOTIDX = 0
config.DATASET.DATA_FORMAT = 'jpg'
config.DATASET.BBOX = 2000

# training data augmentation
config.DATASET.MPII_SCALE_FACTOR = 0
config.DATASET.MPII_ROT_FACTOR = 0
config.DATASET.MPII_FLIP = False

config.DATASET.H36M_SCALE_FACTOR = 0
config.DATASET.H36M_ROT_FACTOR = 0
config.DATASET.H36M_FLIP = False

config.DATASET.COCO_SCALE_FACTOR = 0
config.DATASET.COCO_ROT_FACTOR = 0
config.DATASET.COCO_FLIP = False

# params for mpii dataset and h36m dataset
config.DATASET.MPII_ROOTIDX = 6
config.DATASET.H36M_ROOTIDX = 0

# params for mixed dataset, balance batch smaples
# h36m:mpii = 39:1
config.DATASET.IF_SAMPLE = False
config.DATASET.H36M_WEIGHT = 1
config.DATASET.MPII_WEIGHT = 10

config.DATASET.COLOR_JITTER = False

# train
config.TRAIN = edict()
config.TRAIN.LR_FACTOR = 0.1
config.TRAIN.LR_STEP = [90, 110]
config.TRAIN.LR = 0.001
config.TRAIN.LR_DISCRIMINATOR = 0.001

config.TRAIN.OPTIMIZER = 'adam'
config.TRAIN.MOMENTUM = 0.9
config.TRAIN.WD = 0.0001
config.TRAIN.NESTEROV = False
config.TRAIN.GAMMA1 = 0.99
config.TRAIN.GAMMA2 = 0.0

config.TRAIN.BEGIN_EPOCH = 0
config.TRAIN.END_EPOCH = 140

config.TRAIN.RESUME = False
config.TRAIN.RESUME_PATH = ''
config.TRAIN.ON_SERVER_CLUSTER = False

config.TRAIN.BATCH_SIZE = 8
config.TRAIN.SHUFFLE = True

config.TRAIN.FIX_BACKBONE = False

# testing
config.TEST = edict()
config.TEST.BATCH_SIZE = 8
config.TEST.STATE = ''
config.TEST.FLIP_TEST = False
config.TEST.POST_PROCESS = False
config.TEST.SHIFT_HEATMAP = False
config.TEST.USE_GT_BBOX = False
config.TEST.IMAGE_THRE = 0.1
config.TEST.NMS_THRE = 0.6
config.TEST.OKS_THRE = 0.5
config.TEST.IN_VIS_THRE = 0.0
config.TEST.BBOX_FILE = ''
config.TEST.BBOX_THRE = 1.0
config.TEST.MATCH_IOU_THRE = 0.3
config.TEST.DETECTOR = 'fpn_dcn'
config.TEST.DETECTOR_DIR = ''
config.TEST.MODEL_FILE = ''
config.TEST.FUSE_OUTPUT = True

# debug
config.DEBUG = edict()
config.DEBUG.DEBUG = True
config.DEBUG.SAVE_BATCH_IMAGES_GT = True
config.DEBUG.SAVE_BATCH_IMAGES_PRED = True
config.DEBUG.SAVE_HEATMAPS_GT = True
config.DEBUG.SAVE_HEATMAPS_PRED = True
config.DEBUG.SAVE_ALL_PREDS = False


# pictorial structure
config.PICT_STRUCT = edict()
config.PICT_STRUCT.FIRST_NBINS = 16
config.PICT_STRUCT.RECUR_NBINS = 2
config.PICT_STRUCT.RECUR_DEPTH = 10
config.PICT_STRUCT.LIMB_LENGTH_TOLERANCE = 150
config.PICT_STRUCT.GRID_SIZE = 2000
config.PICT_STRUCT.DEBUG = False
config.PICT_STRUCT.TEST_PAIRWISE = False
config.PICT_STRUCT.SHOW_ORIIMG = False
config.PICT_STRUCT.SHOW_CROPIMG = False
config.PICT_STRUCT.SHOW_HEATIMG = False

# pseudo label
config.PSEUDO_LABEL = edict()
config.PSEUDO_LABEL.CONFIDENCE_THRE = 0.6  # -100 no threshold
config.PSEUDO_LABEL.IF_RANSAC = True
config.PSEUDO_LABEL.NUM_INLIERS = 4
config.PSEUDO_LABEL.REPROJ_THRE = 10
config.PSEUDO_LABEL.USE_REPROJ = False
config.PSEUDO_LABEL.REPROJ_TO_OUTLIERS = False  # valid when USE_REPROJ=True
config.PSEUDO_LABEL.IF_LOOP = False

def _update_dict(k, v):
    if k == 'DATASET':
        if 'MEAN' in v and v['MEAN']:
            v['MEAN'] = np.array([eval(x) if isinstance(x, str) else x
                                  for x in v['MEAN']])
        if 'STD' in v and v['STD']:
            v['STD'] = np.array([eval(x) if isinstance(x, str) else x
                                 for x in v['STD']])
    if k == 'NETWORK':
        if 'HEATMAP_SIZE' in v:
            if isinstance(v['HEATMAP_SIZE'], int):
                v['HEATMAP_SIZE'] = np.array([v['HEATMAP_SIZE'],
                                              v['HEATMAP_SIZE']])
            else:
                v['HEATMAP_SIZE'] = np.array(v['HEATMAP_SIZE'])
        if 'IMAGE_SIZE' in v:
            if isinstance(v['IMAGE_SIZE'], int):
                v['IMAGE_SIZE'] = np.array([v['IMAGE_SIZE'], v['IMAGE_SIZE']])
            else:
                v['IMAGE_SIZE'] = np.array(v['IMAGE_SIZE'])
    for vk, vv in v.items():
        if vk in config[k]:
            config[k][vk] = vv
        else:
            raise ValueError("{}.{} not exist in config.py".format(k, vk))


def update_config(config_file):
    exp_config = None
    with open(config_file) as f:
        exp_config = edict(yaml.load(f, Loader=yaml.FullLoader))
        for k, v in exp_config.items():
            if k in config:
                if isinstance(v, dict):
                    _update_dict(k, v)
                else:
                    if k == 'SCALES':
                        config[k][0] = (tuple(v))
                    else:
                        config[k] = v
            else:
                raise ValueError("{} not exist in config.py".format(k))


def gen_config(config_file):
    cfg = dict(config)
    for k, v in cfg.items():
        if isinstance(v, edict):
            cfg[k] = dict(v)

    with open(config_file, 'w') as f:
        yaml.dump(dict(cfg), f, default_flow_style=False)


def update_dir(model_dir, log_dir, data_dir):
    if model_dir:
        config.OUTPUT_DIR = model_dir

    if log_dir:
        config.LOG_DIR = log_dir

    if data_dir:
        config.DATA_DIR = data_dir

    config.DATASET.ROOT = os.path.join(
            config.DATA_DIR, config.DATASET.ROOT)

    config.TEST.BBOX_FILE = os.path.join(
            config.DATA_DIR, config.TEST.BBOX_FILE)

    config.NETWORK.PRETRAINED = os.path.join(
            config.DATA_DIR, config.NETWORK.PRETRAINED)


def get_model_name(cfg):
    name = '{model}_{num_layers}'.format(
        model=cfg.MODEL,
        num_layers=cfg.POSE_RESNET.NUM_LAYERS)
    deconv_suffix = ''.join(
        'd{}'.format(num_filters)
        for num_filters in cfg.POSE_RESNET.NUM_DECONV_FILTERS)
    full_name = '{height}x{width}_{name}_{deconv_suffix}'.format(
        height=cfg.NETWORK.IMAGE_SIZE[1],
        width=cfg.NETWORK.IMAGE_SIZE[0],
        name=name,
        deconv_suffix=deconv_suffix)

    return name, full_name


if __name__ == '__main__':
    import sys
    gen_config(sys.argv[1])
