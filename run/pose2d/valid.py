# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint

import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.loss import JointsMSELoss, FundamentalLoss, MILoss
from core.function import validate
from utils.utils import create_logger

import dataset
import models


def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)

    args, rest = parser.parse_known_args()
    # update config
    update_config(args.cfg)

    # training
    parser.add_argument(
        '--frequent',
        help='frequency of logging',
        default=config.PRINT_FREQ,
        type=int)
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument(
        '--state',
        help='the state of model which is used to test (best or final)',
        default='best',
        type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)
    parser.add_argument('--model-file', help='model state file', type=str, default='')
    parser.add_argument(
        '--flip-test', help='use flip test', action='store_true')
    parser.add_argument(
        '--post-process', help='use post process', action='store_true')
    parser.add_argument(
        '--shift-heatmap', help='shift heatmap', action='store_true')

    # philly
    parser.add_argument(
        '--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument(
        '--dataDir', help='data directory', type=str, default='')
    parser.add_argument(
        '--data-format', help='data format', type=str, default='')
    parser.add_argument(
        '--no-distortion', help='wheter use no distortion data', action='store_true')
    parser.add_argument(
        '--save-all-preds', help='wheter save all pred results', action='store_true')

    args = parser.parse_args()

    update_dir(args.modelDir, args.logDir, args.dataDir)

    return args


def reset_config(config, args):
    if args.gpus:
        config.GPUS = args.gpus
    if args.data_format:
        config.DATASET.DATA_FORMAT = args.data_format
    if args.workers:
        config.WORKERS = args.workers
    if args.flip_test:
        config.TEST.FLIP_TEST = args.flip_test
    if args.post_process:
        config.TEST.POST_PROCESS = args.post_process
    if args.shift_heatmap:
        config.TEST.SHIFT_HEATMAP = args.shift_heatmap
    if args.model_file:
        config.TEST.MODEL_FILE = args.model_file
    if args.state:
        config.TEST.STATE = args.state
    if args.no_distortion:
        config.DATASET.NO_DISTORTION = args.no_distortion
    if args.save_all_preds:
        config.DEBUG.SAVE_ALL_PREDS = args.save_all_preds


def main():
    args = parse_args()
    reset_config(config, args)

    logger, final_output_dir, tb_log_dir = create_logger(
        config, args.cfg, 'valid')

    logger.info(pprint.pformat(args))
    logger.info(pprint.pformat(config))

    # cudnn related setting
    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    backbone_model = eval('models.' + config.BACKBONE_MODEL + '.get_pose_net')(
        config, is_train=False)

    base_model = eval('models.' + config.MODEL + '.get_multiview_pose_net')(
        backbone_model, config)

    model_dict = {}
    model_dict['base_model'] = base_model
    config.LOSS.USE_GLOBAL_MI_LOSS = False
    config.LOSS.USE_LOCAL_MI_LOSS = False
    config.LOSS.USE_FUNDAMENTAL_LOSS = False
    # if config.LOSS.USE_GLOBAL_MI_LOSS:
    #     global_discriminator = models.discriminator.GlobalDiscriminator(config)
    #     model_dict['global_discriminator'] = global_discriminator
    # if config.LOSS.USE_LOCAL_MI_LOSS:
    #     local_discriminator = models.discriminator.LocalDiscriminator(config)
    #     model_dict['local_discriminator'] = local_discriminator

    if config.TEST.MODEL_FILE:
        logger.info('=> loading model from {}'.format(config.TEST.MODEL_FILE))
        state_dict = torch.load(config.TEST.MODEL_FILE)
    else:
        model_path = 'model_best.pth.tar' if config.TEST.STATE == 'best' else 'final_state.pth.tar'
        model_state_file = os.path.join(final_output_dir, model_path)
        logger.info('=> loading model from {}'.format(model_state_file))
        state_dict = torch.load(model_state_file)
    if 'state_dict_base_model' in state_dict:
        logger.info('=> new loading mode')
        for key, model in model_dict.items():
            # delete params of the aggregation layer
            if key == 'base_model' and not config.NETWORK.AGGRE:
                for param_key in list(state_dict['state_dict_base_model'].keys()):
                    if 'aggre_layer' in param_key:
                        state_dict['state_dict_base_model'].pop(param_key)
            model_dict[key].load_state_dict(state_dict['state_dict_' + key])
    else:
        logger.info('=> old loading mode')
        # delete params of the aggregation layer
        if not config.NETWORK.AGGRE:
            for param_key in list(state_dict.keys()):
                if 'aggre_layer' in param_key:
                    state_dict.pop(param_key)
        model_dict['base_model'].load_state_dict(state_dict)

    gpus = [int(i) for i in config.GPUS.split(',')]
    for key, model in model_dict.items():
        model_dict[key] = torch.nn.DataParallel(model, device_ids=gpus).cuda()

    # define loss function (criterion) and optimizer
    criterion_dict = {}
    criterion_dict['mse_weights'] = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT).cuda()
    criterion_dict['mse'] = torch.nn.MSELoss(reduction='mean').cuda()

    # if config.LOSS.USE_FUNDAMENTAL_LOSS:
    #     criterion_dict['fundamental'] = FundamentalLoss(config)

    # if config.LOSS.USE_GLOBAL_MI_LOSS or config.LOSS.USE_LOCAL_MI_LOSS:
    #     criterion_dict['mutual_info'] = MILoss(config, model_dict)

    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    valid_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,  # training set, is_trainin=True
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        '',
        config.DATASET.NO_DISTORTION)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE * len(gpus),
        shuffle=False,
        num_workers=config.WORKERS,
        pin_memory=True)

    # evaluate on validation set
    validate(config, valid_loader, valid_dataset, model_dict, criterion_dict,
             final_output_dir, None, rank=0)


if __name__ == '__main__':
    main()
