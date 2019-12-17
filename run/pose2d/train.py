# ------------------------------------------------------------------------------
# multiview.pose3d.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import torch
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import _init_paths
from core.config import config
from core.config import update_config
from core.config import update_dir
from core.config import get_model_name
from core.loss import JointsMSELoss, FundamentalLoss, MILoss, ViewMILoss, JointsMILoss, HeatmapMILoss
from core.function import train
from core.function import validate
from utils.utils import get_optimizer
from utils.utils import save_checkpoint, load_checkpoint
from utils.utils import create_logger, get_training_loader, get_valid_loader
import dataset
import models

import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn.parallel
import torch.multiprocessing as mp
import logging
from collections import OrderedDict
import time
import signal

def signal_handler(sig, frame):
    assert 0, 'This process is killed manually!'

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    parser.add_argument(
        '--frequent',
        help='frequency of logging',
        default=config.PRINT_FREQ,
        type=int)
    parser.add_argument('--gpus', help='gpus', type=str)
    parser.add_argument('--workers', help='num of dataloader workers', type=int)

    parser.add_argument(
        '--modelDir', help='model directory', type=str, default='')
    parser.add_argument('--logDir', help='log directory', type=str, default='')
    parser.add_argument(
        '--dataDir', help='data directory', type=str, default='')
    parser.add_argument(
        '--data-format', help='data format', type=str, default='')
    parser.add_argument(
        '--on-server-cluster', help='if training on cluster', action='store_true')
    parser.add_argument(
        '--iteration', help='the kth times of training', type=int, choices=range(1,10), default=1)
    parser.add_argument(
        '--no-distortion', help='wheter use no distortion data', action='store_true')
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
    if args.on_server_cluster:
        config.TRAIN.ON_SERVER_CLUSTER = args.on_server_cluster
    if args.no_distortion:
        config.DATASET.NO_DISTORTION = args.no_distortion


def main():
    args = parse_args()
    reset_config(config, args)

    cudnn.benchmark = config.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = config.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = config.CUDNN.ENABLED

    gpus = [int(i) for i in config.GPUS.split(',')]
    num_gpus = len(gpus)
    assert num_gpus <= torch.cuda.device_count(), 'available GPUS: {}, designate GPUs: {}'.format(torch.cuda.device_count(), num_gpus)

    # Shared Dataset
    # logger.info('=> loading dataset')
    # normalize = transforms.Normalize(
    #     mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    # train_dataset = eval('dataset.' + config.DATASET.TRAIN_DATASET)(
    #     config, config.DATASET.TRAIN_SUBSET, True,
    #     transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ]),
    #     config.DATASET.PSEUDO_LABEL_PATH,
    #     config.DATASET.NO_DISTORTION)
    # valid_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
    #     config, config.DATASET.TEST_SUBSET, False,
    #     transforms.Compose([
    #         transforms.ToTensor(),
    #         normalize,
    #     ]),
    #     '',
    #     config.DATASET.NO_DISTORTION)
    print('=> initializing multiple processes')
    mp.spawn(main_worker, nprocs=num_gpus, args=(args, config, num_gpus))


def main_worker(rank, args, config, num_gpus):
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend='nccl', rank=rank, world_size=num_gpus)
    print('Rank: {} finished initializing, PID: {}'.format(rank, os.getpid()))

    if rank == 0:
        logger, final_output_dir, tb_log_dir = create_logger(
            config, args.cfg, 'train')
        logger.info(pprint.pformat(args))
        logger.info(pprint.pformat(config))
    else:
        final_output_dir = None
        tb_log_dir = None

    # Gracefully kill all subprocesses by command <'kill subprocess 0'>
    signal.signal(signal.SIGTERM, signal_handler)
    if rank == 0:
        logger.info('Rank {} has registerred signal handler'.format(rank))

    # device in current process
    device = torch.device('cuda', rank)

    backbone_model = eval('models.' + config.BACKBONE_MODEL + '.get_pose_net')(
        config, is_train=True)
    base_model = eval('models.' + config.MODEL + '.get_multiview_pose_net')(
        backbone_model, config)

    model_dict = OrderedDict()
    model_dict['base_model'] = base_model.to(device)

    if config.LOSS.USE_GLOBAL_MI_LOSS:
        global_discriminator = models.discriminator.GlobalDiscriminator(config)
        model_dict['global_discriminator'] = global_discriminator.to(device)
    if config.LOSS.USE_LOCAL_MI_LOSS:
        local_discriminator = models.discriminator.LocalDiscriminator(config)
        model_dict['local_discriminator'] = local_discriminator.to(device)
    if config.LOSS.USE_DOMAIN_TRANSFER_LOSS:
        domain_discriminator = models.discriminator.DomainDiscriminator(config)
        model_dict['domain_discriminator'] = domain_discriminator.to(device)
    if config.LOSS.USE_VIEW_MI_LOSS:
        view_discriminator = models.discriminator.ViewDiscriminator(config)
        model_dict['view_discriminator'] = view_discriminator.to(device)
    if config.LOSS.USE_JOINTS_MI_LOSS:
        joints_discriminator = models.discriminator.JointsDiscriminator(config)
        model_dict['joints_discriminator'] = joints_discriminator.to(device)
    if config.LOSS.USE_HEATMAP_MI_LOSS:
        heatmap_discriminator = models.discriminator.HeatmapDiscriminator(config)
        model_dict['heatmap_discriminator'] = heatmap_discriminator.to(device)

    # copy model files and print model config
    if rank == 0:
        this_dir = os.path.dirname(__file__)
        shutil.copy2(
            os.path.join(this_dir, '../../lib/models', config.MODEL + '.py'),
            final_output_dir)
        shutil.copy2(args.cfg, final_output_dir)
        logger.info(pprint.pformat(model_dict['base_model']))
        if config.LOSS.USE_GLOBAL_MI_LOSS:
            logger.info(pprint.pformat(model_dict['global_discriminator']))
        if config.LOSS.USE_LOCAL_MI_LOSS:
            logger.info(pprint.pformat(model_dict['local_discriminator']))
        if config.LOSS.USE_DOMAIN_TRANSFER_LOSS:
            logger.info(pprint.pformat(model_dict['domain_discriminator']))
        if config.LOSS.USE_VIEW_MI_LOSS:
            logger.info(pprint.pformat(model_dict['view_discriminator']))
        if config.LOSS.USE_JOINTS_MI_LOSS:
            logger.info(pprint.pformat(model_dict['joints_discriminator']))
        if config.LOSS.USE_HEATMAP_MI_LOSS:
            logger.info(pprint.pformat(model_dict['heatmap_discriminator']))
        if config.LOSS.USE_GLOBAL_MI_LOSS or config.LOSS.USE_LOCAL_MI_LOSS \
            or config.LOSS.USE_DOMAIN_TRANSFER_LOSS or config.LOSS.USE_VIEW_MI_LOSS \
            or config.LOSS.USE_JOINTS_MI_LOSS or config.LOSS.USE_HEATMAP_MI_LOSS:
            shutil.copy2(
                os.path.join(this_dir, '../../lib/models', 'discriminator.py'),
                final_output_dir)

    # tensorboard writer
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    } if rank == 0 else None

    # dump_input = torch.rand(
    #     (config.TRAIN.BATCH_SIZE, 3,
    #      config.NETWORK.IMAGE_SIZE[1], config.NETWORK.IMAGE_SIZE[0]))
    # writer_dict['writer'].add_graph(model, (dump_input,))

    # first resume, then parallel
    for key in model_dict.keys():
        model_dict[key] = torch.nn.parallel.DistributedDataParallel(model_dict[key], device_ids=[rank], output_device=rank)
        # one by one
        dist.barrier()

    # get optimizer
    optimizer_dict = {}
    optimizer_base_model = get_optimizer(config, model_dict['base_model'])
    optimizer_dict['base_model'] = optimizer_base_model
    if config.LOSS.USE_GLOBAL_MI_LOSS:
        optimizer_global = get_optimizer(config, model_dict['global_discriminator'], is_discriminator=True)
        optimizer_dict['global_discriminator'] = optimizer_global
    if config.LOSS.USE_LOCAL_MI_LOSS:
        optimizer_local = get_optimizer(config, model_dict['local_discriminator'], is_discriminator=True)
        optimizer_dict['local_discriminator'] = optimizer_local
    if config.LOSS.USE_DOMAIN_TRANSFER_LOSS:
        optimizer_domain = get_optimizer(config, model_dict['domain_discriminator'], is_discriminator=True)
        optimizer_dict['domain_discriminator'] = optimizer_domain
    if config.LOSS.USE_VIEW_MI_LOSS:
        optimizer_view = get_optimizer(config, model_dict['view_discriminator'], is_discriminator=True)
        optimizer_dict['view_discriminator'] = optimizer_view
    if config.LOSS.USE_JOINTS_MI_LOSS:
        optimizer_joints = get_optimizer(config, model_dict['joints_discriminator'], is_discriminator=True)
        optimizer_dict['joints_discriminator'] = optimizer_joints
    if config.LOSS.USE_HEATMAP_MI_LOSS:
        optimizer_heatmap = get_optimizer(config, model_dict['heatmap_discriminator'], is_discriminator=True)
        optimizer_dict['heatmap_discriminator'] = optimizer_heatmap

    # resume
    if config.TRAIN.RESUME:
        assert config.TRAIN.RESUME_PATH != '', 'You must designate a path for config.TRAIN.RESUME_PATH, rank: {}'.format(rank)
        if rank == 0:
            logger.info('=> loading model from {}'.format(config.TRAIN.RESUME_PATH))
        # !!! map_location must be cpu, otherwise a lot memory will be allocated on gpu:0.
        state_dict = torch.load(config.TRAIN.RESUME_PATH, map_location=torch.device('cpu'))
        if 'state_dict_base_model' in state_dict:
            if rank == 0:
                logger.info('=> new loading mode')
            for key in model_dict.keys():
                # delete params of the aggregation layer
                if key == 'base_model' and not config.NETWORK.AGGRE:
                    for param_key in list(state_dict['state_dict_base_model'].keys()):
                        if 'aggre_layer' in param_key:
                            state_dict['state_dict_base_model'].pop(param_key)
                model_dict[key].module.load_state_dict(state_dict['state_dict_' + key])
        else:
            if rank == 0:
                logger.info('=> old loading mode')
            # delete params of the aggregation layer
            if not config.NETWORK.AGGRE:
                for param_key in list(state_dict.keys()):
                    if 'aggre_layer' in param_key:
                        state_dict.pop(param_key)
            model_dict['base_model'].module.load_state_dict(state_dict)

    # Traing on server cluster, resumed when interrupted
    start_epoch = config.TRAIN.BEGIN_EPOCH
    if config.TRAIN.ON_SERVER_CLUSTER:
        start_epoch, model_dict, optimizer_dict, loaded_iteration = load_checkpoint(model_dict, optimizer_dict,
                                                        final_output_dir)
        if args.iteration < loaded_iteration:
            # this training process shold be skipped
            if rank == 0:
                logger.info('=> Skipping training iteration #{}'.format(args.iteration))
            return

    # lr schedulers have different starting points yet share same decay strategy.
    lr_scheduler_dict = {}
    for key in optimizer_dict.keys():
        lr_scheduler_dict[key] = torch.optim.lr_scheduler.MultiStepLR(
            optimizer_dict[key], config.TRAIN.LR_STEP, config.TRAIN.LR_FACTOR)

    # torch.set_num_threads(8)

    criterion_dict = {}
    criterion_dict['mse_weights'] = JointsMSELoss(
        use_target_weight=config.LOSS.USE_TARGET_WEIGHT).to(device)
    criterion_dict['mse'] = torch.nn.MSELoss(reduction='mean').to(device)

    if config.LOSS.USE_FUNDAMENTAL_LOSS:
        criterion_dict['fundamental'] = FundamentalLoss(config)

    if config.LOSS.USE_GLOBAL_MI_LOSS or config.LOSS.USE_LOCAL_MI_LOSS:
        criterion_dict['mutual_info'] = MILoss(config, model_dict)

    if config.LOSS.USE_DOMAIN_TRANSFER_LOSS:
        criterion_dict['bce'] = torch.nn.BCELoss().to(device)

    if config.LOSS.USE_VIEW_MI_LOSS:
        criterion_dict['view_mi'] = ViewMILoss(config, model_dict)

    if config.LOSS.USE_JOINTS_MI_LOSS:
        criterion_dict['joints_mi'] = JointsMILoss(config, model_dict)

    if config.LOSS.USE_HEATMAP_MI_LOSS:
        criterion_dict['heatmap_mi'] = HeatmapMILoss(config, model_dict)

    # Data loading code
    if rank == 0:
        logger.info('=> loading dataset')
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    train_dataset = eval('dataset.' + config.DATASET.TRAIN_DATASET)(
        config, config.DATASET.TRAIN_SUBSET, True,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        config.DATASET.PSEUDO_LABEL_PATH,
        config.DATASET.NO_DISTORTION)
    valid_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
        config, config.DATASET.TEST_SUBSET, False,
        transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]),
        '',
        config.DATASET.NO_DISTORTION)
    # Debug ##################
    # print('len of mixed dataset:', len(train_dataset))
    # print('len of multiview h36m dataset:', len(valid_dataset))

    train_loader, train_sampler = get_training_loader(train_dataset, config)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=config.TEST.BATCH_SIZE,  # no need to multiply len(gpus)
        shuffle=False,
        num_workers=int(config.WORKERS / num_gpus),
        pin_memory=False)

    best_perf = 0
    best_model = False

    dist.barrier()

    for epoch in range(start_epoch, config.TRAIN.END_EPOCH):
        for lr_scheduler in lr_scheduler_dict.values():
            lr_scheduler.step()

        train_sampler.set_epoch(epoch)

        train(config, train_loader, model_dict, criterion_dict, optimizer_dict, epoch,
                final_output_dir, writer_dict, rank)
        perf_indicator = validate(config, valid_loader, valid_dataset, model_dict,
                                  criterion_dict, final_output_dir, writer_dict, rank)

        if rank == 0:
            if perf_indicator > best_perf:
                best_perf = perf_indicator
                best_model = True
            else:
                best_model = False

            logger.info('=> saving checkpoint to {}'.format(final_output_dir))

            save_dict = {
                'epoch': epoch + 1,
                'model': get_model_name(config),
                'perf': perf_indicator,
                'iteration': args.iteration
            }
            model_state_dict = {}
            optimizer_state_dict = {}
            for key, model in model_dict.items():
                model_state_dict['state_dict_' + key] = model.module.state_dict()
                optimizer_state_dict['optimizer_' + key] = optimizer_dict[key].state_dict()
            save_dict.update(model_state_dict)
            save_dict.update(optimizer_state_dict)
            save_checkpoint(save_dict, best_model, final_output_dir)
        dist.barrier()

    if rank == 0:
        final_model_state_file = os.path.join(final_output_dir,
                                              'final_state.pth.tar')
        logger.info('saving final model state to {}'.format(final_model_state_file))
        torch.save(model_state_dict, final_model_state_file)
        writer_dict['writer'].close()

    print('Rank {} exit'.format(rank))


if __name__ == '__main__':
    main()
