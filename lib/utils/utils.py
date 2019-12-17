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
import logging
import time
from pathlib import Path

import torch
import torch.optim as optim


from core.config import get_model_name
from torch.utils.data.distributed import  DistributedSampler
import torch.distributed as dist


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.TRAIN_DATASET
    model, _ = get_model_name(cfg)
    cfg_name_list = os.path.basename(cfg_name).split('.')[:-1]
    cfg_name = '.'.join(cfg_name_list)

    final_output_dir = root_output_dir / dataset / model / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / model / \
        (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


def get_optimizer(cfg, model, is_discriminator=False):
    optimizer = None
    if not is_discriminator and cfg.TRAIN.FIX_BACKBONE:
        for param in model.resnet.parameters():
            param.requires_grad = False
        updated_param = model.aggre_layer.parameters()
    else:
        updated_param = model.parameters()

    if cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = optim.SGD(
            updated_param,
            lr=cfg.TRAIN.LR if not is_discriminator else cfg.TRAIN.LR_DISCRIMINATOR,
            momentum=cfg.TRAIN.MOMENTUM,
            weight_decay=cfg.TRAIN.WD,
            nesterov=cfg.TRAIN.NESTEROV
        )
    elif cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = optim.Adam(
            updated_param,
            lr=cfg.TRAIN.LR if not is_discriminator else cfg.TRAIN.LR_DISCRIMINATOR
        )

    return optimizer

def load_checkpoint(model_dict, optimizer_dict, output_dir, filename='checkpoint.pth.tar'):
    file = os.path.join(output_dir, filename)
    if os.path.isfile(file):
        checkpoint = torch.load(file, map_location=torch.device('cpu'))
        start_epoch = checkpoint['epoch']
        for key, model in model_dict.items():
            model.module.load_state_dict(checkpoint['state_dict_' + key])
            optimizer.load_state_dict(checkpoint['optimizer_' + key])
        if 'iteration' not in checkpoint:
            iteration = 1
        else:
            iteration = checkpoint['iteration']
        logger = logging.getLogger()
        logger.info('=> load checkpoint {} (epoch {})'
              .format(file, start_epoch))

        return start_epoch, model_dict, optimizer_dict, iteration

    else:
        logger.info('=> no checkpoint found at {}'.format(file))
        return 0, model_dict, optimizer_dict, 1


def save_checkpoint(states, is_best, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))
    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'],
                   os.path.join(output_dir, 'model_best.pth.tar'))


def get_training_loader(trainset, config):
    if trainset.dataset_type == 'mixed' and config.DATASET.IF_SAMPLE:
        assert 0, 'weightd distributed sampler not implemented so far!'
        from torch.utils.data.sampler import  WeightedRandomSampler
        # h36m + mpii
        weights = [config.DATASET.H36M_WEIGHT for _ in range(trainset.h36m_group_size)]
        weights += [config.DATASET.MPII_WEIGHT for _ in range(trainset.mpii_group_size)]
        assert len(weights) == len(trainset)
        my_sampler = WeightedRandomSampler(weights, num_samples=len(trainset))
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=config.TRAIN.BATCH_SIZE,  # distributed, no need to multiply len_gpus
            sampler= my_sampler,
            num_workers=config.WORKERS,
            pin_memory=False)
    else:
        my_sampler = DistributedSampler(trainset)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=False,
            sampler=my_sampler,
            num_workers=int(config.WORKERS / dist.get_world_size()),
            pin_memory=False)
    return train_loader, my_sampler

def get_valid_loader(testset, config):
    my_sampler = DistributedSampler(testset)
    valid_loader = torch.utils.data.DataLoader(
            testset,
            batch_size=config.TEST.BATCH_SIZE,
            shuffle=False,
            sampler=my_sampler,
            num_workers=int(config.WORKERS / dist.get_world_size()),
            pin_memory=False)
    return valid_loader, my_sampler

