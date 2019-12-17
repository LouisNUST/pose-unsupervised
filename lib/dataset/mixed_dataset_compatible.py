# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from core.config import config
from dataset.joints_dataset_compatible import JointsDatasetCompatible
from dataset.multiview_h36m_compatible import MultiViewH36MCompatible
from dataset.mpii_compatible import MPIIDatasetCompatible


class MixedDatasetCompatible(JointsDatasetCompatible):

    def __init__(self, cfg, image_set, is_train, transform=None, pseudo_label_path='', no_distortion=False):
        super().__init__(cfg, image_set, is_train, transform)
        h36m = MultiViewH36MCompatible(cfg, image_set, is_train, transform, pseudo_label_path, no_distortion)
        mpii = MPIIDatasetCompatible(cfg, image_set, is_train, transform)
        self.h36m_size = len(h36m.db)
        self.db = h36m.db + mpii.db

        self.grouping = h36m.grouping + self.reindex_mpii_group(mpii.grouping, 
            start_frame=len(h36m.db))

        self.group_size = len(self.grouping)
        self.dataset_type = 'mixed'
        self.h36m_group_size = len(h36m.grouping)
        self.mpii_group_size = len(mpii.grouping)

        # Data Augmentaion for mpii images
        self.aug_param_dict = {'mpii':{'scale_factor': cfg.DATASET.MPII_SCALE_FACTOR,
                                       'rotation_factor': cfg.DATASET.MPII_ROT_FACTOR,
                                       'flip': cfg.DATASET.MPII_FLIP},
                               'h36m':{'scale_factor': cfg.DATASET.H36M_SCALE_FACTOR,
                                       'rotation_factor': cfg.DATASET.H36M_ROT_FACTOR,
                                       'flip': cfg.DATASET.H36M_FLIP}}

        # only valid for h36m
        if pseudo_label_path:
            self.pseudo_label = True
        else:
            self.pseudo_label = False
        self.no_distortion = no_distortion

    def __len__(self):
        return self.group_size

    def __getitem__(self, idx):
        input, target, weight, meta = [], [], [], []
        items = self.grouping[idx]
        for item in items:
            i, t, w, m = super().__getitem__(item)
            input.append(i)
            target.append(t)
            weight.append(w)
            meta.append(m)
        return input, target, weight, meta

    def reindex_mpii_group(self, group, start_frame):
        return list(map(lambda x: [x+start_frame for x in x], group))

    # def mpii_grouping(self, db, start_frame=1):
    #     mpii_grouping = []
    #     mpii_length = len(db)
    #     for i in range(mpii_length // 4):
    #         mini_group = []
    #         for j in range(4):
    #             index = i * 4 + j
    #             mini_group.append(index + start_frame)
    #         mpii_grouping.append(mini_group)
    #     return mpii_grouping
