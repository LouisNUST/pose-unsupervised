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
from dataset.mpii_compatible import MPIIDatasetCompatible
from dataset.coco_compatible import COCODatasetCompatible


class COCOMPIIDatasetCompatible(JointsDatasetCompatible):

    def __init__(self, cfg, image_set, is_train, transform=None, pseudo_label_path='', no_distortion=False):
        super().__init__(cfg, image_set, is_train, transform)
        coco = COCODatasetCompatible(cfg, image_set, is_train, transform)
        mpii = MPIIDatasetCompatible(cfg, image_set, is_train, transform)
        self.coco_size = len(coco.db)
        self.db = coco.db + mpii.db

        self.grouping = coco.grouping + self.reindex_mpii_group(mpii.grouping, 
            start_frame=len(coco.db))

        self.group_size = len(self.grouping)
        self.dataset_type = 'coco_mpii'
        self.coco_group_size = len(coco.grouping)
        self.mpii_group_size = len(mpii.grouping)

        # Data Augmentaion
        self.aug_param_dict = {'mpii':{'scale_factor': cfg.DATASET.MPII_SCALE_FACTOR,
                                       'rotation_factor': cfg.DATASET.MPII_ROT_FACTOR,
                                       'flip': cfg.DATASET.MPII_FLIP},
                               'coco':{'scale_factor': cfg.DATASET.COCO_SCALE_FACTOR,
                                       'rotation_factor': cfg.DATASET.COCO_ROT_FACTOR,
                                       'flip': cfg.DATASET.COCO_FLIP}}

        self.pseudo_label = False
        self.no_distortion = False

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
