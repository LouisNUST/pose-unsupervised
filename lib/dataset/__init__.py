# ------------------------------------------------------------------------------
# pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dataset.mpii_compatible import MPIIDatasetCompatible as mpii
# from dataset.h36m import H36MDataset as h36m
from dataset.multiview_h36m_compatible import MultiViewH36MCompatible as multiview_h36m
from dataset.mixed_dataset_compatible import MixedDatasetCompatible as mixed
from dataset.coco_compatible import COCODatasetCompatible as coco
from dataset.coco_mpii_compatible import COCOMPIIDatasetCompatible as coco_mpii
