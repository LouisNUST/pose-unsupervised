
import pickle
import numpy as np
import cv2
# from lib.multiviews.cameras import project_pose, camera_to_world_frame, unfold_camera_param
import argparse
# from pymvg.multi_camera_system import build_example_system
import numpy as np
import itertools
import logging
from pathlib import Path
import yaml
import os
import h5py

# import json_tricks as json
# from pycocotools.coco import COCO
# import matplotlib.pyplot as plt
# from PIL import Image


# coco = COCO('data/coco/annot/person_keypoints_val2017.json')
# image_ids = coco.getImgIds()
# print(len(image_ids))
# index = image_ids[2]  # load image 0

# im_ann = coco.loadImgs(index)[0]
# width = im_ann['width']
# height = im_ann['height']

# annIds = coco.getAnnIds(imgIds=index, iscrowd=False)
# objs = coco.loadAnns(annIds)

# file_name = im_ann['file_name']
# file_path = '/data/cihai/Code/pose_multiview/data/coco/images/val2017/{}'.format(file_name)
# I = Image.open(file_path)

# fig = plt.figure()

# ax = fig.add_subplot(111)
# ax.set_axis_off()
# ax.imshow(I)
# coco.showAnns(objs)

# save_path = os.path.join('test.jpg')
# plt.savefig(save_path, bbox_inches='tight', transparent='True', pad_inches=0)
# plt.close(fig)


# import torch
# low_features = torch.ones((2, 2048, 6, 6))
# high_features = torch.ones((2, 2048, 64, 64))
# meta = {}
# meta['joints_2d_transformed'] = torch.from_numpy(np.random.randint(0, 256, size=(2, 16, 2)))

# def _sample_indices(batch_size, n_locs, n_samples, replacement=True):
#     """
#     Return: indices of [batch_size, n_samples], each row ranges from 0 to n_locs
#     """
#     weights = torch.ones(batch_size, n_locs, dtype=torch.float)
#     idx = torch.multinomial(weights, n_samples, replacement=replacement)
#     return idx

# def sample_locations(enc, n_samples):
#     '''Randomly samples locations from localized features.

#     Used for saving memory.

#     Args:
#         enc: [N, L, C]
#         n_samples: int 
#     Returns:
#         [N, n_samples, C]

#     '''
#     batch_size, n_locs = enc.shape[:2]
#     idx = _sample_indices(batch_size, n_locs, n_samples, replacement=True)  # [N, n_samples]
#     adx = torch.arange(0, batch_size).long()
#     enc = enc[adx[:, None], idx]
#     return enc


# def extract_local_pairs(low_features, high_features, target_weight, meta):
#     """
#     Image specific now
#     low_features: [N, 2048, 6, 6]  after conv preprocess
#     high_features: [N, 2048, 64, 64]
#     target_weight: [N, 16, 1]
#     Return: [N, K, 64, 64], [N, 256, 64, 64]
#     """
#     assert low_features.shape[:2] == high_features.shape[:2]
#     batch_size, c, h_high, w_high = high_features.size()
#     _, _, 64, w_low = low_features.size()
#     assert h_high == w_high
#     assert h_low == w_low
#     if h_low == 6 and h_high == 64:
#         factor = 8
#         low_patches = low_features.view(batch_size, c, -1)  # [N, 2048, 6*6]
#         # size, stride = 3, 1
#         # low_patches = low_features.unfold(2, size, stride).unfold(3, size, stride)  # [N, 2048, 6, 6, 3, 3]
#         # _, _, h_num, w_num, _, _ = low_patches.size()
#         # assert h_num == 6, 'number of patches {} not equal 6'.format(h_num)
#         # low_patches = low_patches.permute(0, 4, 5, 1, 2, 3).contiguous().view(batch_size, -1, h_num*w_num)  # [N, 9*2048, 6*6]

#         # extract positive pairs
#         # low_indices = [0] + [i for i in range(h_num)] + [int(h_num)-1]
#         # low_indices = torch.tensor(low_indices).view(-1, 1).repeat(1, factor).view(-1)  # [64]
#         # low_indices_h = low_indices[:, None]
#         # low_indices_w = low_indices[None, :]
#         # low_pos = low_patches[:, :, low_indices_h, low_indices_w]  # [N, 9*2048, 64, 64]
#         # high_pos = high_features
#         pos_indices_high = _sample_indices(batch_size, h_high, 100*2).view(batch_size, 100, 2)  # [N, K, 2]
#         gt_locations = meta['joints_2d_transformed'] / 4 + 0.5  # in order (w, h)
#         gt_locations = gt_locations.to(dtype=torch.long).clamp_(min=0, max=h_high-1)
#         pos_indices_high = torch.cat([gt_locations, pos_indices_high], dim=1)  # [N, K+16, 2]
#         pos_indices_low = pos_indices_high / factor
#         pos_indices_low = pos_indices_low.to(dtype=torch.long) - 1
#         pos_indices_low.clamp_(min=0, max=h_low - 1)
#         print(pos_indices_low.min(), pos_indices_low.max())
#         print(pos_indices_high.min(), pos_indices_high.max())
#         pos_indices_high = pos_indices_high[:, :, 1] * w_high + pos_indices_high[:, :, 0]  # [N, K+16]
#         pos_indices_low = pos_indices_low[:, :, 1] * w_low + pos_indices_low[:, :, 0]  # [N, K+16]
#         pos_batch_indices = torch.arange(batch_size, dtype=torch.long)[:, None]

#         high_pos = high_features.permute(0, 2, 3, 1).view(batch_size, -1, c)  # [N, 64*64, 2048], take care that img_h is in front of img_w
#         high_pos = high_pos[pos_batch_indices, pos_indices_high].transpose(1, 2)  # [N, 2048, K+16]
#         low_pos = low_patches.transpose(1, 2)  # [N, 6*6, 2048]
#         low_pos = low_pos[pos_batch_indices, pos_indices_low].transpose(1, 2)  # [N, 2048, K+16]

#         # extract negative pairs
#         high_neg = high_pos.unsqueeze(2).expand(-1, -1, 5, -1).contiguous().view(batch_size, c, -1) # [N, 2048, Q*(K+16)]
#         def _neg_batch_indices(n):
#             x = [[i for i in range(n) if i != k] for k in range(n)]
#             return torch.tensor(list(itertools.chain(*x)), dtype=torch.long)
#         neg_batch_indices = _neg_batch_indices(batch_size)  # [N*(N-1)]
#         low_neg = low_patches[neg_batch_indices, ...].view(batch_size, batch_size-1, low_patches.size(1), -1)  # [N, N-1, 2048, 6*6]
#         low_neg = low_neg.permute(0, 1, 3, 2).contiguous().view(batch_size, -1, low_patches.size(1))  # [N, (N-1)*6*6, 2048]
#         low_neg = sample_locations(low_neg, high_neg.size(-1))  # [N, Q*(K+16), 2048]
#         low_neg = low_neg.transpose(1, 2)  # [N, 2048, Q*(K+16)], [N, C, L]
#         assert low_neg.size(2) == high_neg.size(2)

#         # calculate score
#         pos_score = torch.sum(high_pos * low_pos, dim=1)  # [N, K+16]
#         neg_score = torch.sum(high_neg * low_neg, dim=1)  # [N, Q*(K+16)]
#     else:
#         assert 0, 'not implemented feature map size, low:{}, high:{}'.format(low_features.size(2),
#             high_features.size(2))

#     return pos_score, neg_score


# # pos_score, neg_score= extract_local_pairs(low_features, high_features, 1, meta)

# # print(pos_score.shape)
# # print(neg_score.shape)

# a = torch.tensor(1)
# print(isinstance(1, torch.Tensor))



import torch
import torch.distributed as dist
import torch.utils.data.distributed
import torch.nn.parallel
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch import autograd
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import time
import signal
from PIL import Image


# from lib.utils import zipreader
# image_file = '/data/cihai/Code/pose_multiview/data/h36m/images.zip@/images/s_01_act_02_subact_01_ca_01/s_01_act_02_subact_01_ca_01_000001.jpg'
# img = zipreader.imread(image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)  # [h, w, 3] B,G,R
# img = img[:, :, ::-1]  # [h, w, 3] R,G,B

# to_pil = transforms.ToPILImage()
# color_jitter = transforms.ColorJitter(brightness=(0.7,3.0), contrast=(0.5,2.0), saturation=(0.5, 2.0), hue=0.2)

# pil_img = to_pil(img)
# # pil_img = TF.adjust_brightness(pil_img, 0.7)  # 0.7 - 3.0
# # pil_img = TF.adjust_contrast(pil_img, 5.0)  # 0.5 - 2.0
# # pil_img = TF.adjust_hue(pil_img, -0.1)  # -0.2 -0.2 or -0.1 ~0.1
# # pil_img = TF.adjust_saturation(pil_img, 3)  # 0.5 ~ 2.0
# pil_img = color_jitter(pil_img)


# pil_img.save('test.jpg')

a = torch.arange(24).view(3, 4, 2)
print(a)
b = a.flip((2,))
print(b)


# if __name__ == '__main__':
#     main()
    