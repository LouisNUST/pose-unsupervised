# ------------------------------------------------------------------------------
# pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import cv2
import torch

def flip_back(output_flipped, matched_parts):
    '''
    output_flipped: numpy.ndarray(batch_size, num_joints, height, width)
    '''
    assert output_flipped.ndim == 4,\
        'output_flipped should be [batch_size, num_joints, height, width]'

    output_flipped = output_flipped[:, :, :, ::-1]

    for pair in matched_parts:
        tmp = output_flipped[:, pair[0], :, :].copy()
        output_flipped[:, pair[0], :, :] = output_flipped[:, pair[1], :, :]
        output_flipped[:, pair[1], :, :] = tmp

    return output_flipped


def flip_back_th(output_flipped, matched_parts):
    """
    output_flipped: a list of torch.ndarray(batch_size, num_joints, height, width)
    """
    assert len(output_flipped) == 4
    assert output_flipped[0].dim() == 4

    output_flipped = [torch.flip(view, dims=[3]) for view in output_flipped]
    new_order = list(range(output_flipped[0].size(1)))
    for pair in matched_parts:
        new_order[pair[0]] = pair[1]
        new_order[pair[1]] = pair[0]
    new_order = torch.tensor(new_order).cuda()
    output_flipped = [torch.index_select(view, 1, new_order) for  view in output_flipped]
    return output_flipped


def fliplr_joints(joints, joints_vis, width, matched_parts):
    """
    flip coords
    """
    # Flip horizontal
    joints[:, 0] = width - joints[:, 0] - 1

    # Change left-right parts
    for pair in matched_parts:
        joints[pair[0], :], joints[pair[1], :] = \
            joints[pair[1], :], joints[pair[0], :].copy()
        joints_vis[pair[0], :], joints_vis[pair[1], :] = \
            joints_vis[pair[1], :], joints_vis[pair[0], :].copy()

    return joints*joints_vis, joints_vis


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    # for p in range(coords.shape[0]):
    #     target_coords[p, 0:2] = affine_transform(coords[p, 0:2], trans)
    target_coords[:, :2] = affine_transform(coords[:, :2], trans)
    return target_coords


def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale * 200.0
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    """
    pt: [N, 2] or [2]
    t: [2, 3]
    """
    if pt.ndim == 1:
        pt = pt[np.newaxis, ...]
    pt = np.concatenate((pt, np.ones((pt.shape[0], 1))), axis=-1)
    return np.dot(pt, t.T).squeeze()


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def crop(img, center, scale, output_size, rot=0):
    trans = get_affine_transform(center, scale, rot, output_size)

    dst_img = cv2.warpAffine(img,
                             trans,
                             (int(output_size[0]), int(output_size[1])),
                             flags=cv2.INTER_LINEAR)

    return dst_img


def generate_integral_preds_2d_th(heatmaps):
    """
    heatmaps: [N, 16, h, w]
    """
    # assert isinstance(heatmaps, torch.Tensor)
    # heatmaps = torch.div(heatmaps, torch.sum(heatmaps, dim=(2, 3), keepdim=True))

    batch_size, n_joints, h, w = heatmaps.shape
    heatmaps = heatmaps * 100  # multiply by a factor 100, avoid small number
    heatmaps = heatmaps.view(batch_size, n_joints, -1)  # [N, 16, h*w]
    heatmaps = torch.nn.functional.softmax(heatmaps, dim=-1)  # [N, 16, h*w]
    heatmaps = heatmaps.view(batch_size, n_joints, h, w)

    accu_w = heatmaps.sum(dim=2)  # [N, 16, w]
    accu_h = heatmaps.sum(dim=3)  # [N, 16, h]

    w_coordinates = torch.arange(w, dtype=torch.float32).to(device=heatmaps.get_device())
    h_coordinates = torch.arange(h, dtype=torch.float32).to(device=heatmaps.get_device())
    w_coordinates = torch.sum(accu_w * w_coordinates.view(1, 1, -1), dim=2)  # [N, 16]
    h_coordinates = torch.sum(accu_h * h_coordinates.view(1, 1, -1), dim=2)  # [N, 16]

    preds2d = torch.stack([w_coordinates, h_coordinates], dim=2)  # [N, 16, 2]
    return preds2d


def transform_back_th(cfg, joints_2d_list, meta):
    """
    Transform back preds2d
    """
    output_device = joints_2d_list[0].device
    batch_size, n_joints = joints_2d_list[0].shape[:2]
    results = []  # 4 views, [N, 16, 3]
    for p, m in zip(joints_2d_list, meta):
        # p: [N, 16, 2], m: {N}
        c = m['center'].numpy()
        s = m['scale'].numpy()
        trans = []
        for s_c, s_s in zip(c, s):
            trans.append(get_affine_transform(s_c, s_s, 0, [cfg.NETWORK.HEATMAP_SIZE[0], cfg.NETWORK.HEATMAP_SIZE[1]], inv=1))
        trans = np.stack(trans, axis=0)  # [N, 2, 3]
        trans = torch.from_numpy(trans).to(device=output_device, dtype=torch.float32)
        new_p = torch.cat((p, 
            torch.ones(batch_size, n_joints, 1).to(device=output_device)),
            dim=2)  # [N, 16, 3]

        # [N, 16, 3] @ [N, 3, 2] --> [N, 16, 2]
        transformed_p = torch.matmul(new_p, trans.transpose(2, 1))
        assert len(transformed_p) == batch_size
        results.append(transformed_p)
    return results