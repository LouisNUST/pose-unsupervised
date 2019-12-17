# ------------------------------------------------------------------------------
# multiview.3d.pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

import pickle
import numpy as np
import itertools

from pymvg.camera_model import CameraModel
from pymvg.multi_camera_system import MultiCameraSystem
from multiviews.cameras import unfold_camera_param


def build_multi_camera_system(cameras, no_distortion=False):
    """
    Build a multi-camera system with pymvg package for triangulation

    Args:
        cameras: list of camera parameters
    Returns:
        cams_system: a multi-cameras system
    """
    pymvg_cameras = []
    for (name, camera) in cameras:
        R, T, f, c, k, p = unfold_camera_param(camera, avg_f=False)
        camera_matrix = np.array(
            [[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1]], dtype=float)
        proj_matrix = np.zeros((3, 4))
        proj_matrix[:3, :3] = camera_matrix
        distortion = np.array([k[0], k[1], p[0], p[1], k[2]])
        distortion.shape = (5,)
        T = -np.matmul(R, T)
        M = camera_matrix.dot(np.concatenate((R, T), axis=1))
        camera = CameraModel.load_camera_from_M(
            M, name=name, distortion_coefficients=None if no_distortion else distortion)
        pymvg_cameras.append(camera)
    return MultiCameraSystem(pymvg_cameras)


def triangulate_one_point(camera_system, points_2d_set):
    """
    Triangulate 3d point in world coordinates with multi-views 2d points

    Args:
        camera_system: pymvg camera system
        points_2d_set: list of structure (camera_name, point2d)
    Returns:
        points_3d: 3x1 point in world coordinates
    """
    points_3d = camera_system.find3d(points_2d_set)
    return points_3d


def triangulate_poses(camera_params, poses2d, joints_vis=None, no_distortion=False):
    """
    Triangulate 3d points in world coordinates of multi-view 2d poses
    by interatively calling $triangulate_one_point$

    Args:
        camera_params: a list of camera parameters, each corresponding to
                       one prediction in poses2d
        poses2d: [N, k, 2], len(cameras) == N
        joints_vis: [N, k], only visible joints participate in triangulatioin
    Returns:
        poses3d: ndarray of shape N/nviews x k x 3
    """
    nviews = 4
    njoints = poses2d.shape[1]
    ninstances = len(camera_params) // nviews
    if joints_vis is not None:
        assert np.all(joints_vis.shape == poses2d.shape[:2])
    else:
        joints_vis = np.ones((poses2d.shape[0], poses2d.shape[1]))

    poses3d = []
    for i in range(ninstances):
        cameras = []
        for j in range(nviews):
            camera_name = 'camera_{}'.format(j)
            cameras.append((camera_name, camera_params[i * nviews + j]))
        camera_system = build_multi_camera_system(cameras, no_distortion)

        pose3d = np.zeros((njoints, 3))
        for k in range(njoints):
            points_2d_set = []

            for j in range(nviews):
                if joints_vis[i * nviews + j, k]:
                    camera_name = 'camera_{}'.format(j)
                    points_2d = poses2d[i * nviews + j, k, :]
                    points_2d_set.append((camera_name, points_2d))
            if len(points_2d_set) < 2:
                continue
            pose3d[k, :] = triangulate_one_point(camera_system, points_2d_set).T
        poses3d.append(pose3d)
    return np.array(poses3d)


def ransac(poses2d, camera_params, joints_vis, config):
    """
    An group is accepted only if support inliers are not less 
    than config.PSEUDO_LABEL.NUM_INLIERS, i.e. num of Trues
    in a 4-view group is not less than config.PSEUDO_LABEL.NUM_INLIERS
    Param:
        poses2d: [N, 16, 2]
        camera_params: a list of [N]
        joints_vis: [N, 16], only visible joints participate in triangulation
    Return:
        res_vis: [N, 16]
    """
    nviews = 4
    njoints = poses2d.shape[1]
    ninstances = len(camera_params) // nviews

    res_vis = np.zeros_like(joints_vis)
    for i in range(ninstances):
        cameras = []
        for j in range(nviews):
            camera_name = 'camera_{}'.format(j)
            cameras.append((camera_name, camera_params[i * nviews + j]))
        camera_system = build_multi_camera_system(cameras, config.DATASET.NO_DISTORTION)

        for k in range(njoints):
            points_2d_set = []

            for j in range(nviews):
                camera_name = 'camera_{}'.format(j)
                # select out visible points from all 4 views
                if joints_vis[i * nviews + j, k]:
                    points_2d = poses2d[i * nviews + j, k, :]
                    points_2d_set.append((camera_name, points_2d))

            # points < 2, invalid instance, abandon samples of 1 view
            if len(points_2d_set) < 2:
                continue

            best_inliers = []
            best_error = 10000
            for points_pair in itertools.combinations(points_2d_set, 2):
                point_3d = triangulate_one_point(camera_system, list(points_pair)).T
                in_thre = []
                mean_error = 0
                for j in range(nviews):
                    point_2d_proj = camera_system.find2d('camera_{}'.format(j), point_3d)
                    error = np.linalg.norm(point_2d_proj - poses2d[i * nviews + j, k, :])
                    if error < config.PSEUDO_LABEL.REPROJ_THRE:
                        in_thre.append(j)
                        mean_error += error
                num_inliers = len(in_thre)
                if num_inliers < config.PSEUDO_LABEL.NUM_INLIERS:
                    continue
                mean_error /= num_inliers
                # update best candidate
                if num_inliers > len(best_inliers):
                    best_inliers = in_thre
                    best_error = mean_error
                elif num_inliers == len(best_inliers):
                    if mean_error < best_error:
                        best_inliers = in_thre
                        best_error = mean_error
            for idx_view in best_inliers:
                res_vis[i * nviews + idx_view, k] = 1
    return res_vis


def reproject_poses(poses2d, camera_params, joints_vis, no_distortion=False):
    """
    Triangulate 3d points in world coordinates of multi-view 2d poses
    by interatively calling $triangulate_one_point$

    Args:
        camera_params: a list of camera parameters, each corresponding to
                       one prediction in poses2d
        poses2d: [N, k, 2], len(cameras) == N
        joints_vis: [N, k], only visible joints participate in triangulatioin
    Returns:
        proj_2d: ndarray of shape [N, k, 2]
        res_vis: [N, k]
    """
    nviews = 4
    njoints = poses2d.shape[1]
    ninstances = len(camera_params) // nviews
    assert np.all(joints_vis.shape == poses2d.shape[:2])
    proj_2d = np.zeros_like(poses2d)  # [N, 16, 2]
    res_vis = np.zeros_like(joints_vis)

    for i in range(ninstances):
        cameras = []
        for j in range(nviews):
            camera_name = 'camera_{}'.format(j)
            cameras.append((camera_name, camera_params[i * nviews + j]))
        camera_system = build_multi_camera_system(cameras, no_distortion)

        for k in range(njoints):
            points_2d_set = []

            for j in range(nviews):
                if joints_vis[i * nviews + j, k]:
                    camera_name = 'camera_{}'.format(j)
                    points_2d = poses2d[i * nviews + j, k, :]
                    points_2d_set.append((camera_name, points_2d))
            if len(points_2d_set) < 2:
                continue
            point_3d = triangulate_one_point(camera_system, points_2d_set).T

            for j in range(nviews):
                point_2d_proj = camera_system.find2d('camera_{}'.format(j), point_3d)
                proj_2d[i * nviews + j, k, :] = point_2d_proj
                res_vis[i * nviews + j, k] = 1
    return proj_2d, res_vis