
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

my_file_path = "./data/h36m/annot/h36m_validation.pkl"
# old_file_path = r"D:\Dataset\h36m\h36m_validation_nodistortion.pkl"

with open(my_file_path, 'rb') as f:
    my_dataset = pickle.load(f)

for idx, item in enumerate(my_dataset):
    pose3d_camera = item['joints_3d']
    camera = item['camera']
    R, T, f, c, k, p = unfold_camera_param(camera, avg_f=False)
    camera_matrix = np.array(
            [[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1]], dtype=float)
    pose2d, _ = cv2.projectPoints(pose3d_camera, (0, 0, 0), (0, 0, 0), camera_matrix, None)
    pose2d = pose2d.squeeze()  # [17, 2]
    item['joints_2d'] = pose2d  # replace original pose2d
    if idx % 10000 == 0:
        print(idx)

new_file_path = "./data/h36m/annot/h36m_validation_nodistortion.pkl"
with open(new_file_path, 'wb') as f:
    pickle.dump(my_dataset, f)



