
import pickle
import numpy as np
import cv2
import argparse
import numpy as np
from pathlib import Path
import os

import _init_paths
import dataset
from core.config import config
from core.config import update_config
from multiviews.cameras import project_pose, camera_to_world_frame, unfold_camera_param
from utils import zipreader

def parse_args():
    parser = argparse.ArgumentParser(
        description='Test Pseudo Labels')
    # Required Param
    parser.add_argument(
        '--cfg', help='experiment configure file name', required=True, type=str)
    args, rest = parser.parse_known_args()
    update_config(args.cfg)

    args = parser.parse_args()
    return args
args = parse_args()


test_dataset = eval('dataset.' + config.DATASET.TEST_DATASET)(
    config, 'train', False)


for img_idx in range(len(test_dataset.db)):
    db_rec = test_dataset.db[img_idx]

    image_dir = 'images.zip@' if test_dataset.data_format == 'zip' else ''
    image_file = os.path.join(test_dataset.root, db_rec['source'], image_dir, 'images',
                          db_rec['image'])

    if test_dataset.data_format == 'zip':
        data_numpy = zipreader.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    else:
        data_numpy = cv2.imread(
            image_file, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)

    # read camera
    camera = db_rec['camera']
    R, T, f, c, k, p = unfold_camera_param(camera, avg_f=False)
    mtx = np.array(
            [[f[0], 0, c[0]], [0, f[1], c[1]], [0, 0, 1]], dtype=float)
    dist = np.array((k[0],k[1],p[0],p[1],k[2]))

    # undistort
    h, w = data_numpy.shape[:2]  # [h, w, 3]
    newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))
    dst = cv2.undistort(data_numpy, mtx, dist, None, newcameramtx)
    x,y,w,h = roi
    dst = dst[y:y+h, x:x+w]

    # Save undistorted image
    new_img_path = os.path.join(test_dataset.root, db_rec['source'], 'images', db_rec['image'])
    new_dir_path = os.path.dirname(new_img_path)

    if not os.path.exists(new_dir_path):
        os.makedirs(new_dir_path)

    cv2.imwrite(new_img_path, dst)

    if img_idx % 10000 == 0:
        print(img_idx)

