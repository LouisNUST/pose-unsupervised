# ------------------------------------------------------------------------------
# multiview.3d.pose.pytorch
# Copyright (c) 2018-present Microsoft
# Licensed under The Apache-2.0 License [see LICENSE for details]
# Written by Chunyu Wang (chnuwa@microsoft.com)
# ------------------------------------------------------------------------------

import numpy as np


class HumanBody(object):

    def __init__(self):
        self.skeleton = self.get_skeleton()
        self.skeleton_sorted_by_level = self.sort_skeleton_by_level(
            self.skeleton)
        self.root_idx = 6

    def get_skeleton(self):
        # mpii naming and order
        # 16 joints
        joint_names = [
            'rank', 'rkne', 'rhip', 'lhip', 'lkne', 'lank', 'root', 'thorax',
            'upper neck', 'head top', 'rwri', 'relb', 'rsho', 'lsho', 'lelb',
            'lwri'
        ]
        children = [[], [0], [1], [4], [5], [], [2,3,7], [8,12,13], [9], [],
                    [], [10], [11], [14], [15], []]

        skeleton = []
        for i in range(len(joint_names)):
            skeleton.append({
                'idx': i,
                'name': joint_names[i],
                'children': children[i]
            })
        return skeleton

    def sort_skeleton_by_level(self, skeleton):
        njoints = len(skeleton)
        level = np.zeros(njoints)

        queue = [skeleton[6]]  # root idx of human body
        while queue:
            cur = queue[0]
            for child in cur['children']:
                skeleton[child]['parent'] = cur['idx']
                level[child] = level[cur['idx']] + 1
                queue.append(skeleton[child])
            del queue[0]

        desc_order = np.argsort(level)[::-1]
        sorted_skeleton = []
        for i in desc_order:
            skeleton[i]['level'] = level[i]
            sorted_skeleton.append(skeleton[i])
        return sorted_skeleton


if __name__ == '__main__':
    hb = HumanBody()
    # print(hb.skeleton)
    print(hb.skeleton_sorted_by_level)
