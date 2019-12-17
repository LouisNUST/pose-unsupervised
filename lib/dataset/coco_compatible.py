# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os
import pickle
from collections import defaultdict
from collections import OrderedDict

import json_tricks as json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from dataset.joints_dataset_compatible import JointsDatasetCompatible
# from nms.nms import oks_nms


logger = logging.getLogger()


class COCODatasetCompatible(JointsDatasetCompatible):
    def __init__(self, cfg, image_set, is_train, transform=None, pseudo_label_path='', no_distortion=False):
        super().__init__(cfg, image_set, is_train, transform)
        self.actual_joints = {
            0: 'nose',
            1: 'leye',
            2: 'reye',
            3: 'lear',
            4: 'rear',
            5: 'lsho',
            6: 'rsho',
            7: 'lelb',
            8: 'relb',
            9: 'lwri',
            10: 'rwri',
            11: 'lhip',
            12: 'rhip',
            13: 'lkne',
            14: 'rkne',
            15: 'lank',
            16: 'rank'
        }
        self.pseudo_label = False
        self.no_distortion = False
        self.subset += '2017'  # plus year for coco

        self.use_gt_bbox = True
        self.oks_thre = cfg.TEST.OKS_THRE
        self.in_vis_thre = cfg.TEST.IN_VIS_THRE
        self.aspect_ratio = self.image_size[0] * 1.0 / self.image_size[1]
        self.coco = COCO(self._get_ann_file_keypoint())

        # deal with class names
        cats = [cat['name']
                for cat in self.coco.loadCats(self.coco.getCatIds())]
        self.classes = ['__background__'] + cats
        logger.info('=> classes: {}'.format(self.classes))
        self.num_classes = len(self.classes)
        self._class_to_ind = dict(zip(self.classes, range(self.num_classes)))
        self._class_to_coco_ind = dict(zip(cats, self.coco.getCatIds()))
        self._coco_ind_to_class_ind = dict([(self._class_to_coco_ind[cls],
                                             self._class_to_ind[cls])
                                            for cls in self.classes[1:]])

        # load image file names
        self.image_set_index = self._load_image_set_index()
        self.num_images = len(self.image_set_index)
        logger.info('=> coco num_images: {}'.format(self.num_images))

        # load db
        self.db = self._get_db()
        self.u2a_mapping = self.get_mapping()
        super().do_mapping()

        self.grouping = self.get_group(self.db)
        self.group_size = len(self.grouping)
        logger.info('=> coco load {} samples'.format(self.group_size * 4))

        self.dataset_type = 'coco'

        # Data Augmentation
        self.aug_param_dict = {'coco':{'scale_factor': cfg.DATASET.COCO_SCALE_FACTOR,
                                       'rotation_factor': cfg.DATASET.COCO_ROT_FACTOR,
                                       'flip': cfg.DATASET.COCO_FLIP}}

    def get_mapping(self):
        union_keys = list(self.union_joints.keys())
        union_values = list(self.union_joints.values())

        mapping = {k: '*' for k in union_keys}
        for k, v in self.actual_joints.items():
            if v in union_values:
                idx = union_values.index(v)
                key = union_keys[idx]
                mapping[key] = k
        return mapping

    def get_group(self, db):
        coco_grouping = []
        coco_length = len(db)
        for i in range(coco_length // 4):
            mini_group = []
            for j in range(4):
                index = i * 4 + j
                mini_group.append(index)
            coco_grouping.append(mini_group)
        return coco_grouping

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

    def _get_ann_file_keypoint(self):
        """ data/coco/annot/person_keypoints_train2017.json """
        prefix = 'person_keypoints' \
            if 'test' not in self.subset else 'image_info'
        return os.path.join(self.root, 'coco', 'annot',
                            prefix + '_' + self.subset + '.json')

    def _load_image_set_index(self):
        """ image id: int """
        image_ids = self.coco.getImgIds()
        return image_ids

    def _get_db(self):
        gt_db = self._load_coco_keypoint_annotations()
        # if self.is_train or self.use_gt_bbox:
        #     # use ground truth bbox
        #     gt_db = self._load_coco_keypoint_annotations()
        # else:
        #     use bbox from detection
        #     gt_db = self._load_coco_person_detection_results()
        return gt_db

    def _load_coco_keypoint_annotations(self):
        """ ground truth bbox and keypoints """
        gt_db = []
        for index in self.image_set_index:
            gt_db.extend(self._load_coco_keypoint_annotation_kernal(index))
        return gt_db

    def _load_coco_keypoint_annotation_kernal(self, index):
        """
        coco ann: [u'segmentation', u'area', u'iscrowd', u'image_id', u'bbox', u'category_id', u'id']
        iscrowd:
            crowd instances are handled by marking their overlaps with all categories to -1
            and later excluded in training
        bbox:
            [x1, y1, w, h]
        :param index: coco image id
        :return: db entry
        """
        im_ann = self.coco.loadImgs(index)[0]
        width = im_ann['width']
        height = im_ann['height']

        annIds = self.coco.getAnnIds(imgIds=index, iscrowd=False)
        objs = self.coco.loadAnns(annIds)

        # sanitize bboxes
        valid_objs = []
        for obj in objs:
            x, y, w, h = obj['bbox']
            x1 = np.max((0, x))
            y1 = np.max((0, y))
            x2 = np.min((width - 1, x1 + np.max((0, w - 1))))
            y2 = np.min((height - 1, y1 + np.max((0, h - 1))))
            if obj['area'] > 0 and x2 >= x1 and y2 >= y1:
                # obj['clean_bbox'] = [x1, y1, x2, y2]
                obj['clean_bbox'] = [x1, y1, x2-x1, y2-y1]
                valid_objs.append(obj)
        objs = valid_objs

        rec = []
        for obj in objs:
            cls = self._coco_ind_to_class_ind[obj['category_id']]
            if cls != 1:
                continue

            # ignore objs without keypoints annotation
            if max(obj['keypoints']) == 0:
                continue

            joints_3d = np.zeros((17, 3), dtype=np.float)
            joints_3d_vis = np.zeros((17, 3), dtype=np.float)
            for ipt in range(17):
                joints_3d[ipt, 0] = obj['keypoints'][ipt * 3 + 0]
                joints_3d[ipt, 1] = obj['keypoints'][ipt * 3 + 1]
                joints_3d[ipt, 2] = 0
                t_vis = obj['keypoints'][ipt * 3 + 2]
                if t_vis > 1:
                    t_vis = 1
                joints_3d_vis[ipt, 0] = t_vis
                joints_3d_vis[ipt, 1] = t_vis
                joints_3d_vis[ipt, 2] = 0

            center, scale = self._box2cs(obj['clean_bbox'][:4])
            rec.append({
                'image': self.image_path_from_index(index),
                'center': center,
                'scale': scale,
                'joints_2d': joints_3d[:, :2],
                'joints_3d': joints_3d,
                'joints_vis': joints_3d_vis,
                'source': 'coco'
            })

        return rec

    def _box2cs(self, box):
        x, y, w, h = box[:4]

        center = np.zeros((2), dtype=np.float)
        center[0] = x + w * 0.5
        center[1] = y + h * 0.5

        if w > self.aspect_ratio * h:
            h = w * 1.0 / self.aspect_ratio
        elif w < self.aspect_ratio * h:
            w = h * self.aspect_ratio
        scale = np.array(
            [w * 1.0 / 200, h * 1.0 / 200],
            dtype=np.float)
        if center[0] != -1:
            scale = scale * 1.25

        return center, scale

    def image_path_from_index(self, index):
        """ example: train2017 or train2017.zip / 000000119993.jpg """
        file_name = '%012d.jpg' % index
        if '2014' in self.subset:
            file_name = 'COCO_%s_' % self.subset + file_name

        prefix = 'test2017' if 'test' in self.subset else self.subset

        data_name = os.path.join(prefix + '.zip@', prefix) if self.data_format == 'zip' else prefix

        image_path = os.path.join(data_name, file_name)

        return image_path

    # # need double check this API and classes field
    # def evaluate(self, cfg, preds, output_dir, all_boxes, img_path,
    #              *args, **kwargs):
    #     res_folder = os.path.join(output_dir, 'results')
    #     if not os.path.exists(res_folder):
    #         os.makedirs(res_folder)
    #     res_file = os.path.join(
    #         res_folder, 'keypoints_%s_results.json' % self.subset)

    #     # person x (keypoints)
    #     _kpts = []
    #     for idx, kpt in enumerate(preds):
    #         _kpts.append({
    #             'keypoints': kpt,
    #             'center': all_boxes[idx][0:2],
    #             'scale': all_boxes[idx][2:4],
    #             'area': all_boxes[idx][4],
    #             'score': all_boxes[idx][5],
    #             'image': int(img_path[idx][-16:-4])
    #         })
    #     # image x person x (keypoints)
    #     kpts = defaultdict(list)
    #     for kpt in _kpts:
    #         kpts[kpt['image']].append(kpt)

    #     # rescoring and oks nms
    #     num_joints = 17
    #     in_vis_thre = self.in_vis_thre
    #     oks_thre = self.oks_thre
    #     oks_nmsed_kpts = []
    #     for img in kpts.keys():
    #         img_kpts = kpts[img]
    #         for n_p in img_kpts:
    #             box_score = n_p['score']
    #             kpt_score = 0
    #             valid_num = 0
    #             for n_jt in range(0, num_joints):
    #                 t_s = n_p['keypoints'][n_jt][2]
    #                 if t_s > in_vis_thre:
    #                     kpt_score = kpt_score + t_s
    #                     valid_num = valid_num + 1
    #             if valid_num != 0:
    #                 kpt_score = kpt_score / valid_num
    #             # rescoring
    #             n_p['score'] = kpt_score * box_score
    #         keep = oks_nms([img_kpts[i] for i in range(len(img_kpts))],
    #                        oks_thre)
    #         if len(keep) == 0:
    #             oks_nmsed_kpts.append(img_kpts)
    #         else:
    #             oks_nmsed_kpts.append([img_kpts[_keep] for _keep in keep])

    #     self._write_coco_keypoint_results(
    #         oks_nmsed_kpts, res_file)
    #     if 'test' not in self.subset:
    #         info_str = self._do_python_keypoint_eval(
    #             res_file, res_folder)
    #         name_value = OrderedDict(info_str)
    #         return name_value, name_value['AP']
    #     else:
    #         return {'Null': 0}, 0

    # def _write_coco_keypoint_results(self, keypoints, res_file):
    #     data_pack = [{'cat_id': self._class_to_coco_ind[cls],
    #                   'cls_ind': cls_ind,
    #                   'cls': cls,
    #                   'ann_type': 'keypoints',
    #                   'keypoints': keypoints
    #                   }
    #                  for cls_ind, cls in enumerate(self.classes) if not cls == '__background__']

    #     results = self._coco_keypoint_results_one_category_kernel(data_pack[0])
    #     logger.info('=> Writing results json to %s' % res_file)
    #     with open(res_file, 'w') as f:
    #         json.dump(results, f, sort_keys=True, indent=4)
    #     try:
    #         json.load(open(res_file))
    #     except Exception:
    #         content = []
    #         with open(res_file, 'r') as f:
    #             for line in f:
    #                 content.append(line)
    #         content[-1] = ']'
    #         with open(res_file, 'w') as f:
    #             for c in content:
    #                 f.write(c)

    # def _coco_keypoint_results_one_category_kernel(self, data_pack):
    #     cat_id = data_pack['cat_id']
    #     keypoints = data_pack['keypoints']
    #     cat_results = []

    #     for img_kpts in keypoints:
    #         if len(img_kpts) == 0:
    #             continue

    #         _key_points = np.array([img_kpts[k]['keypoints']
    #                                 for k in range(len(img_kpts))])
    #         key_points = np.zeros(
    #             (_key_points.shape[0], 17 * 3), dtype=np.float)

    #         for ipt in range(17):
    #             key_points[:, ipt * 3 + 0] = _key_points[:, ipt, 0]
    #             key_points[:, ipt * 3 + 1] = _key_points[:, ipt, 1]
    #             key_points[:, ipt * 3 + 2] = _key_points[:, ipt, 2]  # keypoints score.

    #         result = [{'image_id': img_kpts[k]['image'],
    #                    'category_id': cat_id,
    #                    'keypoints': list(key_points[k]),
    #                    'score': img_kpts[k]['score'],
    #                    'center': list(img_kpts[k]['center']),
    #                    'scale': list(img_kpts[k]['scale'])
    #                    } for k in range(len(img_kpts))]
    #         cat_results.extend(result)

    #     return cat_results

    # def _do_python_keypoint_eval(self, res_file, res_folder):
    #     coco_dt = self.coco.loadRes(res_file)
    #     coco_eval = COCOeval(self.coco, coco_dt, 'keypoints')
    #     coco_eval.params.useSegm = None
    #     coco_eval.evaluate()
    #     coco_eval.accumulate()
    #     coco_eval.summarize()
    #     stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

    #     info_str = []
    #     for ind, name in enumerate(stats_names):
    #         info_str.append((name, coco_eval.stats[ind]))

    #     eval_file = os.path.join(
    #         res_folder, 'keypoints_%s_results.pkl' % self.subset)

    #     with open(eval_file, 'wb') as f:
    #         pickle.dump(coco_eval, f, pickle.HIGHEST_PROTOCOL)
    #     logger.info('=> coco eval results saved to %s' % eval_file)

    #     return info_str