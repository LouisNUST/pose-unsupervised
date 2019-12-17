Unsupervised 3D pose estimation on H36M

# Quick start
## Installation
1. Clone this repo, and we'll call the directory that you cloned pose.pytorch as ${POSE_ROOT}
2. Install dependencies.
3. Download pytorch imagenet pretrained models. Please download them under ${POSE_ROOT}/models, and make them look like this:

   ```
   ${POSE_ROOT}/models
   └── pytorch
       └── imagenet
           ├── resnet152-b121ed2d.pth
           ├── resnet50-19c8e357.pth
           └── mobilenet_v2.pth.tar
   ```
   They can be downloaded from the following link:
   https://1drv.ms/f/s!AjX41AtnTHeThyJfayggVZSd0M6P
   

4. Init output(training model output directory) and log(tensorboard log directory) directory.

   ```
   mkdir ouput 
   mkdir log
   ```

   and your directory tree should like this

   ```
   ${POSE_ROOT}
   ├── data
   ├── experiments
   ├── lib
   ├── log
   ├── models
   ├── output
   ├── README.md
   ├── requirements.txt
   ```

## Data preparation
**For MPII data**, please download from [MPII Human Pose Dataset](http://human-pose.mpi-inf.mpg.de/), the original annotation files are matlab's format. We have converted to json format, you also need download them from [OneDrive](https://1drv.ms/f/s!AhIXJn_J-blWzyhpADBbpJRusuT0).
Extract them under {POSE_ROOT}/data, and make them look like this:

```
${POSE_ROOT}
|-- data
|-- |-- MPII
    |-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   |-- valid.json
        |-- images
            |-- 000001163.jpg
            |-- 000003072.jpg
```

If you zip the image files into a single zip file, you should organize the data like this:

```
${POSE_ROOT}
|-- data
`-- |-- MPII
    `-- |-- annot
        |   |-- gt_valid.mat
        |   |-- test.json
        |   |-- train.json
        |   |-- trainval.json
        |   `-- valid.json
        `-- images.zip
            `-- images
                |-- 000001163.jpg
                |-- 000003072.jpg
```



**For Human36M data**, please download them from [BaiduPan](https://pan.baidu.com/s/1V65vwuAtTUd5Wy5IDlGiaw)
Extract them under {POSE_ROOT}/data, and make them look like this:

```
${POSE_ROOT}
|-- data
|-- |-- h36m
    |-- |-- annot
        |   |-- h36m_train.pkl
        |   |-- h36m_validation.pkl
        |-- images
            |-- s_01_act_02_subact_01_ca_01 
            |-- s_01_act_02_subact_01_ca_02
```

If you zip the image files into a single zip file, you should organize the data like this:
```
${POSE_ROOT}
|-- data
`-- |-- h36m
    `-- |-- annot
        |   |-- h36m_train.pkl
        |   |-- h36m_validation.pkl
        `-- images.zip
            |-- images
                |-- s_01_act_02_subact_01_ca_01
                |-- s_01_act_02_subact_01_ca_02
```

**For COCO data**, please download them from COCO official website
Extract them under {POSE_ROOT}/data, and make them look like this:

```
${POSE_ROOT}
|-- data
|-- |-- coco
    |-- |-- annot
        |   |-- person_keypoints_train2017.json
        |   |-- person_keypoints_val2017.json
        |-- images
            |-- train2017
                |-- 000000000009.jpg
                |-- 000000000025.jpg
            |-- val2017
                |-- 000000000139.jpg
                |-- 000000000285.jpg
```

If you zip the image files into a single zip file, you should organize the data like this:
```
${POSE_ROOT}
|-- data
|-- |-- coco
    |-- |-- annot
        |   |-- person_keypoints_train2017.json
        |   |-- person_keypoints_val2017.json
        |-- images
            |-- train2017.zip
                |-- train2017
                    |-- 000000000009.jpg
                    |-- 000000000025.jpg
            |--val2017.zip
                |-- val2017
                    |-- 000000000139.jpg
                    |-- 000000000285.jpg
```


## 2D Training and Testing
### Training: MPII with data augmentation. Testing: MPII
```
python run/pose2d/train.py --cfg experiments/mpii/resnet50/140e_32batch.yaml
python run/pose2d/valid.py \
    --cfg experiments/mpii/test/mpii_test_50.yaml \
    --flip-test \
    --model-file output/mpii/multiview_pose_resnet_50/140e_32batch/final_state.pth.tar
```

### Training: MPII without data augmentation. Testing: MPII
```
python run/pose2d/train.py --cfg experiments/mpii/resnet50/140e_32batch_noaug.yaml
python run/pose2d/valid.py \
    --cfg experiments/mpii/test/mpii_test_50.yaml \
    --flip-test \
    --model-file output/mpii/multiview_pose_resnet_50/140e_32batch_noaug/final_state.pth.tar
```

### Training: MPII + Pseudo Label. Testing: MultiviewH36M
```
python run/pose2d/train.py --cfg experiments/mixed/resnet50/pseudo_label/256_dist_nofusion_resume_pseudo_3_10_0.7_1_pseudo_label.yaml
python run/pose2d/valid.py \
    --cfg experiments/multiview_h36m/test/h36m_multiview_test50.yaml \
    --flip-test \
    --model-file output/mixed/multiview_pose_resnet_50/256_dist_nofusion_resume_pseudo_3_10_0.7_1_pseudo_label/final_state.pth.tar \
    --save-all-preds
```

### Train on Miscrosoft server cluster
Able to resume from unexpected interruption. Take care not to resume from previous training process (with the same cfg name).
```
python run/pose2d/train.py --cfg experiments/mpii/resnet50/140e_32batch.yaml --on-server-cluster
```

### Results on MPII val
| Arch | rank | rkne | rhip | lhip | lkne | lank | root | thorax | upper neck | head top | rwri | relb | rsho | lsho | lelb | lwri | mean |
| ---- | ---- | ---- | ---- | ---- | ---- | ---- | ---- | ------ | ---------- | -------- | ---- | ---- | ---- | ---- | ---- | ---- | ---- |
| /w Aug | 0.786 | 0.843 | 0.880 | 0.881 | 0.830 | 0.800 | 0.922 | 0.984 | 0.977 | 0.965 | 0.838 | 0.888 | 0.944 | 0.956 | 0.890 | 0.832 | 0.893 |
| /wo Aug | 0.652 | 0.730 | 0.817 | 0.818 | 0.725 | 0.673 | 0.878 | 0.967 | 0.966 | 0.944 | 0.718 | 0.806 | 0.903 | 0.911 | 0.799 | 0.722 | 0.822 |

## 3D Reconstruction
### MPII Baseline
#### Triangulate
```
python run/test/test_triangulate.py \
    --cfg experiments/mpii/resnet50/140e_32batch.yaml \
    --heatmap output/mpii/multiview_pose_resnet_50/140e_32batch/heatmaps_locations_valid_multiview_h36m.h5
```
#### RPSM
```
python run/test/generate_data_for_rpsm.py \
    --cfg experiments/mpii/resnet50/140e_32batch.yaml \
    --heatmap output/mpii/multiview_pose_resnet_50/140e_32batch/heatmaps_locations_valid_multiview_h36m.h5
python run/test/test_rpsm.py \
    --cfg experiments/mpii/resnet50/140e_32batch.yaml
```

## 2D + 3D Together
To run the whole process together or repeatedly (train pseudo 2d, test 3D MPJPE, generate 2d pseudo labels of H36M training set)
```
./train.sh -i 3 -r 10 -f 0.6 --ransac 1 --repeats 1 --nofusion --fund 5
```

### Ablation on Pseudo Label 
Based on model mpii + aug

| Arch | RANSAC inliers | proj thresh (px) | confidence thresh | PCKh@0.5 | visible percent |
| -- | -- | -- | -- | -- | -- |
| pseudo label 0 | - | - | - | 0.904 | 1 |
| pseudo label 1 | 3 | 10 | 0.7 | 0.967 | 0.90 |

### Results on Multiview H36M
| Arch | Triangulate | RPSM | 2D (16j) | 2D (15j) | PCKh@0.4 | PCKh@0.3 | PCKh@0.2 | PCKh@0.1 |
| -- | -- | -- | -- | -- | -- | -- | -- | -- |
| mpii | 130.21 (16j) | 177.15 (16j) | 0.856 | 0.875 | 0.807 | 0.658 | 0.416 | 0.137 |
| mpii(Aug) | 109.17 (16j) | 150.12 (16j) | 0.912 | 0.923 | 0.870 | 0.746 | 0.515 | 0.180 |
| mpii(Aug) + pseudo #0 | 95.81 (16j) | 155.94 (16j) | 0.909 | 0.929 | 0.876 | 0.747 | 0.509 | 0.174 |
| mpii(Aug) + pseudo #0(Aug) | 84.55 (16j) | - | 0.913 | 0.933 | 0.881 | 0.753 | 0.513 | 0.181 |
| mpii(Aug) + pseudo #1 (dist) | -- | -- | -- | 0.957 | 0.912 | 0.794 | 0.550 | 0.191 |
| mpii(Aug) + pseudo #1 + fund5 (parallel) | -- | -- | -- | 0.961 | 0.916 | 0.799 | 0.559 | 0.198 |
| mpii(Aug) + pseudo #1 + fund5 (dist) | -- | -- | -- | 0.961 | 0.917 | 0.801 | 0.558 | 0.197 |
| mpii(Aug) + pseudo #1 + fund5 (dist) no weight | -- | -- | -- | 0.958 | 0.914 | 0.800 | 0.562 | 0.201 |
| mixed | 38.20 (16j) | 30.08 (17j) | -- | 0.977 | 0.967 | 0.945 | 0.872 | 0.544 |
| mixed + Agg | 35.96 (16j) | 31.29 (17j) | 0.948 | 0.946 | -- | -- | -- | -- |
