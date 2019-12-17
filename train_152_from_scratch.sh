#!/bin/bash

set -eux
DATA_DIR=$PWD
LOG_DIR=$PWD/log
MODEL_DIR=$PWD/output
FUSION="fusion"
FUND=''
# parsing command line arguments:
while [[ $# > 0 ]]
do
key="$1"

case $key in
    -h|--help)
    echo "Usage: toolkit-execute [run_options]"
    echo "Options:"
    # echo "  -c|--cfg <config> - which configuration file to use (default NONE)"
    echo "  -d|--dataDir <path> - directory path to input files (default NONE)"
    echo "  -l|--logDir <path> - directory path to save the log files (default \$PWD)"
    echo "  -m|--modelDir <path> - directory path to save the model files (default NONE)"
    exit 1
    ;;
    # -c|--cfg)
    # CFG_NAME="$2"
    # CFG=$PWD/experiments/mixed/resnet50/pseudo_label/${CFG_NAME}.yaml
    # shift # pass argument
    # ;;
    -d|--dataDir)
    DATA_DIR="$2"
    shift # pass argument
    ;;
    -l|--logDir)
    LOG_DIR="$2"
    shift # pass argument
    ;;
    -m|--modelDir)
    MODEL_DIR="$2"
    shift # pass argument
    ;;
    -i|--inliers)
    INLIERS=$2
    shift # pass argument
    ;;
    -r|--reproj-thre)
    REPROJ_THRE=$2
    shift # pass argument
    ;;
    -f|--confidence-thre)
    CONFIDENCE_THRE=$2
    shift # pass argument
    ;;
    -t|--ransac)
    IF_RANSAC=$2
    shift # pass argument
    ;;
    -p|--repeats)
    REPEATS=$2
    shift # pass argument
    ;;
    -n|--nofusion)
    FUSION="nofusion"
    # shift # pass argument
    ;;
    --fund)
    FUND=fund$2_
    shift # pass argument
    ;;
    # *)
    # EXTRA_ARGS="$EXTRA_ARGS $1"
    # ;;
esac
shift # past argument or value
done

# train from scratch
MPII_CFG_NAME=140e_8batch
MPII_CFG=$PWD/experiments/mpii/resnet152/${MPII_CFG_NAME}.yaml

echo "train on mpii ---------------"
python run/pose2d/train.py --cfg ${MPII_CFG} --dataDir ${DATA_DIR} --logDir ${LOG_DIR} --modelDir ${MODEL_DIR} --on-server-cluster --iteration 1

echo "test 3D --------------"
python run/test/test_triangulate.py --cfg ${MPII_CFG} --dataDir ${DATA_DIR} --logDir ${LOG_DIR} --modelDir ${MODEL_DIR}  --heatmap ${MODEL_DIR}/mpii/multiview_pose_resnet_152/${MPII_CFG_NAME}/heatmaps_locations_validation_multiview_h36m.h5

echo "generate heatmaps of training set -------------------"
python run/pose2d/valid_trainset.py --cfg ${MPII_CFG} --dataDir ${DATA_DIR} --logDir ${LOG_DIR} --modelDir ${MODEL_DIR} --model-file ${MODEL_DIR}/mpii/multiview_pose_resnet_152/${MPII_CFG_NAME}/final_state.pth.tar

echo "generate pseudo label of training set -------------------"
if [ $IF_RANSAC -eq 1 ]; then
    python run/test/test_pseudo_label.py --net-layers 152 --inliers ${INLIERS} --reproj-thre ${REPROJ_THRE} --confidence-thre ${CONFIDENCE_THRE} --ransac --use-reproj --loop --cfg ${DATA_DIR}/experiments/multiview_h36m/test/test_pseudo_label.yaml --dataDir ${DATA_DIR} --logDir ${LOG_DIR} --modelDir ${MODEL_DIR} --heatmap ${MODEL_DIR}/mpii/multiview_pose_resnet_152/${MPII_CFG_NAME}/heatmaps_locations_train_multiview_h36m.h5
else
    python run/test/test_pseudo_label.py --net-layers 152 --inliers ${INLIERS} --reproj-thre ${REPROJ_THRE} --confidence-thre ${CONFIDENCE_THRE} --loop --cfg ${DATA_DIR}/experiments/multiview_h36m/test/test_pseudo_label.yaml --dataDir ${DATA_DIR} --logDir ${LOG_DIR} --modelDir ${MODEL_DIR} --heatmap ${MODEL_DIR}/mpii/multiview_pose_resnet_152/${MPII_CFG_NAME}/heatmaps_locations_train_multiview_h36m.h5
fi

# train with pseudo labels
CFG_NAME=320_${FUND}${FUSION}_resume_pseudo_${INLIERS}_${REPROJ_THRE}_${CONFIDENCE_THRE}_${IF_RANSAC}_pseudo_label
CFG=$PWD/experiments/mixed/resnet152/pseudo_label/$CFG_NAME.yaml

if [ $REPEATS -eq 1 ]; then
    echo "train iteration 1 ---------------"
    python run/pose2d/train.py --cfg ${CFG} --dataDir ${DATA_DIR} --logDir ${LOG_DIR} --modelDir ${MODEL_DIR} --on-server-cluster --iteration 1

    echo "test 3D --------------"
    python run/test/test_triangulate.py --cfg ${CFG} --dataDir ${DATA_DIR} --logDir ${LOG_DIR} --modelDir ${MODEL_DIR}  --heatmap ${MODEL_DIR}/mixed/multiview_pose_resnet_152/${CFG_NAME}/heatmaps_locations_validation_multiview_h36m.h5
else
    ITERATION=1
    until [ $ITERATION -gt $REPEATS ]; do
        echo "train iteration $ITERATION ---------------"
        python run/pose2d/train.py --cfg ${CFG} --dataDir ${DATA_DIR} --logDir ${LOG_DIR} --modelDir ${MODEL_DIR} --on-server-cluster --iteration $ITERATION

        echo "test 3D --------------"
        python run/test/test_triangulate.py --cfg ${CFG} --dataDir ${DATA_DIR} --logDir ${LOG_DIR} --modelDir ${MODEL_DIR}  --heatmap ${MODEL_DIR}/mixed/multiview_pose_resnet_152/${CFG_NAME}/heatmaps_locations_validation_multiview_h36m.h5

        # Generate Pseudo labels for next iteration

        echo "generate heatmaps of training set -------------------"
        python run/pose2d/valid_trainset.py --cfg ${CFG} --dataDir ${DATA_DIR} --logDir ${LOG_DIR} --modelDir ${MODEL_DIR} --model-file ${MODEL_DIR}/mixed/multiview_pose_resnet_152/${CFG_NAME}/final_state.pth.tar

        echo "generate pseudo label of training set -------------------"
        if [ $IF_RANSAC -eq 1 ]; then
            python run/test/test_pseudo_label.py --net-layers 152 --inliers ${INLIERS} --reproj-thre ${REPROJ_THRE} --confidence-thre ${CONFIDENCE_THRE} --ransac --use-reproj --loop --cfg ${DATA_DIR}/experiments/multiview_h36m/test/test_pseudo_label.yaml --dataDir ${DATA_DIR} --logDir ${LOG_DIR} --modelDir ${MODEL_DIR} --heatmap ${MODEL_DIR}/mixed/multiview_pose_resnet_152/${CFG_NAME}/heatmaps_locations_train_multiview_h36m.h5
        else
            python run/test/test_pseudo_label.py --net-layers 152 --inliers ${INLIERS} --reproj-thre ${REPROJ_THRE} --confidence-thre ${CONFIDENCE_THRE} --loop --cfg ${DATA_DIR}/experiments/multiview_h36m/test/test_pseudo_label.yaml --dataDir ${DATA_DIR} --logDir ${LOG_DIR} --modelDir ${MODEL_DIR} --heatmap ${MODEL_DIR}/mixed/multiview_pose_resnet_152/${CFG_NAME}/heatmaps_locations_train_multiview_h36m.h5
        fi

        let ITERATION=ITERATION+1
    done
fi
