#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
IMAGE=sylee/onnxim
TAG=latest
NAME=sylee_onnxim
# DATA_DIR=/data/jwhwang/data
# SSD_DIR=/mnt/ssd

docker run -it \
    --net=host \
    -v ${SCRIPT_DIR}:/workspace/ai-framework-sim \
    --shm-size=110gb \
    --name=${NAME} \
    ${IMAGE}


#    --gpus all \
#    -v ${DATA_DIR}:/workspace/data \
#    -v ${SSD_DIR}:/workspace/ssd \
