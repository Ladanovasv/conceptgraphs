#!/bin/bash

DATA_PATH=$1
CODE=${2:-`pwd`/..}

if [[ ! $DATA_PATH ]]; then
    echo "Please provide path to data."
    exit 1
fi
xhost +local:
docker run -itd --rm \
           --network host \
           --env="DISPLAY=$DISPLAY" \
           --env="QT_X11_NO_MITSHM=1" \
           --privileged \
           -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
           --gpus all \
           -v $CODE:/home/docker_user/concept-graphs:rw \
           -v $DATA_PATH:/datasets/:rw \
           --name ladanova_concept_graphs \
           concept_graphs:latest
xhost -