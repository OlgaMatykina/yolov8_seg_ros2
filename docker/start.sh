#!/bin/bash

orange=`tput setaf 3`
reset_color=`tput sgr0`

export ARCH=`uname -m`

cd "$(dirname "$0")"
root_dir=$PWD 
cd $root_dir

echo "Running on ${orange}${ARCH}${reset_color}"

if [ "$ARCH" == "x86_64" ] 
then
    ARGS="--ipc host --gpus all -e NVIDIA_DRIVER_CAPABILITIES=all"
elif [ "$ARCH" == "aarch64" ] 
then
    ARGS="--runtime nvidia"
else
    echo "Arch ${ARCH} not supported"
    exit
fi

xhost +
docker run -it -d \
        $ARGS \
        --env="DISPLAY=$DISPLAY" \
        --env="QT_X11_NO_MITSHM=1" \
        --privileged \
        --name yolov8_seg_dds \
        --net=host \
        --ipc=host \
        --pid=host \
        -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
        -v `pwd`/../yolov8_seg_ros2:/home/docker_semseg/colcon_ws/src/yolov8_seg_ros2:rw \
        -v `pwd`/../yolov8_seg_interfaces:/home/docker_semseg/colcon_ws/src/yolov8_seg_interfaces:rw \
        -v `pwd`/../bags:/home/docker_semseg/bags:rw \
        ${ARCH}_ros2/yolov8_seg_dds:latest
xhost -
