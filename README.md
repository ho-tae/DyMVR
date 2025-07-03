# DyMVR: Dynamic masked multi-scale voxel representation for self-supervised 3D object detection

## Installation

Please follow [INSTALL](docs/INSTALL.md) to install DyMVR.

## Docker

### Docker Image
docker build -t dymvr_test .

### Docker Container
xhost local:root && docker run -it -e SDL_VIDEODRIVER=x11 -e DISPLAY=$DISPLAY --env='DISPLAY' --gpus 'all,"capabilities=compute,utility,graphics"' --ipc host --privileged --name dymvr --network host -p 8080:8081 -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v "your-path" dymvr_test /bin/bash

### Getting Started

Please follow [GETTING_START](docs/GETTING_STARTED.md) to train or evaluate the models.
