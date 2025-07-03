# DyMVR: Dynamic masked multi-scale voxel representation for self-supervised 3D object detection

## Installation

### Docker Image
docker build -t dymvr_test .

### Docker Container
1. xhost local:root && docker run -it -e SDL_VIDEODRIVER=x11 -e DISPLAY=$DISPLAY --env='DISPLAY' --gpus 'all,"capabilities=compute,utility,graphics"' --ipc host --privileged --name dymvr --network host -p 8080:8081 -v /tmp/.X11-unix:/tmp/.X11-unix:rw -v "your-path" dymvr_test /bin/bash

2. git clone https://github.com/ho-tae/DyMVR.git

3. python setup.py devlop

4. voxel_layer.cpython-37m-x86_64-linux-gnu.so file -> move ./pcdet/models/backbones_3d/util/

5. Please follow [GETTING_START](docs/GETTING_STARTED.md) to train or evaluate the models

6. python train.py --cfg_file tools/cfgs/dymvr_mae_ws10.yaml

