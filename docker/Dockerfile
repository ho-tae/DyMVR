ARG PYTORCH="1.9.0"
ARG CUDA="11.1"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list
RUN apt-key del 7fa2af80 && apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/3bf863cc.pub

RUN apt-get update -y \
    && apt-get install build-essential libgl1-mesa-glx \
    && apt-get install -y apt-utils git curl ca-certificates bzip2 tree htop wget \
    && apt-get install -y libglib2.0-0 libsm6 libxext6 libxrender-dev bmon iotop g++ python3.7 python3.7-dev python3.7-distutils

# Install cmake v3.13.2
RUN apt-get purge -y cmake && \
    mkdir /root/temp && \
    cd /root/temp && \
    wget https://github.com/Kitware/CMake/releases/download/v3.13.2/cmake-3.13.2.tar.gz && \
    tar -xzvf cmake-3.13.2.tar.gz && \
    cd cmake-3.13.2 && \
    bash ./bootstrap && \
    make && \
    make install && \
    cmake --version && \
    rm -rf /root/temp

# Install python
RUN ln -sv /usr/bin/python3.7 /usr/bin/python
RUN wget https://bootstrap.pypa.io/get-pip.py && \
        python get-pip.py && \
        rm get-pip.py

# Install python packages
RUN PIP_INSTALL="python -m pip --no-cache-dir install" && \
    $PIP_INSTALL numpy==1.19.3 llvmlite numba

# Install torch and torchvision
# See https://pytorch.org/ for other options if you use a different version of CUDA
#RUN pip install --user torch==1.6 torchvision==0.7.0 -f https://download.pytorch.org/whl/cu102/torch_stable.html

# Install python packages
RUN PIP_INSTALL="python -m pip --no-cache-dir install" && \
    $PIP_INSTALL tensorboardX easydict pyyaml scikit-image tqdm SharedArray six

WORKDIR /root

# Install Boost geometry
RUN wget https://jaist.dl.sourceforge.net/project/boost/boost/1.68.0/boost_1_68_0.tar.gz && \
    tar xzvf boost_1_68_0.tar.gz && \
    cp -r ./boost_1_68_0/boost /usr/include && \
    rm -rf ./boost_1_68_0 && \
    rm -rf ./boost_1_68_0.tar.gz


RUN pip install mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

RUN pip uninstall -y SharedArray && \
    pip install SharedArray==3.1.0 spconv-cu113 torch-scatter==2.0.9 opencv-python tensorflow==2.4.0 waymo-open-dataset-tf-2-4-0 kornia==0.5 mmdet==2.14.0