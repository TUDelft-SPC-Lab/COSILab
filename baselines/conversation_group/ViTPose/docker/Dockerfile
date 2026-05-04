ARG PYTORCH="1.11.0"
ARG CUDA="11.3"
ARG CUDNN="8"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 6.2 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# Fix Nvidia GPG keys error
RUN rm /etc/apt/sources.list.d/cuda.list
RUN rm /etc/apt/sources.list.d/nvidia-ml.list

RUN apt-get update && apt-get install -y git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 libgl1-mesa-glx\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install xtcocotools
RUN pip install cython
RUN pip install xtcocotools

# Install MMCV
RUN pip install -U openmim
RUN mim install mmcv-full==1.5.0

# Install MMPose
RUN conda clean --all
RUN git clone https://github.com/open-mmlab/mmpose.git /mmpose
WORKDIR /mmpose
RUN mkdir -p /mmpose/data
ENV FORCE_CUDA="1"
RUN pip install -r requirements/build.txt
RUN pip install --no-cache-dir -e .

# Hotfixes needed due to not specifing the dependencies properly in the requirements.txt files
# Fix numpy binary incompatiblity with xtcocotools
# Add timm dependency
# Fix format error with yapf
RUN pip install numpy==1.23.5 timm yapf==0.40.1
WORKDIR /workspace
