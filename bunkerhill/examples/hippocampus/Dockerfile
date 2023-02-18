FROM ubuntu:jammy-20221101

# Sets arguments
ARG USER_ID

# Sets working directory
WORKDIR app

# Installs utilities
RUN apt update -y \
  && apt install -y curl nano vim wget

# Installs GPU
# see for template: https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/11.7.0/ubuntu2204/base/Dockerfile
ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64

ENV NVARCH x86_64

ENV CUDA_VERSION 11.7.0
ENV NV_CUDA_CUDART_VERSION 11.7.60-1
ENV NV_CUDA_COMPAT_PACKAGE cuda-compat-11-7
ENV NV_CUDA_LIB_VERSION 11.7.0-1

# see for environment variables: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/user-guide.html#environment-variables-oci-spec
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV NVIDIA_REQUIRE_CUDA "cuda>=11.7 brand=tesla,driver>=450,driver<451 brand=tesla,driver>=470,driver<471 brand=unknown,driver>=470,driver<471 brand=nvidia,driver>=470,driver<471 brand=nvidiartx,driver>=470,driver<471 brand=quadrortx,driver>=470,driver<471 brand=unknown,driver>=510,driver<511 brand=nvidia,driver>=510,driver<511 brand=nvidiartx,driver>=510,driver<511 brand=quadrortx,driver>=510,driver<511"

RUN apt install -y --no-install-recommends \
  gnupg2 \
  ca-certificates
# see: https://github.com/NVIDIA/nvidia-docker/issues/1632#issuecomment-1125739652
RUN curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${NVARCH}/3bf863cc.pub | apt-key add -
RUN echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list
RUN rm -rf /var/lib/apt/lists/*
# see for package versions: https://ubuntu.pkgs.org/22.04/cuda-amd64/
RUN apt-get update && apt-get install -y --no-install-recommends \
  cuda-cudart-11-7=${NV_CUDA_CUDART_VERSION} \
  cuda-libraries-11-7=${NV_CUDA_LIB_VERSION} \
  ${NV_CUDA_COMPAT_PACKAGE} \
  && ln -s cuda-11.7 /usr/local/cuda \
  && rm -rf /var/lib/apt/lists/*
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf \
  && echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf

# Installs Python 3.9
RUN apt update -y \
  && apt install -y software-properties-common
RUN DEBIAN_FRONTEND="noninteractive" add-apt-repository ppa:deadsnakes/ppa
RUN DEBIAN_FRONTEND="noninteractive" apt install -y python3.9

# python3.9 also installs python3.10, and on ubuntu:jammy python3.10 is the
# default version. To override this, manually delete and recreate the python3
# symbolic link.
RUN rm /usr/bin/python3
RUN ln -s python3.9 /usr/bin/python3

RUN DEBIAN_FRONTEND="noninteractive" apt install -y \
  python-is-python3 \
  python3-gdcm \
  python3-opencv \
  python3-pip \
  python3.9-distutils

# Download model weights
RUN wget https://zenodo.org/record/4003545/files/Task004_Hippocampus.zip

# Toggle SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL until
# https://github.com/MIC-DKFZ/nnUNet/pull/1209 is included in next nnUNet release
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

# Install PyPI requirements for this example
COPY bunkerhill/examples/hippocampus/requirements.txt requirements.txt
RUN pip install -r requirements.txt

# Prepares /app/model_release
COPY bunkerhill bunkerhill
COPY README.md README.md
COPY setup.py setup.py
RUN pip install --editable .

ENV PYTHONUNBUFFERED=1

# Add a new user "host_user"
RUN useradd -u ${USER_ID} host_user
# Change to non-root privilege
USER host_user

