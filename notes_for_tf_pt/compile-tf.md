# [Build from source](https://www.tensorflow.org/install/source)

## Version info

- OS info
    ```bash
    ζ lsb_release -a
    No LSB modules are available.
    Distributor ID: Ubuntu
    Description:    Ubuntu 16.04.2 LTS
    Release:        16.04
    Codename:       xenial
    ```
- GCC 5.4.0
- Python 3.6.5
- ~~bazel 0.23.0~~ bazel 0.24.1
    - ~~_**NOTE: You have to use bazel 0.23.0 if you are building TensorFlow r2.0, otherwise, TensorFlow complains.**_~~
- Cuda 10.0

## Compile

1. Configure
    ```bash
    ./configure
    ```

2. Compile.

    ```bash
    bazel build \
     --verbose_failures \
     --config=opt \
     --config=cuda \
     --action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
     //tensorflow/tools/pip_package:build_pip_package
    ```

    ```bash
    bazel build \
      --config=opt \
      --config=cuda \
      --cxxopt="-D_GLIBCXX_USE_CXX11_ABI=0" \
      --config=mkl \
      --config=v2 \
      --verbose_failures \
      --action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
     //tensorflow/tools/pip_package:build_pip_package
    ```
3. Build wheel.

    ```bash
    bazel-bin/tensorflow/tools/pip_package/build_pip_package ~/tensorflow_package
    ```

    ```bash
    bazel build --config opt tensorflow/core/profiler:profiler
    ```

## Update (2019-10-17)

1. It seems that from TensorFlow r2.0, `configure.py` does not ask users to set `Cuda`, `cuDNN`, and `NCCL` installation path. If there are multiple Cuda libraries installed on your machine, you can set below environment variables to set `Cuda`, `cuDNN` and `NCCL`'s paths:

    ```bash
    export TF_CUDA_PATHS="/usr/local/cuda-10.0"
    export CUDA_TOOLKIT_PATH="/usr/local/cuda-10.0"

    export TF_CUDNN_VERSION="7.6"
    export CUDNN_INSTALL_PATH="/home/yincao/opt/cudnn7.6_cuda10.0"

    export TF_NCCL_VERSION="2.4"
    export NCCL_INSTALL_PATH="/home/yincao/opt/nccl_2.4.8-1+cuda10.0_x86_64"
    export NCCL_HDR_PATH="/home/yincao/opt/nccl_2.4.8-1+cuda10.0_x86_64/include"
    ```
