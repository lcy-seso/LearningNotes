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
- bazel 0.23.0
    - _**NOTE: You have to use bazel 0.23.0 if you are building TensorFlow r2.0, otherwise, TensorFlow complains.**_
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
      --verbose_failures \
      --action_env="LD_LIBRARY_PATH=${LD_LIBRARY_PATH}" \
     //tensorflow/tools/pip_package:build_pip_package
    ```
3. Build wheel.
    ```bash
    bazel build --config opt tensorflow/core/profiler:profiler
    ```
