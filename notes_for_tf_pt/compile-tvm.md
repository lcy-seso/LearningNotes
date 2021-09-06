# Build from source

1. Get the source codes

    ```bash
    git clone --recursive https://github.com/apache/tvm tvm

    mkdir build
    cp cmake/config.cmake build
    ```
1. Modify the config.

    Edit the `config.cmake` file. Some options that may be useful:

    ```bash
    set(USE_CUDA ON)  # default is OFF, set to ON and let cmake find cuda path.
    set(USE_LLVM /data/yincao/llvm/bin/llvm-config)
    set(USE_GRAPH_RUNTIME OFF)
    set(USE_GRAPH_RUNTIME_DEBUG OFF)
    set(USE_MKL /home/yincao/intel/oneapi/mkl/latest)
    set(USE_RELAY_DEBUG ON)
    ```

1. compile

    ```bash
    cd build
    make -j28
    ```
    Then you will see `libtvm.so` and `libtvm_runtime.so` in the build directory.

# How to use TVM

# Reference

1. Offical document: [Install from souce](https://tvm.apache.org/docs/install/from_source.html#install-from-source)
