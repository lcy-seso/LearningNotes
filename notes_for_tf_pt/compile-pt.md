1. Install anaconda.
2. Install dependent Python package

    ```bash
    conda create --name pytorch-build python=3.7.3 numpy=1.16.3
    conda activate pytorch-build # or `source activate pytorch-build`
    conda install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
    conda install -c pytorch magma-cuda100
    ```

3. Get the source codes

    ```bash
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch
    # if you are updating an existing checkout
    git submodule sync
    git submodule update --init --recursive
    ```
4. Compile

    maybe you need this:

    ```bash
    sudo apt-get install libomp-dev
    ```

    ```bash
    export USE_CUDA=1 USE_CUDNN=1 USE_MKLDNN=1 USE_NCCL=1 USE_SYSTEM_NCCL=0
    export CUDNN_LIBRARY_DIR="/home/yincao/opt/cudnn7.6_cuda10.0"
    export CUDNN_INCLUDE_DIR="/home/yincao/opt/cudnn7.6_cuda10.0/include"
    export CMAKE_PREFIX_PATH="$HOME/anaconda3/envs/pytorch-build"

    cd ~/pytorch
    python setup.py install
    ```

# Reference

1. [Official document](https://github.com/pytorch/pytorch#from-source)
1. [A Tour of PyTorch Internals](http://pytorch.cn/2017/05/11/Internals.html)
1. [PyTorch Internals Part II - The Build System](http://pytorch.cn/2017/06/27/Internals2.html)
