# Preparation

## Install MKL

Check [official webpage](https://software.seek.intel.com/performance-libraries) for latest release if needed.

```bash
wget http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/15816/l_mkl_2019.5.281.tgz
tar xzvf l_mkl_2019.5.281.tgz
cd l_mkl_2019.5.281
./install.sh
```
By default, MKL will be installed to `$HOME/intel`.

## Compile GCC

Latest PyTorch requires GCC 5+.

1. Download the source codes. Check this [webpage](https://bigsearcher.com/mirrors/gcc/releases/).

    ```bash
    wget https://bigsearcher.com/mirrors/gcc/releases/gcc-5.5.0/gcc-5.5.0.tar.gz
    tar xfz gcc-5.5.0.tar.gz
    ```

1. Download prerequisites.
   ```bash
   cd gcc-5.5.0
   # execute below command under the root directory of gcc source codes.
   ./contrib/download_prerequisites
   ```

2. Create a new directory, for example `gcc_build` under the root directory of the source codes. Do not compile under the root directory.

   ```bash
   mkdir gcc_build
   cd gcc_build

   ../configure      \
      --prefix=$HOME/opt/gcc_5.5.0 \
      --enable-shared                                  \
      --enable-threads=posix                           \
      --enable-__cxa_atexit                            \
      --enable-clocale=gnu                             \
      --enable-languages=all                           \
      --disable-multilib
    ```

3. Install

    ```
    make & make install
    ```

## Compile PyTorh

1. Install dependent Python package

    ```bash
    pip3 install numpy pyyaml mkl mkl-include setuptools cmake cffi typing
    ```

1. Get the source codes

    ```bash
    git clone --recursive https://github.com/pytorch/pytorch
    cd pytorch
    # if you are updating an existing checkout
    git submodule sync
    git submodule update --init --recursive
    ```
1. Compile

    If you need to adjust pre-detected libraries such as CuDNN/MKL/NCCL, run:

    ```bash
    python3 setup.py build --cmake-only
    ```

    then:

    ```bash
    cd build
    ccmake ../
    ```
    make any adjustion you need.

    ```base
    DEBUG=0 python3 setup.py install
    ```

## Test Installation

```bash
>>> import torch
>>> print(torch.cuda.is_available())
>>> print(torch.backends.cudnn.is_acceptable(torch.cuda.FloatTensor(1)))
>>> print(torch.backends.cudnn.version())
```

## Compile debug version

# Reference

1. [Official document](https://github.com/pytorch/pytorch#from-source)
1. [A Tour of PyTorch Internals](http://pytorch.cn/2017/05/11/Internals.html)
1. [PyTorch Internals Part II - The Build System](http://pytorch.cn/2017/06/27/Internals2.html)
