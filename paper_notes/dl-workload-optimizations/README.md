* Computation
    * [Cudnn RNN optimization](https://devblogs.nvidia.com/optimizing-recurrent-neural-networks-cudnn-5/)
    * [Mixed precision training](https://github.com/NVIDIA/OpenSeq2Seq) : need device-level support, may for for NVIDIA Volta GPUs
    * [Persistent RNN](http://proceedings.mlr.press/v48/diamos16.pdf)
* Memory Acess

_**Usually the above two goals are achieved by optimized kenerl implementation.**_

* Communication
  * optimize TensorFlow's original implementation
      * [Baidu's allreduce for TensorFlow](https://github.com/baidu-research/tensorflow-allreduce/compare/allreduce-patch-1.0)
      * [Uber's Horovod](https://github.com/uber/horovod)
  * quantized gradients
* Operator scheduling
  * overlap computation and memory copy
* synchronization overhead by synchronous SGD algorithm

---

For RNN model, how to train _**very large and deep models for very long sequences**_ on one GPU efficiently.
