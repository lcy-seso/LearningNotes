# XLA: The TensorFlow compiler framework

* [TensorFlow, Compiled!](https://autodiff-workshop.github.io/slides/JeffDean.pdf)

## Goals

1. Improved execution speed.
1. Improved tensor buffer memory usage.
1. Make the performance of low-level Ops be the same as that of hand-written fused implementations.
1. Improved mobile footprint. Eliminate the TensorFlow runtime.
1. Improved protability.
    * It should be relatively easy to write a new back-end for novel hardware.

## XLA

* The semantics of operations are _**high level**_. This preserves enough information to allow sophisticated scheduling and optimization.

![](https://www.tensorflow.org/images/how-does-xla-work.png)

* XLA program = static, decomposed TF ops
  * math-looking _**primitive ops**_
  * make _**macro-ops by composition**_

### A key question: why write every new macro-op?

* Why write every new macro-op in C++?
* Why can't compose new operators out of existing TF ops?

### Compliation benefits

1. Eliminates op dispatch overhead.
1. Fuses ops.
    * reduce memory access
1. Memory usage analysis
    * reuse memory
    * update in-place
1. Models to executables: reduce executable size by generating what you need.
