<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [DeepCPU Serving RNN-based Deep Learning Models 10x Faster](#deepcpu-serving-rnn-based-deep-learning-models-10x-faster)
	- [Problem to address](#problem-to-address)
	- [Key techniques](#key-techniques)
	- [Takeaways](#takeaways)
- [References](#references)

<!-- /TOC -->

# DeepCPU Serving RNN-based Deep Learning Models 10x Faster

[link](https://www.usenix.org/system/files/conference/atc18/atc18-zhang-minjia.pdf)

## Problem to address

Using TensorFlow/CNTK for RNN serving only gets less than 2% to peak performance? Are we dealing with an intrinsically challenging workload or less optimized systems?

The paper characterizes the reasons are:
1. due to latency SLA(service level agreement), small batch size (1~10) is used in serving. The computation is dominated by several vector-matrix multiplications that have poor _data resue_.
1. DL framework relies on parallel GEMM implementations which are designed to optimize large MMs with high data reuse.
    - MMs in RNNs are usually much smaller, fitting entirely in shared $L3$ cache, but with minimal data reuse.
1. RNN serving is computationally intensive but with limited parallelism.

Find an optimized implementation for RNN execution that _**maximizes data reuse**_ while also efficiently using low-level hardware resources.

## Key techniques

1. private-cache-aware partition
    - optimize the data movement between _**shared**_ $L3$ cache to _**private**_ $L2$ cache.
1. weight-centric streamlining
    - moves computation to where weights are stored to maximize data reuse across multiple steps of RNN execution.
    - This idea seems to be similar to [[1](#References)].
1. other techniques, MM fusion, reuse-aware parallelism decision.

## Takeaways

1. I learned the concepts of the Roofline model and data reuse.
    > _**Data reuse at a particular level of the memory hierarchy is a measure of the number of computational ops that can be executed per data load/store at that level of the memory hierarchy.**_

1. GEMM works poorly for small MMs. [This work](https://github.com/lcy-seso/LearningNotes/blob/master/paper_notes/DL_workload_optimization/Optimizing_RNN_performance/Optimizing_RNN_performance.md#investigating-performance-of-gpu-blas-libraries) also makes the same claim, and in most cases, no one hopes to implement GEMM himself, so we need MM fusion.
1. A methodology for optimization work
    1. An important start of optimization is to define a concise search space.
    1. Reduce the search space.
1. Some facts:
    1. partition/merge/in-place memory access seems to be useful semantics to be supported at the back-end.
    1. The computation of RNN is dominated by MMs.
    1. RNN's computation is intrinsically sequential, but if we split the computation into more fine-grained granularities (no details are defined at the moment), or stack more RNN cells, there is still a large space for parallelism execution. And, parallelism boots compute capacity but may also increase data movement. This is a soft scheduling space.

# References

1. [Persistent RNNs: Stashing Recurrent Weights On-Chip](http://proceedings.mlr.press/v48/diamos16.pdf)
1. [Roofline: An Insightful Visual Performance Model for Floating-Point Programs and Multicore Architectures](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2008/EECS-2008-134.pdf)
1. [Scaling deep learning](https://berkeley-deep-learning.github.io/cs294-131-s17/slides/Catanzaro_Berkeley_CS294.pdf)
1. [Roofline Model与深度学习模型的性能分析](https://zhuanlan.zhihu.com/p/34204282)
1. [让深度学习更高效运行的两个视角](https://zhuanlan.zhihu.com/p/33693725)
