<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Reading List](#reading-list)
  - [Latency-oriented scenarios with high-end accelerators](#latency-oriented-scenarios-with-high-end-accelerators)
  - [Collaborative computing](#collaborative-computing)
  - [Memory footprint optimization](#memory-footprint-optimization)
- [Text generation inference](#text-generation-inference)
  - [projects](#projects)
- [一些背景资料](#一些背景资料)
- [一些项目](#一些项目)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Reading List

- [ ] "Tabi: An Efficient Multi-Level Inference System for Large Language Models" (EuroSys 23)[[PDF]](https://dl.acm.org/doi/pdf/10.1145/3552326.3587438)
- [ ] "ByteTransformer: A High-Performance Transformer Boosted for Variable-Length Inputs" [[PDF]](https://arxiv.org/abs/2210.03052) [[codes]](https://github.com/bytedance/ByteTransformer)
- [ ] "SpecInfer: Accelerating Generative LLM Serving th Speculative Inference and Token Tree verification"[[PDF]](https://arxiv.org/pdf/2305.09781.pdf)
- [ ] "S3: Increasing GPU Utilization during Generative Inference for Higher Throughput" [[PDF]](https://arxiv.org/pdf/2306.06000.pdf)
- [x] "[FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU](./FlexGen.md)" (ICML 2023)[[PDF]](https://arxiv.org/pdf/2303.06865.pdf)[[codes]](https://github.com/FMInference/FlexGen)

## Latency-oriented scenarios with high-end accelerators
- [ ] [Faster Transformer](https://github.com/NVIDIA/FasterTransformer)
- [ ] **DeepSpeed Zero-Inference**: "Deepspeed-inference: Enabling efficient inference of transformer models at unprecedented scale (SC 22)" [[PDF]](https://arxiv.org/pdf/2207.00032.pdf)
- [ ] **Orca**: "Orca: A Distributed Serving System for Transformer-Based Generative Models" osdi 22, [[PDF]](https://www.usenix.org/system/files/osdi22-yu.pdf)
- [ ] **LightSeq**: "LightSeq: A High Performance Inference Library for Transformers" [[PDF]](https://arxiv.org/pdf/2010.13887.pdf)
- [ ] **PaML inference**: "Efficiently Scaling Transformer Inference" [[PDF]](https://arxiv.org/pdf/2211.05102.pdf)
- [ ] **Turbo Transformers**: "Turbotransformers: an efficient gpu serving system for transformer models" [[PDF]](https://arxiv.org/pdf/2010.05680.pdf)
- [ ] **Hugging Face Accelerate**:
- [ ] vLLM: Easy, fast, and cheap LLM serving for everyone [[blog]](https://www.anyscale.com/blog/continuous-batching-llm-inference)[[github]](https://github.com/vllm-project/vllm)

- [ ] "Fast Distributed Inference Serving for Large Language Models (2023)" [[PDF]](https://arxiv.org/pdf/2305.05920.pdf)
- [ ] "ALT: Breaking the Wall between Data Layout and Loop Optimizations for Deep Learning Compilation" [[PDF]](https://dl.acm.org/doi/pdf/10.1145/3552326.3587440)

## Collaborative computing

- [ ] "Petals: Collaborative inference and fine-tuning of large models" [[PDF]](https://arxiv.org/pdf/2209.01188.pdf)

## Memory footprint optimization

- [ ] "OLLA: Optimizing the Lifetime and Location of Arrays to Reduce the Memory Usage of Neural Networks" [[PDF]](https://arxiv.org/pdf/2210.12924.pdf)
- [ ] "Zero-offload: Democratizing billion-scale model training." [[PDF]](https://arxiv.org/pdf/2101.06840.pdf)

# Text generation inference

## projects

1. huggingface text generation inference [[github]](https://github.com/huggingface/text-generation-inference)
1. faster transformer [[github]](https://github.com/NVIDIA/FasterTransformer)
1. vLLM [[blog]](https://vllm.ai/)

# 一些背景资料

1. [分析transformer模型的参数量、计算量、中间激活、KV cache](https://zhuanlan.zhihu.com/p/624740065)
1. [On Layer Normalization in the Transformer Architecture](http://proceedings.mlr.press/v119/xiong20b/xiong20b.pdf)
1. [Accelerated Inference for Large Transformer Models Using NVIDIA Triton Inference Server](https://developer.nvidia.com/blog/accelerated-inference-for-large-transformer-models-using-nvidia-fastertransformer-and-nvidia-triton-inference-server/)
   
<p align="center">
<img src="figures/pre-post-layer-normalization-in-transformer.png" width=50%>
</p>

# 一些项目

1. [HELM benchmarks](https://crfm.stanford.edu/helm/latest/)
1. [xFormer](https://github.com/facebookresearch/xformers)
1. [mosec](https://github.com/mosecorg/mosec)