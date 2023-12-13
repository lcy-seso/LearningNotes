# MABA

序列建模的一个根本性问题是将context压缩为state。LTI（Linear Time Invariant）：线性时不变系统，参数与输入无关，CNN和RNN模型都可以认为是LTI。**而attention的成功arguably地认为system的dynamics是data dependent**。计算attention时，QKV序列的token之间会进行交互。

RNN将上下文压缩进有限长度的state，相比之下，attention完全不压缩context。autoagressive模式预测时，压缩context到固定长度的状态，决定了RNN在时间和空间上都是高效的，而attention要保留所有的context不进行压缩，计算和空间都是不高效的。

RNN模型的有效性受到how well the context is compresed的影响。

# Reference

1. Mamba: Linear-Time Sequence Modeling with Selective State Spaces: [[paper]](https://arxiv.org/pdf/2312.00752.pdf) [[codes]](https://github.com/state-spaces/mamba)[open review](https://openreview.net/forum?id=AL1fq05o7H)
1. [Transformer Quality in Linear Time](https://arxiv.org/pdf/2202.10447.pdf)
1. [Mamba - a replacement for Transformers?](https://www.youtube.com/watch?v=ouF-H35atOY)
1. [Legendre Memory Units: Continuous- Time Representation in Recurrent Neural Networks](https://proceedings.neurips.cc/paper/2019/file/952285b9b7e7a1be5aa7849f32ffff05-Paper.pdf)
1. [Combining recurrent, convolutional, and continuous-time models with linear state-space layers](https://arxiv.org/pdf/2110.13985.pdf)

几篇与RNN有关的工作

1. Transformers are RNNs: Fast Autoregressive Transformers with Linear Attention [[youtube vedio]](https://www.youtube.com/watch?v=hAooAOFRsYc)
1. Retentive Network: A Successor to Transformer for Large Language Models [[youtube vedio]](https://www.youtube.com/watch?v=ec56a8wmfRk)
1. RWKV: Reinventing RNNs for the Transformer Era [[youtube vedio]](https://www.youtube.com/watch?v=x8pW19wKfXQ)