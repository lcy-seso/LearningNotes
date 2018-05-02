# [Convolutional Sequence to Sequence Learning](https://arxiv.org/pdf/1705.03122.pdf)

- Motivation

- Architecture / Basic blocks

1. positional embedding
    - look_up_table but has to support padding_idx
    - addition
1. convolution block structure: one dimensional convolution followed by a GLU.
    > Is it necessary to implement GLU in one operator to optimize the time efficiency. This can be determined later.

    - sequence convolution
        - 2D convolution: sequence_conv_op
    - GLU
        - **offset operator** ?? (To be determined later)
        - sigmoid
        - element-wise multiplication
        - addition
    - attention
        - matmul_op (the [batched matrix multiplication]( https://github.com/PaddlePaddle/Paddle/blob/develop/paddle/operators/matmul_op.cc))
        - <font color=#DB7093>**softmax along the specified axis**</font>
        - reshape op
        - softmax
    - weight normalization
