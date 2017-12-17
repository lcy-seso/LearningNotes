#### Background

- LeNet style models
    - simple stacks of convolutions for feature extraction and max-pooling operations for spatial sub-sampling.
    - refine into AlexNet
        - convolution operations were being repeated multiple times in-between max-pooling.
    - this style network goes deeper: a refined version: VGG
- a new style architecture: Inception architecture
    - inspired by [network-int-network](https://arxiv.org/abs/1312.4400)
    - architectures:
        - GoogleLeNet : InceptionV1
        - InceptionV2 : [Batch normalization: Accelerating
deep network training by reducing internal covariate shift](https://arxiv.org/abs/1502.03167)
        - InceptionV3 : [Rethinking the inception architecture for computer vision](https://arxiv.org/abs/1512.00567)
        - Inception-ResNet : [Inception-v4,
inception-resnet and the impact of residual connections on
learning](https://arxiv.org/abs/1602.07261)

#### Inception hypothesis

>cross-channel correlations and spatial correlations are sufficiently decoupled that it is
preferable not to map them jointly.

- A convolution layer attempts to learn filters in a 3D space, with 2 spatial dimensions (width and height) and a channel dimension
- thus a single convolution kernel is tasked with **simultaneously mapping cross-channel correlations and spatial correlations**.
- make this process easier and more efficient by explicitly factoring it into a series of operations that would **independently** look at (1) cross-channel correlations and at (2) spatial correlations.
