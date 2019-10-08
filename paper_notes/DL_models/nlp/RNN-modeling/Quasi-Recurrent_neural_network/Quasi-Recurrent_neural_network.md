# Quasi-Recurrent Neural Networks

_**input-dependent pooling + gated linear combination of convolution features**_.

## _Model_

* Parallel computation across both timestep and minibatch dimensions, enabling high throughput and good scaling to long sequences.
* Allow the output to depend on the overall order of elements in the sequences.

Each layer of Quasi-RNN consists of two kinds of subcomponents:

1. the convolutional components
    * _**fully parallel computation**_ across minibatches and spatial dimensions (sequence dimension)
1. the pooling components
    * _**fully parallel computation**_ across minibatches and feature dimensions.

### _Equations_

* Input $\mathbf{X} \in \mathcal{R}^{T \times n}$.
* $T$: number of time step.
* $n$: hidden dimension.
* $\ast$: masked convolition along time dimension

#### _The convolution part_

$$ \mathbf{Z} = \text{tanh} (\mathbf{W} \ast \mathbf{X})$$
$$ \mathbf{F} = \sigma(\mathbf{W}_f \ast \mathbf{X})$$
$$ \mathbf{O} = \sigma(\mathbf{W}_o \ast \mathbf{X})$$

#### _The pooling part_

1. $f$-pooling:  _**only forget gate**_

$$\mathbf{h}_t = \mathbf{f}_t \odot \mathbf{h}_{t-1} + (\mathbf{1} - \mathbf{f}_t) \odot \mathbf{z}_t$$

2. $fo$-pooling: _**forget gate and output gate**_

$$ \mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t - 1} + (\mathbf{1} - \mathbf{f}_t) \odot \mathbf{z}_t$$
$$ \mathbf{h}_t = \mathbf{o}_t \odot \mathbf{c}_t$$

3. $ifo$-pooling: _**forget gate, input gate and output gate**_

$$ \mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t - 1} + \mathbf{i}_t \odot \mathbf{z}_t$$
$$ \mathbf{h}_t = \mathbf{o}_t \odot \mathbf{c}_t$$

**The recurrent parts must be calculated for each timestep in sequence.**

## _Variants_

1. Dropout
    * choose a new subset of channels to "zone out" at each time step
    * for these chosen channels, the network copies states from one timestep to the next _**without modification**_.
    * modify the forget gates. The pooling function itself does not need to modify.

      $$ \mathbf{F} = \mathbf{1} -\text{dropout} (1 - \sigma(\mathbf{W}_f \ast \mathbf{X})) $$

2. Densely-connected layers

    * for sequence classification tasks, the author found it is helpful to use skip connections between _**every QRCNN**_.
    * add connections between embeddings and every QRCNN layer, and between every pair of QRCNN layers.
      * _**concatenate**_ each QRCNN's input to its output along the channel dimension before feeding the state to the next layer

1. Encoder-Decoder models

    * simply feeding the last encoder hidden state would not allow the encoder state to affect the gate or update values that are provided to the decoder's pooling layer.
    * _**how to fix**_
      * for the $l$-th decoder QRNN layer, outputs of its convolution functions is added with a linearly projected copy of the $l$-th encoder's last encoder state:

        $$ \mathbf{Z}^l = \text{tanh}(\mathbf{W}_z^l \ast \mathbf{X}^l + \mathbf{V}_z^l \tilde{\mathbf{h}}_T^l)$$
        $$ \mathbf{F}^l = \text{tanh}(\mathbf{W}_f^l \ast \mathbf{X}^l + \mathbf{V}_f^l \tilde{\mathbf{h}}_T^l)$$
        $$ \mathbf{O}^l = \text{tanh}(\mathbf{W}_o^l \ast \mathbf{X}^l + \mathbf{V}_o^l \tilde{\mathbf{h}}_T^l)$$

    * attention, in the below equations, $L$ is the last layer.

      $$\alpha_{st} = \text{softmax} (\mathbf{c}_t^L \cdot \tilde{\mathbf{h}}_s^L)$$
      $$ \mathbf{k}_t = \sum_{\alpha}\alpha_{st} \tilde{\mathbf{h}}_s^L$$
      $$\mathbf{h_t}^L = \mathbf{o}_t \odot (\mathbf{W}_k\mathbf{k}_t + \mathbf{W}_c\mathbf{c}_t^L)$$

      * use dot products of encoder hidden states with the decoder's last layer's _**un-gated**_ hidden states.
