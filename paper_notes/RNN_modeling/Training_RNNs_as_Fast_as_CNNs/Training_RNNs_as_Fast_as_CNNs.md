# [Training RNNs as Fast as CNNs](https://arxiv.org/abs/1709.02755)

## Model

### Main motivations

1. _**process the input at each step independently of the other inputs.**_
1. _**do the recurrent combination with relatively lightweight computaion (element-wise operations that can be fused into a single kernel function call).**_

### Euqatioins of Simple Recurrent Units (SRU)

* _**linear transformation of the input**_
  $$ \mathbf{\tilde{x}}_t = \mathbf{W}\mathbf{x}_t $$

* _**forget gate**_

  $$ \mathbf{f}_t = \sigma(\mathbf{W}_f\mathbf{x}_t + \mathbf{b}_f) $$

* _**reset gate**_
  $$ \mathbf{r}_t = \sigma(\mathbf{W}_r\mathbf{x}_t + \mathbf{b}_r) $$

* _**internal state**_

  $$ \mathbf{c}_t = \mathbf{f}_t \odot \mathbf{c}_{t - 1} + (\mathbf{1} - \mathbf{f}_t) \odot \mathbf{\tilde{x}}_t $$

* _**output state**_

  $$ \mathbf{h}_t = \mathbf{r}_t \odot g(\mathbf{c_t}) + (\mathbf{1} - \mathbf{r}_t \odot \mathbf{x}_t)$$
