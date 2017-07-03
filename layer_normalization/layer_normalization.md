> Layer Normalizaiton is just the "transpose of batch normalizaion".

1. transpose the input, and set the factor of moving average of mean and std to 0. Then the calculations in this layer is the same as batch norm.
2. layer normalization dose not need to save the mean and std, and the training and testing process are the same.
3. It seems that many codes can be resued between batch normalization and layer normalizaiton.

## forward
- $x_i$ is the vector representation of the summed inputs to the neurons in a layer
$$\mathbf{x} = \mathbf{W}^T\mathbf{h}$$
- $H$ is number of neurons in the layer
- $m$ is number of training samples in one mini-batch
- in mini-batch training, $\mathbf{x}$ is a matrix whose size is: $m \times H$

- In the forward pass, first compute the layer normalization statistics over the hidden units in the same layer:
$$\begin{split}
&\mu &= \frac{1}{H}\sum_{i=1}^{H} x_i \\
&\sigma^2 &= \frac{1}{H}\sum_{i=1}^H(x_i - \mu)^2
\end{split}$$

- normalize the output:
$$\begin{split}
&\mathbf{\hat x} &= \frac{\mathbf{x} - \mu}{\sqrt{\sigma^2 + \epsilon}} \\
&\mathbf{y} &= \gamma \odot \mathbf{\hat{x}} + \beta
\end{split}$$

**Note that both $\mu$ and $\sigma^2$ are matrix and they have the same size as $\mathbf{\hat{x}}_{m \times H}$**

### Some Notes:
- learnable parameters: $\gamma$ and $\beta$
- all the hidden units in a layer share the same normalization terms
- different training sample have different normalization terms

## backward
- the backward pass computes two things:
     1. partial derivatives of loss function with regard to learnable parameters: $\frac{\partial \mathcal{L}}{\partial \gamma}$ and $\frac{\partial \mathcal{L}}{\partial \beta}$
     2. partial derivatives of the loss function with regard to input: $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$

### 1. partial derivatives of $\mathcal{\mathcal{L}}$ with respect to learnable parameters:
$$\begin{eqnarray*}
&\frac{\partial \mathcal{L}}{\partial \mathbf{\gamma}} &= \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i} \odot \mathbf{\hat{x_i}} \tag{1}\\
&\frac{\partial \mathcal{L}}{\partial \mathbf{\beta}} &= \sum_{i=1}^{m} \frac{\partial \mathcal{L}}{\partial y_i} \tag{2}
\end{eqnarray*}$$

### 2. partial derivative of $\mathcal{L}$ with respect to input $\mathbf{x}$:
- $\mathcal{L}$ can be regarded as a function of $\mathcal{L}(\mathbf{\hat{x}}, \sigma^2, \mu)$
- $\mathbf{\hat{x}}$, $\sigma^2$, and $\mu$ are all functions of $\mathbf{x}$
- according to the chain rule, we can obtain the following formular:

$$
\frac{\partial \mathcal{L}}{\partial \mathbf{x}} = \frac{\partial \mathcal{L}}{\partial \mathbf{\hat{x}}}\frac{\partial \mathbf{\hat x}}{\partial \mathbf{x}} + \frac{\partial \mathcal{L}}{\partial \mu}\frac{\partial \mu }{\partial \mathbf{x}} + \frac{\partial \mathcal{L}}{\partial \sigma ^2}\frac{\partial \sigma^2}{\partial \mathbf{x}}
\tag{3}
$$

**let's calculate $\frac{\partial \mathcal{L}}{\partial \mathbf{x}}$ step by step.**

#### 1. the first part:
$$\begin{split}
&\frac{\partial \mathcal{L}}{\partial \mathbf{\hat{x}}} &= \frac{\partial \mathcal{L}}{\partial y} \odot \gamma \\
&\frac{\partial \mathbf{\hat x}}{\partial \mathbf{x}} &= (\sigma^2 + \epsilon)^{-\frac{1}{2}}
\end{split}$$

#### 2. the second part:

$$\frac{\partial \mathcal{L}}{\partial \mu} = \frac{\partial \mathcal{L}}{\partial \mathbf{\hat{x}}} \frac{\partial{\mathbf{\hat{x}}}}{\partial \mu} + \frac{\partial \mathcal{L}}{\partial \sigma^2}\frac{\partial \sigma^2}{\partial \mu} \tag{4}$$

$$\frac{\partial \mathbf{\hat{x}}}{\partial \mu} =  -(\sigma^2 + \epsilon)^{-\frac{1}{2}}
\tag{5}$$

$$\begin{split}
\frac{\partial \sigma^2}{\partial \mu} &= \frac{-2}{H} \sum_{i=1}^{H}(x_i - \mu) \\
&=\frac{-2}{H}(\sum_{i=1}^{H}x_i - \sum_{i=1}^{H} \mu) \\
&=0
\end{split} \tag{6}$$

- substitude $(5)$, $(6)$ into $(4)$:
$$\begin{split}
\frac{\partial \mathcal{L}}{\partial \mu_i} &&= \frac{\partial \mathcal{L}}{\partial \mathbf{\hat{x}}} \frac{\partial{\mathbf{\hat{x}}}}{\partial \mu_i } \\
&&=  -(\sigma_i^2 + \epsilon)^{-\frac{1}{2}} \sum_{j=1}^{H}\frac{\partial \mathcal{L}}{\partial {\mathbf{\hat{x_j}}}}
\end{split} \tag{7}$$

- $\frac{\partial \mathcal{L}}{\partial \mu}$ is a matrix shown as below, and equation ($7$) is one row of this matrix.
$$\left(
\begin{array}{ccc}
\frac{\partial \mathcal{L}}{\partial \mu_1} & ... & \frac{\partial \mathcal{L}}{\partial \mu_1} \\
 & ... &  \\
\frac{\partial \mathcal{L}}{\partial \mu_m} & ... & \frac{\partial \mathcal{L}}{\partial \mu_m} \\
\end{array}
\right)$$

- it is easy to get:
$$\frac{\partial \mu}{\partial \mathbf{x}} = \mathbf{\frac{1}{H}}
\tag{8}$$

#### 3. the third part
$$\begin{split}
\frac{\partial \mathcal{L}}{\partial \sigma_i^2} &= \frac{\partial \mathcal{L}}{\partial \mathbf{\hat{x}}}\frac{\partial \mathbf{\hat{x}}}{\partial \sigma_i^2} \\
 &=-\frac{1}{2}(\sigma_i^2+\epsilon)^{\frac{3}{2}} \sum_{j=1}^{H}&\frac{\partial \mathcal L}{\partial x_j}(x_j - \mu)
\end{split} \tag{9}$$

- $\frac{\partial \mathcal{L}}{\partial \sigma^2}$ is a matrix as shown below, and equation ($9$) is a row of this matrix.
$$\left(
\begin{array}{ccc}
\frac{\partial \mathcal{L}}{\partial \sigma_1^2} & ... & \frac{\partial \mathcal{L}}{\partial \sigma_1^2} \\
 & ... &  \\
\frac{\partial \mathcal{L}}{\partial \sigma_m^2} & ... & \frac{\partial \mathcal{L}}{\partial \sigma_m^2} \\
\end{array}
\right)$$

- for $\frac{\partial \sigma^2}{\partial x}$:
$$\frac{\partial \sigma^2}{\partial \mathbf{x}} = \frac{2}{H}(x_j - \mu),
j = i ... H \tag{10}$$

- $\frac{\partial \sigma^2}{\partial x}$ is a matrix shown as below, and equation ($10$) is one row of this matrix:
$$\left(
\begin{array}{ccc}
\frac{\partial \sigma_1^2}{\partial x_{1,1}} & ... & \frac{\partial \sigma_1^2}{\partial x_{1,H}} \\
 & ... &  \\
\frac{\partial \sigma_H^2}{\partial x_{m,1}} & ... & \frac{\partial \sigma_H^2}{\partial x_{m,H}} \\
\end{array}
\right)$$
