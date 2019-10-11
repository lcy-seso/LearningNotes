# [weight normalization](https://arxiv.org/pdf/1602.07868.pdf)

## Motivations

- the practical success of first-order gradient based optimization is highly dependent on the curvature of the objective that is optimized.
- If the [condition number](https://en.wikipedia.org/wiki/Condition_number) of the Hessian matrix of the objective at the optimum is low, the problem is said to exhibit pathological curvature, then the first-order gradient descent is hard to make progress.

- There may be **multiple equivalent ways of parameterizing the same model**.
    - some are much easier to optimize than others.

- Finding good ways of parameterizing neural networks: improve the conditioning of the cost gradient for general neural network.
    1. preconditioning.
    1. change the parameterization of the model to give gradients that are more like the whitened natural gradients.

    The later one is the way weight normalization chooses.

## What is weight normalization

The computation of each neural is the weighted sum of input features followed by an element-wise nonlinearity, formulated as follows:

$$y = \phi (\mathbf{w} \cdot x + b) \tag{1}$$

weight normalization

### Forward pass

1. explicitly reparameterize each weight vector $\mathbf{w}$ in terms of a parameter vector $\mathbf{v}$ and a scalar parameter $g$.
1. perform stochastic gradient in the new parameters $\lVert \mathbf{v} \rVert$ and $g$.

$$\mathbf{w} = \frac{g}{\lVert \mathbf{v} \rVert} \mathbf{v} \tag{2}$$

1. the parameterization has the effect of fixing the Euclidean norm of the weight vector $\mathbf{w}$.
1. $\lVert\mathbf{w} \rVert = g$ independent of $\lVert \mathbf{v} \rVert$.


### Gradients

$$\bigtriangledown_g L = \frac{\bigtriangledown_w L \cdot \mathbf{v}}{\lVert \mathbf{v} \rVert}$$

$$\bigtriangledown_{\mathbf{v}} L = \frac{g}{\lVert \mathbf{v} \rVert} \bigtriangledown_{\mathbf{w}L} - \frac{g \bigtriangledown_g L}{\lVert \mathbf{v}^2 \rVert}\mathbf{v}$$


### Weight scale invariance
