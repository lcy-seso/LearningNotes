# [Train longer, generalize better: closing the generalization gap in large batch training of neural networks](https://arxiv.org/abs/1705.08741)

- what is the **<font color=#8f4586>generalization gap</font>**?
  there is a persistent degradation in generalization performance when using large batch size.

- generalizaiton gap stems from **<font color=#8080c0>the relatively small number of updates rather than the batch size</font>**.

---

## Theoretical Analysis

- For physical intuition, one can think of the *weight vector* $\mathbf{w}_t$ as: **a particle performing a random walk on the loss ("potential") landscape $L (\mathbf{w_t})$**.
- **<font color=#8f4586 size=5>ultra-slow diffusion</font>**: the asymptotic behavior of the [auto-covariance](https://en.wikipedia.org/wiki/Autocovariance) of the random potential: $ \mathbb{E}( L({\mathbf{w}_1}) L({\mathbf{w}_2})) \sim \parallel \mathbf{w}_1 - \mathbf{w}_2 \parallel^2$, determines the asymptotic behavior of the random walker:
$$\mathbb{E} \parallel \mathbf{w}_t - \mathbf{\mathbf{w}_0} \parallel^2 \sim (\text{log t}^\frac{4}{\alpha})$$
- typically:  $d \triangleq \parallel \mathbf{w}_t - \mathbf{w}_0 \parallel \sim \left(\text{log} t \right) ^\frac{2}{\alpha}$, where $\parallel \cdot \parallel$ is the Euclidean distance of two vector.
  - to climb (or go around) each barrier takes exponentially long time in the heigh of the barrier: $t \sim \text{exp}(d^\frac{\alpha}{2})$