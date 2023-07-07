<!-- START doctoc generated TOC please keep comment here to allow auto update -->
<!-- DON'T EDIT THIS SECTION, INSTEAD RE-RUN doctoc TO UPDATE -->
**Table of Contents**  *generated with [DocToc](https://github.com/thlorenz/doctoc)*

- [Efficient Layer Normalization in CUDA](#efficient-layer-normalization-in-cuda)
  - [The Welford Algorithm](#the-welford-algorithm)
- [Reference](#reference)

<!-- END doctoc generated TOC please keep comment here to allow auto update -->

# Efficient Layer Normalization in CUDA

I came across this little algorithm when thinking about the GPU kernel fusion problems in [Transformer](https://arxiv.org/pdf/1706.03762.pdf).


We want to calculate the mean and unbiased variance of a numeric list $X=[x_1 \ldots x_{N}]$ that has a length $N$:

$$
\begin{align*}
\mu_{N} &= \frac{1}{N}\sum^{N}_{i=1} x_i \\
\sigma^2_{N} &=\frac{1}{N-1} \sum^{N}_{i-1}\left(x_i - \mu_{N}\right)^2 
\end{align*}
$$

The two computations mentioned above scan the input list elements twice, each with a complexity of $O(N)$. Now, suppose we add a new element $x_{N}$ to the list and want to update the mean $\mu_{N+1}$ and variance $\sigma_{N+1}$. Instead of repeating the two $O(N)$ computations, can we update $\mu_{N+1}$ and $\sigma_{N+1}^2$ using the existing values of $N$, $\mu_{N}$, and $\sigma_{N}^2$? If so, we can compute $\mu_N$ and $\sigma_{N}^2$ for the entire list portion by portion for extremely long lists.

Yes, we can do so. That's the [welford's online algorithm](https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance) for calculating variance.

## The Welford Algorithm

Let's start from the new mean $\mu_{N+1}$ from $x_{N+1}$ and $\mu_{N}$. This is rather straightforward:

$$
\begin{align}
\mu_{N+1} &= \frac{x_{N+1}+N\mu_{N}}{N+1} \nonumber \\
&=\mu_{N}+\frac{x_{N+1}-\mu_{N}}{N+1}\tag{1}
\end{align}
$$

Subsitute $N+1$ with $N$ in equation (1), we can also get:

$$
\begin{equation}
\mu_{N}=\mu_{N-1}+\frac{x_{N}-\mu_{N-1}}{N}\tag{2}
\end{equation}
$$

Let's define a quantity called $s_N$:

$$
\begin{align*}
s_{N} &= \sum^{N}_{i=1}(x_i - \mu_{N})^2 \\
&=\sum^{N-1}_{i=1}\left(x_i-\mu_{N}\right)^2+\left(x_N-\mu_{N}\right)^2 \tag{3}
\end{align*}
$$

Substitue equation (2) into equation (3):

$$
\begin{align*}
s_N &= \sum_{i=1}^{N-1}\left(x_i-\mu_{N-1}-\frac{x_N-\mu_{N-1}}{N} \right)^2 + \left(x_N-\mu_{N-1}-\frac{x_N-\mu_{N-1}}{N}\right)^2 \\
&=\sum_{i=1}^{N-1}\left(\left(x_i-\mu_{N-1}\right)-\frac{1}{N}\left(x_N-\mu_{N-1}\right)\right)^2+\left(\frac{N-1}{N}(x_N-\mu_{N-1})\right)^2 \tag{4}
\end{align*}
$$

Let's first look into the first component of equation (4):

$$
\begin{align*}
&\sum_{i=1}^{N-1}\left(\left(x_i-\mu_{N-1}\right)-\frac{1}{N}\left(x_N-\mu_{N-1}\right)\right)^2 \\
=&\sum_{i=1}^{N-1}\left((x_i-\mu_{N-1})^2-\textcolor{red}{\frac{2}{N}(x_i-\mu_{N-1})(x_N-\mu_{N-1})}+(x_N-\mu_{N-1})^2\right) 
\end{align*}
$$

$$
\begin{align*}
&\sum_{i=1}^{N-1}\frac{2}{N}(x_i-\mu_{N-1})(x_N-\mu_{N-1}) \\
=&\frac{2}{N}(N-1)(x_N-\mu_{N-1})\left(\sum_{i=0}^{N-1}x_i-(N-1)\mu_{N-1}\right) \\
=&\frac{2}{N}(N-1)(x_N-\mu_{N-1})\left(\sum_{i=0}^{N-1}x_i-\sum_{i=0}^{N-1}x_i\right) \\
=&\ 0
\end{align*}
$$

Substitue these results into equation (4):

$$
\begin{align*}
s_{N} &=\sum_{i=1}^{N-1}\left((x_i-\mu_{N-1})^2+\frac{1}{N}(x_N-\mu_{N-1})^2\right) + \left(\frac{N-1}{N}(x_N-\mu_{N-1})\right)^2 \\
&=\frac{N-1}{N^2}(x_N-\mu_{N-1})^2+\frac{(N-1)^2}{N^2}(x_N-\mu_{N-1})^2+\sum_{i=1}^{N-1}(x_i-\mu_{N-1})^2 \\
&=\left[\frac{N-1}{N^2} + \frac{(N-1)^2}{N^2}\right](x_N-\mu_{N-1})^2+\sum_{i=1}^{N-1}\left(x_i-\mu_{N-1}^2\right) \\
&=s_{N-1} + \frac{N-1}{N}(x_N-\mu_{N-1})^2 \tag{5}
\end{align*}
$$

since:
$$
\begin{align*}
(N-1)\mu_{N-1} &= N\mu_N-x_N \\
\mu_{N-1} &= \frac{N\mu_{N}-x_N}{N-1}
\end{align*}
$$
substitute it into equation (5):

$$
\begin{align*}
s_N&=s_{N-1} + (x_N-\mu_{N-1})\frac{N-1}{N}(x_N-\frac{N\mu_{N}-x_N}{N-1}) \\
s_N&=s_{N-1} + (x_N-\mu_{N-1})(x_N-\mu_{N})
\end{align*}
$$

Suppose we use Welford's algorithm to calculate the mean and variance for lists $A$ and $B$.
Now we want to obtain the mean and variance for $AB$, which is the concatenation of $A$ and $B$. To do this, we can use the formula described in [3].

$$
\begin{align*}
N_{AB} &= N_{A} + N_{B} \\
\delta &= \mu_B - \mu_A \\
\mu_{AB} &= \mu_A+\delta\frac{N_B}{N_{AB}} \\
M_{2, AB} &=M_{2,A}+M_{2,B}+\delta^2*\frac{N_A N_B}{N_{AB}}
\end{align*}
$$

# Reference

1. Welford's paper on "[Note on a Method for Calculating Corrected Sums of Squares and Products](https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.302.7503&rep=rep1&type=pdf)"
1. [CUDA优化之LayerNorm性能优化实践](https://zhuanlan.zhihu.com/p/443026261)
1. [Updating Formulae and a Pairwise Algorithm for Computing Sample Variances](http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf).