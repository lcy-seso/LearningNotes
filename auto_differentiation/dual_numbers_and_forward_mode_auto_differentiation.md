# Forward Mode Auto Differentiation

1. Forward mode automatic differentiation is accomplished by <span style="background-color:#ACD6FF;">_**augmenting the algebra of real numbers**_</span> and obtaining a new arithmetic.

1. With the help of dual number operations on numbers, it is possible to <span style="background-color:#ACD6FF;">_**calculate the value of $f(x)$ while also calculating $f'(x)$ at the same time**_</span>.

## Dual Numbers

* _**Defination**_

  Dual Numbers are an extensions to real numbers. They have the following form:

    $$a + b\epsilon$$
    where $a$ is the real part, and $\epsilon^2 = 0$.

* _**Basic Dual Number Math**_

  $$ (a + b\epsilon) + (c + d \epsilon) = (a + b) + (c + d)\epsilon$$
  $$ (a + b\epsilon)(c + d \epsilon) = ac + \epsilon(ad + bc) + bd\epsilon ^2 = ac + \epsilon(ad + bc)$$

## Dual Numbers and Forward Mode Auto Differentiation

### How Dual Number works

* A real-valued function $f(x)$ can be approximated by a polynomial according Taylor series expansion, if the function $f(x)$ is infinitely differentiable at a real or complex number $a$

* If $P(x) = p_0 + p_1x+ p_2x^2 + ... + p_nx^n$, then we have:

    $$P(x + x'\epsilon) = p_0 + p_1(x + x'\epsilon)+ p_2(x+x'\epsilon)^2 + ... + p_n(x + x'\epsilon)$$
    $$=p_0 + p_1x + p_2x^2 + p_nx^n + p_1x'\epsilon + 2p_2xx'\epsilon + ... + np_nx^{n-1}x'\epsilon$$
    $$=P(x) + P'(x)x'\epsilon$$

    where $x'$ is called a seed which can be choosen arbitrarily (for example 1 and 0).
    * The equation above means we can feed our function $f$ with $x + x'\epsilon$ and expect the resulting dual number to keep function evaluation and its first derivative.

* The new arithmetic consists of ordered pairs, elements writtene $<x,x'>$, with ordinary arithmetics on the first component, and first order differentiation arithmetic on the second component.
  * See more details about the basic arithmetic and some standard functions for the new arithmetic in [this page]( https://en.wikipedia.org/wiki/Automatic_differentiation#Automatic_differentiation_using_dual_numbers).

* In C++ this leads to a very simple operator overloading implementation, see explanations in [this page](https://en.wikipedia.org/wiki/Automatic_differentiation#Implementation).

## References

1. [Dual Numbers & Automatic Differentiation](https://blog.demofox.org/2014/12/30/dual-numbers-automatic-differentiation/)
1. [Automatic differentiation for machine learning in Julia](https://int8.io/automatic-differentiation-machine-learning-julia/)
1. [Multivariable Dual Numbers & Automatic Differentiation](https://blog.demofox.org/2017/02/20/multivariable-dual-numbers-automatic-differentiation/)
1. [Forward-Mode Automatic Differentiation in Julia](https://arxiv.org/pdf/1607.07892.pdf)
