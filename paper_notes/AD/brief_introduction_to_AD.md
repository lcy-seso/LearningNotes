<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->
-   [Auto Differentiation](#auto-differentiation)
    -   [Forward-Mode AD](#forward-mode-ad)
        -   [What is the Forward-Mode AD](#what-is-the-forward-mode-ad)
        -   [Summary of the Forward-Mode AD](#summary-of-the-forward-mode-ad)
        -   [Dual Number and Forward-Mode AD Implementation](#dual-number-and-forward-mode-ad-implementation)
            -   [Definition of Dual Number](#definition-of-dual-number)
            -   [How Dual number works in AD?](#how-dual-number-works-in-ad)
    -   [Reverse-Mode AD](#reverse-mode-ad)
        -   [What is the Reverse-Mode AD](#what-is-the-reverse-mode-ad)
        -   [Summary of the Reverse-Mode AD](#summary-of-the-reverse-mode-ad)
        -   [Tape-based Reverse-Mode Implementation](#tape-based-reverse-mode-implementation)
    -   [References](#references)

<!-- /TOC -->
# Auto Differentiation

What is the auto differentiation? Let's see an example:

$$ z = x * y + sin(x)$$ where $x$, $y$, and $z$ are scalar.

First, let's think about how a *computer* would evaluate $z$ via a sequence of primitive operations.

1.  $x = ?$
2.  $y = ?$
3.  $a = x * y$
4.  $b = sin(x)$
5.  $z = a + b$

## Forward-Mode AD

### What is the Forward-Mode AD

-   Let's differentiate each equation with respect to some ***yet-to-be-given*** variable $t$

    1.  $\frac{\partial x}{\partial t} = ?$
    2.  $\frac{\partial y}{\partial t} = ?$
    3.  $\frac{\partial a}{\partial t} = y *\frac{\partial x}{\partial t} + x * \frac{\partial y}{\partial t}$
    4.  $\frac{\partial b}{\partial t} = cos(x) * \frac{\partial x}{\partial t}$
    5.  $\frac{\partial z}{\partial t} = \frac{\partial a}{\partial t} + \frac{\partial b}{\partial t}$

-   Let's transform the above equations by using differential variables: $\{\text{d}x, \text{d}y, ...\}$, which stands for $\{\frac{\partial x}{\partial t}, \frac{\partial y}{\partial t}, ...\}$. Then we can get:

    1.  $\text{dx} = ?$
    2.  $\text{dy} = ?$
    3.  $\text{da} = y * \text{d}x + x * \text{d}y$
    4.  $\text{db} = cos(x) * \text{d}x$
    5.  $\text{dz} = \text{da} + \text{db}$

-   ***Let's substitute $t = x$, then we can get***

    1.  $\text{dx} = 1$
    2.  $\text{dy} = 0$
    3.  $\text{da} = y$
    4.  $\text{db} = cos(x)$
    5.  $\text{dz} = y + cos(x)$

-   ***Let's substitute $t = y$, then we can get***

    1.  $\text{dx} = 0$
    2.  $\text{dy} = 1$
    3.  $\text{da} = x$
    4.  $\text{db} = 0$
    5.  $\text{dz} = x$

**The above process is the forward mode AD.**

### Summary of the Forward-Mode AD

-   ***Advantages***
    1.  The differential variables usually depend on the intermediate variables of the forward computation. If we do gradient computations together with forward computation, there is no need to hold on to the intermediate variables until later, saving lots of memory.
    2.  It enables a very simple and direct implementation of the forward mode AD by using dual number arithmetics.
-   ***Disadvantages***
    -   The complexity of forward-mode AD is $O(n)$ and $n$ is the number of input variables.
    -   The forward-mode AD would be very costly if we want to calculate the gradient of a large complicated function of many variables which happens surprisingly open in practice.

### Dual Number and Forward-Mode AD Implementation

Forward mode automatic differentiation is accomplished by [***augmenting the algebra of real numbers***]{style="background-color:#ACD6FF;"} and obtaining a new arithmetic. - An additional component is added to every number which will [***represent the derivative of a function at the number***]{style="background-color:#ACD6FF;"} - And all arithmetic operators are extended for the augmented algebra.

With the help of dual number operations on numbers, it is possible to [***calculate the value of $f(x)$ while also calculating $f'(x)$ at the same time***]{style="background-color:#ACD6FF;"}.

#### Definition of Dual Number

Dual Numbers are extensions to real numbers. They are imaginary numbers and have the following form:

$$x + x'\epsilon$$ where $x$ is the real part, and $\epsilon^2 = 0$.

We can denote the new arithmetic using ***ordered pairs***: $\langle x, x' \rangle$ with ordinary arithmetics on the first component and first order differentiation arithmetic ont the second component.

**The very simple rule which states that $\epsilon = 0$ makes the whole thing work beautifully.** We can expand the calculus to all operations and always associate an expression with its derivative, then we get the following rules:

<p align="center">
<img src="https://wikimedia.org/api/rest_v1/media/math/render/svg/3850ac0df917ff2c2c3152f795e1a0c6ee9d6300">
</p>
#### How Dual number works in AD?

Let get back to our example: $$ z(x, y) = x * y + sin(x)$$

Suppose we are going to calculate $z(3, 4)$ and $z'(3, 4)$ at the same time. Because forward mode AD has to be run $n$ times where $n$ is the number of the input variable, we have to execute the gradient computation for $x$ and $y$ respectively.

***Gradients with respect to $x$***

1.  Convert 3 into dual number 3 + 1$\epsilon$ using 1 for the dual number component since $\frac{\partial x}{\partial x} = 1$
2.  Convert 4 into dual number 4 + 0$\epsilon$ using 0 for the dual number component since $\frac{\partial y}{\partial x} = 0$.
3.  $a = (3 + 1\epsilon) * (4 + 0\epsilon) = 12 + 4\epsilon$
4.  $b = sin(3 + 1\epsilon) = sin(3) + cos(3)\epsilon$
5.  $z = a + b = (12 + sin(3)) + (4 + cos(3))\epsilon$

So we get: $$z(3, 4) = 12 + \text{sin}(3)$$$$z'(3, 4) = 4 + \text{cos}(3)$$

***Gradients with respect to $y$***

1.  Convert 3 into dual number 3 + 0$\epsilon$ using 0 for the dual number component since $\frac{\partial x}{\partial y} = 0$
2.  Convert 4 into dual number 4 + 1$\epsilon$ using 1 for the dual number component since $\frac{\partial y}{\partial y} = 1$.
3.  $a = (3 + 0\epsilon) * (4 + 1\epsilon) = 12 + 3\epsilon$
4.  $b = sin(3 + 0\epsilon) = sin(3) + 0\epsilon$
5.  $z = a + b = (12 + sin(3)) + 3\epsilon$

So we get: $$z(3, 4) = 12 + \text{sin}(3)$$$$z'(3, 4) = 3$$

## Reverse-Mode AD

### What is the Reverse-Mode AD

-   The chain rule is symmetric: it doesn't care what's in the "numerator" or the "denominator." Let's just rewrite the Chain rule, but turn the derivative upside down.

    Let's differentiate a ***yet-to-be-given*** variable $s$ with respect to results of equation 5 \~ 1.

    1.  $\frac{\partial s}{\partial z} = ?$
    2.  $\frac{\partial s}{\partial b} = \frac{\partial s}{\partial z} * \frac{\partial z}{\partial a} = \frac{\partial s}{\partial z}$
    3.  $\frac{\partial s}{\partial a} = \frac{\partial s}{\partial z} * \frac{\partial z}{\partial b} = \frac{\partial s}{\partial z}$
    4.  $\frac{\partial s}{\partial y} = \frac{\partial s}{\partial a} * \frac{\partial a }{\partial y} = \frac{\partial s}{\partial a} * x$
    5.  $\frac{\partial s}{\partial x} = \frac{\partial s}{\partial a} * \frac{\partial a}{\partial x} + \frac{\partial s}{\partial b} * \frac{\partial b}{\partial x} = \frac{\partial s}{\partial a} * y + \frac{\partial s}{\partial b} * \text{cos}(x)$

-   Let's replace the derivatives $\frac{\partial s}{\partial z}, \frac{\partial s}{\partial b}, ...$ with variables $gz, gb, ...$ which we call the adjoint variables. Then we have:

    1.  $\text{g}z = ?$
    2.  $\text{g}b = \text{g}z$
    3.  $\text{g}a = \text{g}z$
    4.  $\text{g}y = \text{g}a * x$
    5.  $\text{g}x = \text{g}a * y + \text{g}b * \text{cos}(x)$

-   ***Let's substitute $s = z$, then we can get:***
    1.  $\text{g}z = 1$
    2.  $\text{g}b = 1$
    3.  $\text{g}a = 1$
    4.  $\text{g}y = x$
    5.  $\text{g}x = y + \text{cos}(x)$

***This process is the reverse-mode auto-differentiation.***

### Summary of the Reverse-Mode AD

-   ***Advantages***
    -   The cost of reverse-mode AD is $O(m)$ where $m$ is the number of output variables. For machine learning tasks, the output is a scalar function (the loss function), so the complexity of the reverse-mode AD for machine learning tasks is $O(1)$.
-   ***Disadvantages***
    -   Because all the derivative calculations appear to be going in reverse to the original program, we can't interleave the derivative calculations with the evaluations of the original expression anymore leading to a large memory footprint.
    -   No obvious to implement: how do we put the "automatic" back into reverse-mode AD? There are two design choices:

        1.  ***Dynamic approach***: a simpler way is to do this "reverse" dynamically: ***tracing***.
            -   dynamically construct the expression as the program runs (see fig 1 below).
        2.  ***Static approach***: parse the original program and then generate an ***adjoint program*** that calculates the derivatives. \* usually quite complicated. \* this way is worthwhile if efficient is critical.

### Tape-based Reverse-Mode Implementation _**[NOT FINISHED]**_

<p align="center">
<img src="https://rufflewind.com/img/reverse-mode-automatic-differentiation-graph.png"> <br>fig 1. Graph of the expression for z.
</p>

1.  Create Node and append them to an existing, growable array.
2.  Each node stores indices to their parents.

<p align="center">
<img src="https://rufflewind.com/img/reverse-mode-automatic-differentiation-links.png"> <br>fig 2. The tape representation.
<p>

We can describe each node using a struct containing two weights and two parent indices:

```julia
mutable struct Node{T, N}
    weights::Array{AbstractArray{T, N}, 1}  # store dx and dy
    dep::Array{UInt, 2}  # store parents' indices
end

mutable struct Tape
  Nodes::Array{Node, 1}
end
```

## References

1.  [Dual Numbers & Automatic Differentiation](https://blog.demofox.org/2014/12/30/dual-numbers-automatic-differentiation/)
2.  [Multivariable Dual Numbers & Automatic Differentiation](https://blog.demofox.org/2017/02/20/multivariable-dual-numbers-automatic-differentiation/)
3.  [Forward-Mode Automatic Differentiation in Julia](https://arxiv.org/pdf/1607.07892.pdf)
4.  [A simple explanation of reverse-mode automatic differentiation](https://justindomke.wordpress.com/2009/03/24/a-simple-explanation-of-reverse-mode-automatic-differentiation/)
5.  [Reverse-mode automatic differentiation: a tutorial](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation)
6.  [Automatic differentiation for machine learning in Julia](https://int8.io/automatic-differentiation-machine-learning-julia/)
