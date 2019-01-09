<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Goals](#goals)
- [Background](#background)
	- [Problem setting && Notation](#problem-setting-notation)
		- [Notation](#notation)
		- [Function call Tracing: dynamic vs. static approaches](#function-call-tracing-dynamic-vs-static-approaches)

<!-- /TOC -->

## Goals

- Algorithmic differentiation based on source code transformation in particular of the Static Single Assignment(SSA) form used by the modern compiler.
- Support control flow, nesting, mutation, recursion, data structures, higher-order functions.

<span style="background-color:#ACD6FF;">_**Differences with Tapenade**_</span>:
1. Zygote never exposes source transformations to the user</span>.
1. Zygote can naturally express nested derivatives.

## Background

AD systems face a tradeoff between providing an expressiveness, full-featured programming model and producing optimized programs.

1. _**Tracing**_ preserves a host language semantics, but adds overhead and precludes many optimizations.
1. _**Source-to-source**_ technique resolves this tradeoff to some extent but has previously been (1) cumbersome or (2) supported only limited semantics.

### Problem setting && Notation

#### Notation

Denote $\frac{\partial f}{\partial x}$ as $\bar{x}$.

_**Inputs**_: a scalar function: $y = f(x_1, x_2, ...)$
_**the AD interface**_: a high-order function $\mathcal{J}$: $ y, \mathcal{B}_y = \mathcal{J}(f, x_1, x_2, ...) $
_**Outputs**_:

1. $y$: the result of forward computation.
1. $\mathcal{B}_y$: the _**backpropagator**_ which accepts gradient with respect to $y$ and returns gradients with respect to each input $x$, then $\bar{x_1}, \bar{x_2}, ... = \mathcal{B}_y(\bar{y})$

#### Function call Tracing: dynamic vs. static approaches

_**You can get background information about Tape-(Wengert list) based AD from [this blog](https://rufflewind.com/2016-12-30/reverse-mode-automatic-differentiation).**_

Let's take the implementation of $x^n$ for example:

```julia
function pow(x, n)
  r = 1
  while n > 0
    n -= 1
    r *= x
  end
end
```

- Typical AD system uses a tracing approach based on operator overloading.
  - $x$ is wrapped in a new object.
  - Overloaded methods such as $\times$ (multiplication) not only just multiply $x$ and $y$, but _**records the operation and its inputs**_. (_**This what Flux does.**_)

- Dynamic vs. Static approaches

  ||dynamic approaches|static approaches|
  |--|--|--|
  |**Explantaions**|Interleave tracing with evaluation of the primal.|Record _**a graph**_ and evaluate it instead of the original program.|
  |**Advantages**|Preserve host language's expressive semantics.|Evaluate the host code _**only once**_.|
  |**Disadvantages**|Traces are en extremely inefficient program representation.<br>Pay heavy cost of building and manipulating the graph anew at every iteration|Come at a high cost to expressiveness. _**Control-flow has to be inserted into the Tape to obtain richer behaviors**_.|

- The limitations of the static approach <span style="background-color:#ACD6FF;">are not fundamental to AD</span>, but instead are <span style="background-color:#ACD6FF;">limitations of the symbolic form or the language we are differentiate -- the Wengert list</font>.

## Solution in this paper: Generalising Wengert List


## References

1. [The Tapenade Automatic Differentiation tool: principles, model, and speciﬁcation](https://hal.inria.fr/hal-00913983/document)
1. [Differentiable Programming](http://www.cs.nuim.ie/~gunes/files/Baydin-MSR-Slides-20160201.pdf)
