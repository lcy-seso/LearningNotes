# Neural Ordinary Differential Equations

---

## Background

### What is an ordinary differential equation(ODE)?

An ODE is an equation that involves some ordinary derivatives, the ordinary here is as opposed to partial derivatives of a function.
* For example, if we know: $\frac{\partial{dx}}{\partial{dt}}(t) = \text{cos}t$, then what is the function $x(t)$
* In general, the basic principle to solve an ODE is _**always integration**_.

[An introduction to ordinary differential equations](https://mathinsight.org/ordinary_differential_equation_introduction)

---

### Input and output of ODE solver?

### The adjoint sensitive method
* http://math.mit.edu/~stevenj/18.336/adjoint.pdf
* A blog about the [Adjoint Sensitivity Method](https://advancedoptimizationatharvard.wordpress.com/2014/03/02/adjoint-sensitivity-method/)

## References

1. [DiffEqFlux.jl – A Julia Library for Neural Differential Equations](https://julialang.org/blog/2019/01/fluxdiffeq)
