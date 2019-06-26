# [Compiling machine learning programs via high-level tracing](https://github.com/google/jax)

## My Takeaways

A big advantage of tracing-based AD is it is straightforward to implement and is able to preserve almost all host language's language feature, however, a big disadvantage is it linearizes the original program by executing it, as a result, the structure of original computation just be stripped out.

The idea of equipping tracer with a tracing caching, and once a cache miss is encountered re-tracing somehow ease this tension. It is an interesting design idea.

## Overview

JAX stands for "just after execution".

### Problem to address

To ease the tension of maximizing access to machine FLOPs and facilitating research-friendly programmability.

A dilemma:
- Dynamic languages like Python offer convenient programming but are too unconstrained to enable optimized code generation.
- Emerging supercomputer platforms while present new hardware capabilities magnify the programmability challenges.

### Approach

1. JAX's design is in light of the PSC observation:
    > _**ML workloads often consist of large, accelerable, pure-and-statically-composed (PSC) subroutines orchestrated by dynamic logic**_.
    - _**pure**_: does not have the side-effect
    - _**statically-composed**_: computation can be described as _**a static data dependency graph**_ on a set of primitive functions that are _**themselves accelerable**_.
1. JAX is built atop:
    1. Python's Autograd's tracing library.
    1. XLA for array-level program optimization and code generation.
1. JAX equips the tracer with a tracing caching:
    1. trace caching creates a monomorphic signature for the parameters to the traced computation
        - newly encountered _array element types_, _array dimensions_, or _tuple members_ trigger a re-compilation.
    1. on a trace _**cache miss**_, JAX executes the corresponding Python and traces its execution into a graph of primitive functions with _**static data dependencies**_.
