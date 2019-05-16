## Conclusion

* the evolution of hardware accelerators favors compiler back ends that hyper-optimize large monolithic kernels
* this reliance on high performance but inflexible kernels reinforces the dominant style of the programming model
* these programming abstractions lack expressiveness, maintainability, and modularity.

## Examples in this paper

This paper gives an example of using stridden 2D convolution to implement Capsule, which is a 7 nested loops around a multiply-accumulate operation. Array layout, vectorization, parallelization, and caching are _**extremely**_

### Problem 1, accelerators are designed for dense computation.

#### What is the current problem?

* computation expensive part is written as a dense linear algebra over multi-dimensional arrays that is comparatively easy to parallel.
* hardware accelerators are designed for dense and parallel computations.
* non-standard and regular computations suffer

#### Where the complexities lie?

In the attempt to prevent data bottlenecks: compiler _**has to**_ consider memory system as well as ALUs.
* _accelerators' parallel capabilities have become _**tightly coupled with the memory system**_._

In general, accelerator codes _**MUST**_ perform:
1. explicit scheduling through the memory hierarchy rather than relying on transparent multi-level caches.
1. complex instruction scheduling across loop iterations.

#### Current Solutions

1. monolithic kernel and then use heuristics or auto-tuning to select one of these pre-tuned implementations at runtime.
1. compilers like Tensor Comprehension and PlaidML to write custom kernels.
    * DSLs with concise syntax that resembles the math
    * _**problems**_:
        * only really suitable for _**compiling small code fragments**_.
        * compilation times are often long.
        * resulting code quality frequently does not come close to peak performance.

>_**My takeaway is, current frameworks are just not designed for searching for a better solution, and it cannot be ignored that the search space is huge.**_
