# [Tiramisu: A Polyhedral Compiler for Expressing Fast and Portable Code](https://arxiv.org/abs/1804.10694)

## What is Tiramisu

TIRIMISU is a DSL embedded in C++ with its _**polyhedral compiler**_. It also designs a <span style="background-color:#ACD6FF;">_**a scheduling language**_</span> featuring novel commands for _**targeting multiple high performance architectures**_.

- It provides a C++ API that allows users to write a high level architecture-independent algorithm and a set of scheduling commands that guide code generation.

Tiramisu is well-suited for implementing data parallelism.

The process is:

1. takes a high-level representation of the program
    - a pure algorithm and a set of scheduling commands
    - the scheduling commands are designed for enabling:
        1. loop transformations
        2. data layout transformations
        5. mapping buffers to different memory hierarchies
        6. for the distributed system:
            1. _**partitioning computations**_
            1. explicit communication
            1. explicit synchronization
1. applies the necessary code transformations
1. generates highly-optimized code for the target architectures

1. To simplify the implementation of the scheduling language, tiramisu explicitly divides the IR into _**four layers**_.
    - separating the architecture-independent algorithm from code transformations, data layout, and communication.

1. architectures Tiramisu targets:
    1. multicore CPUs
    2. CUDA GPUs
    3. distributed architectures
    4. FPGA

### Tiramisu vs. Halide
- Tiramisu relies on _**polyhedral representation**_ while Halide relies on _**interval-based**_ representation.
- polyhedral representation allows to
    1. _**naturally express non-rectangular iteration spaces**_
    2. support programs with cyclic data-flow graphs
    3. apply any affine transformation (iteration space skewing)
- all of the above are not naturally expressible in Halide.

- Halide cannot naturally represent non-rectangular iteration spaces, as a result:
    1. distributed Halide _**over-approximates the amount of data to communicate**_ (send and receive)
    2. over-approximates non-rectangular iteration spaces
    3. the use of interval prevents Halide from performing many complex affine transformations
        - such as _**space skewing**_
- Halide does not have dependence analysis and thus relies on conservative rules to determine whether a schedule is legal.
- Halide assumes the program has an acyclic dataflow graph, as a result, prevents users from expressing many programs with cyclic dataflow.

## Excerpts from the paper

1. Obtaining the best performance requires:
    1. complex code
    1. data layout transformations
    2. management of complex memory hierarchies
    3. efficient data communication and synchronization.
1. Highly-tuned implementations require:
    1. fusing the multiplication and addition loops
    2. applying _**<span style="background-color:#ACD6FF;">two-level tiling</span>**_ (_What's two-level tiling?_)
    3. vectorization
    4. loop unrolling
    5. _**<span style="background-color:#ACD6FF;">array packing**_ (_What's array packing?_)
    6. register blocking
    7. data prefetching
1. tuned implementations separate _**<span style="background-color:#ACD6FF;">partial tiles</span>**_ from _**<span style="background-color:#ACD6FF;">full tiles</span>**_ (_What's partial tiles and full tiles?_).
1. High performance GPU implementations require even more optimizations, including:
    1. coalescing memory accesses
    2. managing data movement between global, shared and register memory
    3. inserting synchronization primitives
1. Previous work using the polyhedral models has shown sucess in implementing:
    1. complex _**<span style="background-color:#ACD6FF;">iteration space</span>**_ transformation;
    2. data locality optimizations
    3. memory management optimizations
1. _**Fully automatic polyhedral compilers**_ do not obtain the desired level of performance since: their search techniques _**only consider only a subset of**_ the necessary optimizations and rely on _**less accurate machine**_ models.

# Reference

1. Kazushige Goto and Robert A. van de Geijn. [Anatomy of high- performance matrix multiplication](http://www.cs.utexas.edu/~flame/pubs/GotoTOMS_revision.pdf). ACM Trans. Math. Softw., 34(3):12:1– 12:25, May 2008.
1. Quilleré F, Rajopadhye S. [Optimizing memory usage in the polyhedral model]()[J]. ACM Transactions on Programming Languages and Systems (TOPLAS), 2000, 22(5): 773-815.
1. Chafi H, Sujeeth A K, Brown K J, et al. [A domain-specific approach to heterogeneous parallelism](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.644.4729&rep=rep1&type=pdf)[J]. ACM SIGPLAN Notices, 2011, 46(8): 35-46.
1. Collins A, Grewe D, Grover V, et al. [NOVA: A functional language for data parallelism](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.666.8678&rep=rep1&type=pdf)[C]//Proceedings of ACM SIGPLAN International Workshop on Libraries, Languages, and Compilers for Array Programming. ACM, 2014: 8.
1. Steuwer M, Remmelg T, Dubach C. [Lift: a functional data-parallel IR for high-performance GPU code generation](http://eprints.gla.ac.uk/146596/1/146596.pdf)[C]//2017 IEEE/ACM International Symposium on Code Generation and Optimization (CGO). IEEE, 2017: 74-85.
