<!-- vscode-markdown-toc -->

- [Auto-vectorizing TensorFlow graphs](#auto-vectorizing-tensorflow-graphs)
  - [Related works](#related-works)
    - [Limitations & Opportunities](#limitations--opportunities)
    - [Design goals of this work](#design-goals-of-this-work)
  - [Elements of the program and assumptions](#elements-of-the-program-and-assumptions)
  - [The greedy compliation process](#the-greedy-compliation-process)
  - [Converters](#converters)
    - [For stateless operations](#for-stateless-operations)
    - [For stateful operations](#for-stateful-operations)
  - [Compiling control-flow](#compiling-control-flow)
    - [Conditionals](#conditionals)
    - [Nested loops](#nested-loops)
- [References](#references)

<!-- vscode-markdown-toc-config
    numbering=true
    autoSave=true
    /vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

# [Auto-vectorizing TensorFlow graphs](https://arxiv.org/pdf/1903.04243.pdf)

## Related works

### Limitations & Opportunities

Auto-batching is a special form of vectorization which is well-researched in loop program analysis.

1. static auto-vectorization approach can reduce the dispatch overheads.
1. vectorize across <ins>nested control flow</ins> to allow <ins>additional post-vectorization optimizations</ins>.

|Approach|Typical Examples|Limitations|
|:-|:-|:-|
|**Low-level compiler**|GCC, LLVM, polyhedral frameworks|Operate at <ins>***a much lower level of abstraction***</ins> and hence miss out on domain-specific optimizations, like leveraging algebra identities|
|***Graph-level optimization in DL compiler***|TVM, XLA, TensorFlow-Fold$^{\S}$, DyNet$^{\S}$, JAX$^\S$|1. These $^{\S}$ works use a <ins>***dynamic vectorization***</ins> approach that potentially incurs huge overheads. <br>2. Auto-batching by rewriting Python AST miss out on optimizations opportunities given that <ins>***types***</ins> may not be known statically|

### Design goals of this work

1. A dataflow programming model extended to handle <ins>state</ins> and <ins>control-flow</ins>.
1. <ins>Static auto-vectorization</ins> on a <ins>strongly typed high-level IR</ins>.

## Elements of the program and assumptions

1. A directed <ins>**cyclic**</ins> graph represents the compuation.
    - the graph can have cycles due to loops
1. Types of all input and output tensors are known at compile time.
    - full or partial shapes **may be** known statically

    >Is "shape" part of a type? How shape is encoded? Under what conditions, the "shape" is fully specified, and under what conditions, the "shape" is partially knowns? and how partially known shape affact program analysis.

1. `parfor`
    - the iterations don't need to run in sequential order.
    - the result of `parafor` is the same under any ordering or interleaving of the execution of the different iterations.

## The greedy compliation process

1. Given a Tensorflow dataflow graph (which may contain cycles), convert it to a DAG by <ins>converting each control block into a single node</ins>.
1. Traverse this DAG in topological order, for each node $\eta$, generate a new set of node $\hat{\eta}$ that efficiently implement the functionality of running $\eta$ in `parfor`.
1. A recursive style treatment is required 

## Converters

### For stateless operations

- $X$ is a tensor.
- $\hat{X}$ represent the verctorized version of $X$ that stores $n$ different versions of $X$ in some layout.

Leverage <ins>loop-invariance of tensors</ins> to generate different codes based on what combinations of input values are loop invariant.

|Treatment|Explanation|Examples|
|:--|:--|:--|
|Extra reshapes is required.|Make sure broadcasting works for the generated codes|Component-wise operations|
|Different conversion exists according to the number of operands that are loop invariant.|`matmul` to `batchedMatmul`|`matmul`|
|Require padding|/|`conv` requires padding.
|Renumber axes||`reduction`, `concatenate`|

### For stateful operations

The safety mechanism introduced.

## Compiling control-flow

### Conditionals

### Nested loops

# References

1. [Efficient Selection of Vector Instructions using Dynamic Programming](https://www.cs.rice.edu/~vs3/PDF/Barik-Zhao-Sarkar-MICRO-2010.pdf)
1. [Vector processor](https://en.wikipedia.org/wiki/Vector_processor)
    >In computing, a vector processor or array processor is a central processing unit (CPU) that implements an instruction set where its instructions are designed to operate efficiently and effectively on large one-dimensional arrays of data called vectors.

    Vector instruction exploits data-level parallelism by performing multiple identical scalar operations in parallel.