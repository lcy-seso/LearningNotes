# [Dynamic Control Flow in Large-Scale Machine Learning](https://arxiv.org/pdf/1805.01772.pdf)

## Objectives and Goals

> The  author claims that after analyzing 11.7 million of uniqe computation graphs in Google, it is found that 65% of them contains some kind of conditional computation, and 5% contain one or more loops.

Confilcting design objectives for the underlying system

* scalable
* expressive and flexible
* ...

1. _**Advocate**_ that machine learning systems should provide general facilities for dynamic control flow.

1. Address the challenge of making them _**work efficiently in heterogeneous distributed**_ systems.
    * expressiveness
    * auto-differentiation.
    * parallel and distributed execution. asychrony.
    * overlap the execution of control-flow logic on CPU, compute kernels on GPU, and memory-copy operations between CPU and GPU.

Two design choices:

* The _**in-graph**_ approach
  * high-level control flow ops are compiled to computation graph.
  * TF choses this way to perform whole-program optimization.
  * allow the entire computaion to stay inside the system runtime during execution.
* The _**out-of-graph**_ approach
  * rely on control-flow features of the host language or static unrolling.
  * In heterogeneous environment, communication and syschronization with the client process can be costly.
