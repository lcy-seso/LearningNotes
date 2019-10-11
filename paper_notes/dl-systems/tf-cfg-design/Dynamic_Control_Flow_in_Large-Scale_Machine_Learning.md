# [Dynamic Control Flow in Large-Scale Machine Learning](https://arxiv.org/pdf/1805.01772.pdf)

## My Takeaways

1. The paper states that three distinctive features are provided by their solution:

    1. the branches of conditionals and bodies of loops <span style="background-color:#ACD6FF;">_**can be**_</span> partitioned across many machines to run on a set of heterogeneous devices.
    1. support AD and distributed gradient computations.
    1. the paper's choice of <span style="background-color:#ACD6FF;">_"non-strict" semantics</span> enables multiple loop iterations to execute in parallel across machines and to overlap compute and I/O operations.

    My questions are:

    1. for contribution 1, we partition the computation graph to make it run more efficiently instead of making it more slowly. According to common sense, conditional branches and loops are intrinsically sequential, it seems that simply partitioning them across many machines will just slow down the computation. So, is TensorFlow able to make the best use of parallelism embedded in such a control-flow graph?
    1. for contribution 3, how to understand this "non-strict" semantics. It seems that they confront with some struggling design choice, so what are the pros and cons.

1. This paper names MoE as an example. It is worthy of further reading.

1. Challenges for providing general facilities for dynamic control-flow
    - work efficiently in heterogeneous distributed systems.

1. Claims and statements from the paper:
    > When designing TensorFlow, we favored the in-graph approach, because it has advantages at both compile and run time. With this approach, the machine learning system processes a single unified dataflow graph and can perform whole-program optimization.

    I always think that many papers of DL frameworks just use terminology with PL research interchangeably, but they are not the same thing. For example, frameworks do not have a real compile time, it is just the graph construction process, so the framework does not make the best use of "compile-time optimization" as in PL. So, is there anything we can borrow from the philosophy and wisdom of our PL research to improve the design of the DL framework?

## Objectives and goals

> The author claims that after analyzing 11.7 million of unique computation graphs in Google, it is found that 65% of them contains some kind of conditional computation, and 5% contain one or more loops.

Conflicting design objectives for the underlying system

* scalable
* expressive and flexible
* ...

1. _**Advocate**_ that machine learning systems should provide general facilities for dynamic control flow.

1. Address the challenge of making them _**work efficiently in heterogeneous distributed**_ systems.
    * expressiveness
    * auto-differentiation.
    * Parallel and distributed execution. asynchrony.
    * overlap the execution of control-flow logic on CPU, compute kernels on GPU, and memory-copy operations between CPU and GPU.

Two design choices:

* The _**in-graph**_ approach
  * high-level control flow ops are compiled to computation graph.
  * TF chooses this way to perform whole-program optimization.
  * allow the entire computation to stay inside the system runtime during execution.
* The _**out-of-graph**_ approach
  * rely on control-flow features of the host language or static unrolling.
  * In a heterogeneous environment, communication and synchronization with the client process can be costly.

## Programming Interface

TF chooses the `in-graph` approach.

1. The operators are embedded as functions in the host programming languages
1. control-flow constructs are also operators
    - the most basic ones:
        1. `cond`
        1. `while_loop`
    - high-order functions
        1. `map_fn`, `foldr`, `scan`
        1. <span style="background-color:#ACD6FF;">_**the high-order functions are actually defined in terms of `while_loop`**_</span>

1. Data structures are important tools for organizing dynamic computation.
    - mutable variables, queues
    - augment TF with a new type `TensorArrays` with _**random read/write**_ accesses

# References

1. [TensorFlow White Paper Notes](https://github.com/samjabrahams/tensorflow-white-paper-notes)
1. [Implementation of Control Flow in TensorFlow](http://download.tensorflow.org/paper/white_paper_tf_control_flow_implementation_2017_11_1.pdf)
