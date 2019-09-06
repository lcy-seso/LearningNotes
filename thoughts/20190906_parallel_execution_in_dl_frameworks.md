## Background

A framework is to map the computation into the underlying system architecture. As we approach the technological limitations, parallel/concurrent execution is a major path to scale the computation.

To achieve maximum parallelism, programs must be partitioned and assigned to available resources. The goal is to maximize the parallelism by partition program into independent executable units while minimizing communication among the executables by assigning dependent units to the same processing element.

Graph partition is an NP-complete problem. Partition and parallelism increase communications and synchronization. If you are going to do some scheduling staff, you have to define the factors you target first (the schedule space).

Two main approaches exist, static (at compile-time using global information, and system resource) and dynamic (uses run-time information on processing loads and program behavior).

The entire space: Front-end users (math/dl algorithm focused) --> front-end programming (user experiences) --> framework/library (runtime design? scale to heterogenous devices? ) --> compiler (has the whole information of low-level system resources)

## Executive Plan??

1. Need to complete the problem analysis first, so that thinking could converge to a more small and doable scope, and then converge to an idea.
    - Control-flow at the "compile-time"?
    - Control-flow essentially is a runtime design problem, but experiment runtime design requires engineering efforts to build the foundation first.

1. What to target first?
    1. A POC work based on TF (improving TF's design?)? PyTorch (improving PyTorch's design?)?
    1. An incremental fix to have the best of both TF and PyTorch?
    1. Build a new tool?

    Usually, it will be 1) --> 2) --> 3), and 3) is the ideal result.

## Some Questions

1. How to give a definition of the incumbent framework's parallel execution and the resulted problem?

    The fact is, the data parallelism, model parallelism we usually talked about is a high-level parallel execution engine. It is more to be operator-level parallel execution.

    Then here comes the next problem: the controversy for fine-grained and coarse-grained operators, so whether there is no space between these two extremes at all for scheduling? What kind of parallelism in DL workloads that could not be utilized at all (only resolve to manual code optimization) due to the framework's design?

    - _**fine-grained**_: interpreting overhead is high, then one solution has to resolve to JIT.

        - However, JIT design and (near) optimal code-generation is a much more complicated issue and requires tremendous engineering efforts.
        - Besides, we could regard kernel-gen not that highly related to specific DL algorithms.

    - _**coarse-grained**_ operator: manual optimizations.

        - It is not scalable.

2. Are we going to extend parallel execution space defined by incumbent frameworks like TensorFlow, if so, to what extent?

3. Available frameworks are not designed for scheduling (this is not in their design principle); as a result, they do not consider modeling system resources into a separate abstraction layer. Instead, frameworks directly hard-code manually tuned optimizations as the default to achieve performance.

    But scheduling happens at many different abstraction layers. We are going to target scheduling at what abstraction layer? And system resources to what granularity should be taken into consideration?

4. Whether it is possible to formalize DL program?  What are the elements of DL programming?

   If we could manage to give a formalized description to restrict the discussion into a well-defined scope, it is also a design. But be cautious, this could be an ultimate question related to the long-existing computational model research, or a pure PL research.

    DL as a tensor program? Differentiable program? And its corresponding interpreter? Imperative or declarative or bridge the gap (this is the programming paradigm problem)?

    This is just oversimplified that hidden too many technical challenges.

    As a side note: elements of programming given by SICP:

    1. primitive expressions
    1. means of the combination
    1. means of abstraction
