[TOC]

# JANUS: Fast and Flexible Deep Learning via Symbolic Graph Execution of Imperative Programs

[link](https://arxiv.org/pdf/1812.01329.pdf)

## My Takeaways and Some Thoughts

1. <span style="background-color:#DB7093;">_**Unlike a language's JIT**_</span>, it is not necessary to optimize the host language execution, but only the computation workload.
1. What is speculative optimization doing in PL research? It seems that we can borrow wisdom from this area to strike a balance between usability and efficiency.
1. Not all dynamic semantics of a dynamic imperative language can be successfully converted into a symbolic data flow program.
1. Work in this paper prototyped on Python and TensorFlow. Dynamic language like Julia has a more sophisticated type system, thus more friendly for optimization. So, suppose we just reproduce this work in Julia, what new things will be achieved?
1. It goes back to the principle: optimization based on specialization to the extreme. Making aggressive assumptions by execution.

|Python|Julia|
|--|--|
|Type inference is a problem for symbolic graph extraction.|It is typed internally.|
|Python is a multiple paradigm PL. Many language features are hard to be converted into DFP.|Julia is a procedure. Values in Julia are true objects and is easy to introspect.|
|Interop with C/C++ has much overhead.|Julia object's ABI is compatible with C.|

## Overall

### Problem Proposed

Diverged efforts for optimizing performance and improving usability.

### Goal

* Fact 1: Imperative DL program written in Python is flexible and highly programmable.
* Fact 2: Symbolic data flow program represented by TensorFlow is efficient.

Achieve the best of both worlds.

### Solution

* Receive an imperative DL program as input and create symbolic graphs with _**<span style="background-color:            #DB7093;">speculative program context assumptions</span>**_.
    * makes environment assumptions on the program context <span style="background-color:#ACD6FF;">??</span>
    * incorrect assumption results in an invalidation of a symbolic graph, casing JANUS falls back to imperative execution to guarantee correctness.
* Transparently convert an imperative DL program written in Python into an efficiently executable symbolic dataflow graph.
    * _**Guarantee correctness**_ (The first concern).
    * Performance.

### Evaluation Methods in this Paper

- JANUS based on TensorFlow
- Evaluate JANUS with 11 imperative DL programs: CNN/RNN/Recursive NN, GAN and RL.
- Compare the models' throughput with TensorFlow eager.

## Related Works

1. PyTorch JIT
    * Limited Python Semantics (what the design choice is??)
1. TF's Eager
1. JAX support only pure-and statically-composed functions
1. Swift4TF (I think authors' comments on S4TF is not fair)
1. AutoGraph enabled TF

## Detail

### Chanllenges

1. Data flow program (DL frameworks) are considered to be static which consists of a restrictive set of operations and lack the dynamic semantics of the PL.

    In Python, below semantics are difficult to be captured in a dataflow graph:
    1. dynamic control flow
        * conditional branch and loop constructs on intermediate values.
    1. Dynamic types
        1. Types can only be determined at execution time.
        1. Non-numerical types are hard to be represented in DFP.
    1. Impure functions/mutation
        * _**In this problem, almost all the work prefer functional style**_ (see [1], and many others).

### Proposed Approach

* **Input**
    The DL program is assumed to be written using the API and programming model of an existing imperative DL model.
* **Workflow**

    1. _**Static analysis**_: Extract the main neural network computation part, over which perform AD.
    1. _**Speculative graph generation and execution**_:
        1. Fast path (common cases)
            - execute the program imperatively for several iterations to avoid making hasty generalizations.
            - the _profiler_gathers runtime information to make a reasonable assumption.
            - convert the program into a symbolic dataflow graph.
                - Traverse AST to generate: (1) graph element for the AST node; (2) assertion operations that can validate the context assumption at runtime.
                - AD
                - Optimization
            - Execute
        1. Accurate path (rare cases)
            - If the assumption fails, fall back to the imperative executor.
                - Simply aborting graph execution is erroneous, states (what is defined as "states"?) are updated in an all-or-nothing manner.
                - Semantics contain invisible state mutation is not converted to a symbolic graph.

#### Symbolic Graph Generation Details

TF back-end has limited semantics, it is inevitable a superset to a small set, so the mapping is constraint by the back-end primitives.

##### Dynamic Control Flow

1. conditional branch --> TF's `switch`/`merge` which is originated from data flow program.
1. while/for --> TF's context construct.
1. InvokeOp --> From Author's work [2]

##### Speculative graph generation

Make aggressive speculations. unrolling or inlining, add `AssertOp` to check the assumption, if it is not meet, fall back into the runtime.

#### Dynamic Type

This problem is from Python's language design but is not a problem for dynamic language that has a more sophisticated language.

* Work in this paper implements a very simple and naive type inference to infer the types of some expression.
* convert:
    1. numerical Python values into TF tensors.
    1. non-numerical values into scalar tensors which hold pointers to the corresponding Python values.

#### Speculative graph generation

##### For expressions types of whose returned value cannot be inferred

1. Only a subset of Python semantics is able to be converted into a symbolic graph.
    - eg. access to Python obj's internal state.
    - This is not a problem in Julia, because Julia values are true objects and have a unified interface to introspect the type definition.
1. A recursive function call is challenging.

Solution: _**Profiler**_ observes the types of expressions during executions.

##### Specialize shapes of Tensor before constructing the graph.

#### Impure Functions

* Does not mutate global states in place on the fly. Create local copies of global states, and mutates only the local copies during symbolic graph execution.
    - <span style="background-color:#DB7093;">Mybe we can try to make the best use of Julia's or a PL's scope constrain?</span>?

## Reference

1. [Automatic differentiation in PyTorch](https://openreview.net/pdf?id=BJJsrmfCZ)
1. [Improving the Expressiveness of Deep Learning Frameworks with Recursion](https://arxiv.org/pdf/1809.00832.pdf)
1. [Why is Python slow](http://blog.kevmod.com/2016/07/why-is-python-slow/)
1. A discussion about [These are a few of my Favourite Things (that are coming with Julia 1.0)](https://news.ycombinator.com/item?id=17203825)
