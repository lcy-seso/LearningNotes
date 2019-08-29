# [Advances in dataflow programming languages](http://www.cs.ucf.edu/~dcm/Teaching/COT4810-Spring2011/Literature/DataFlowProgrammingLanguages.pdf)

## My Takeaways

## Overall Introduction

1. Two major criticisms for Von Neumann processors are:
    1. global program counter
    1. global updatable memory

    _These will become bottlenecks for the exploitation of massive parallelism_.

1. The original motivation for research into dataflow was the <span style="background-color:#ACD6FF;">_**exploitation of massive parallelism**_</span>.
    Dataflow architecture avoids the above two bottlenecks by using:
    1. execution instructions as soon as their operands become available.
    1. local memory.

1. A program in dataflow computer is a directed graph and that data flows between instructions, along its arcs.
1. _**side-effect**_ and _**locality**_ cause troubles in compiling conventional imperative programming languages to run on dataflow hardware.
    - some research found that by restricting certain aspects of conventional imperative programming languages, like _**assignment**_, these languages could be more naturally fitted the dataflow architecture.
1. Research in the 1970s and early 1980s realized that the _**parallelism used in dataflow architectures operated at too fine a grain**_ and that <span style="background-color:#ACD6FF;">_**better performance could be obtained through hybrid von Neumann dataflow architectures**_</span>.

## The pure dataflow model

### Program representation

In the dataflow execution model, a program is represented by a directed graph.

- _**Node**_:
    - primitive instructions;
    - operation of each node is functional;
- _**Directed arcs between the nodes**_:
    - data dependencies between the instructions.

Conceptually, data _**flows as tokens**_ along the arcs.

- The behaviors of data flows are like _**unbounded first-in, first-out queues**_.
- arcs are divided into _input_ arcs and _output_ arcs.

### The execution model

|Dataflow architecture|Von Neumann architecture|
|:--|:--|
|Instructions are scheduled for execution as soon as their operands become available.<br>More than one instruction can be executed at once.|An instruction is only executed when the program counter reaches it, regardless of whether or not it can be executed earlier than this.|

1. When the program begins, special activation nodes place data onto certain key input arcs, triggering the rest of the program.
1. Whenever a specific set of input arcs of a node, called _**a firing set**_, has data on it, the node is said to be fireable.
1. A fireable node is executed at some undefined time after it becomes fireable.
it removes a data token from each node in the firing set, performs its operation, and places a new data token on some or all of its output arcs. It then ceases execution and waits to become fireable again.

## References

1. [Advances in dataflow programming languages](http://www.cs.ucf.edu/~dcm/Teaching/COT4810-Spring2011/Literature/DataFlowProgrammingLanguages.pdf)
