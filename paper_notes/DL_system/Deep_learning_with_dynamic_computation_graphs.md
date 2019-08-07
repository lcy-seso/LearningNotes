# [Deep Learning with Dynamic Computation Graphs](https://arxiv.org/abs/1702.02181)

## Problem proposed in this paper

Create single static graphs that emulate dynamic computation graphs of arbitrary shape and size.

## Dynamic batching

TF fold is a high-level library that provides:
- _**compositional blocks**_: sub-graph. This is to simplify the creation of dynamic graph models.
- _**batch-wise**_ parallel implementations for a variety of models.

This paper proposes the dynamic batching approach.

1. The dynamic batching algorithm is implemented as graph rewriting.
1. Inputs are described as computation graphs. Dynamic batching takes DAG as its input.
1. Schedule on sub-graph, not operations.
1. batch both computation and input data
    - Nodes(operations) with the same height are independent that can be batched together.
    - `gather`, `concatenate`, etc. are inserted to collect input data. Correspondingly, `scatter`, `split`, etc. are inserted in gradient computation.
1. use `tf.while_op` to iterate over depth which relies on input data.
