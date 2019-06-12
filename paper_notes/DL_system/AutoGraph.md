<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [AutoGraph](#autograph)
- [My Takeaways](#my-takeaways)
- [Related Work](#related-work)
- [Reference](#reference)

<!-- /TOC -->

# AutoGraph

[link](https://arxiv.org/pdf/1810.08061.pdf)

_**Problem to address in this paper**_

1. Make machine learning codes that are easy to write and is scalable or fast to execute.
1. Specifically, work in this paper improves the programming experience of the TensorFlow library, and demonstrate usability improvements with no loss in performance compared to native TensorFlow graphs.

_**Approaches and Key insights**_

1. Use of <span style="background-color:#DB7093;">_**staged programming**_</span> in Python, via <span style="background-color:#DB7093;">_**source code transformation**_</span>, offers a midpoint between these two library design patterns, capturing the beneﬁts of both.
    - enable staged programming in Python dispatching on runtime type information.
1. Delay all type-dependent decisions until runtime, similar to dynamic dispatch.

# My Takeaways

1. Learn some new terminologies: _**STAGED programming**_, and LMS. These techniques are also used in Swift for TF. They are worth further reading.
    - Quoted from the paper:
    > 1. Providing easier deferred execution using staged programming or multiple dispatch has a long history.
    > 1. Notable examples include (1) Lightweight Modular Staging's type-based deferred execution model; (2) the paired use of Lua and Terra to stage high-performance numerical code; (3) Julia's multiple dispatch system.
1. The workflow chosen by this paper is:
    1. Allow users to express complex _**ML programs**_(DL programs actually) using a general purpose programming language, Python, to write imperative style programs;
    1. _**Lower**_ Python codes into some optimized IRs (that can be run as fast as some hand-written alternatives)
        - upon this IR, some features that are absent from TensorFlow can be supported, such as re-entrant function calls.
    1. Work in this paper focusing on providing an approach to the "lowering" process.

# Related Work

A notable difference between work in this paper and PyTorch's Torch Script framework is: PyTorch's torch script mode lacks <span style="background-color:#DB7093;">_**staging beyond shape propagation on a dynamically-shaped graph**_</span>.

# Reference

Some random keywords:

1. [LMS](https://scala-lms.github.io/): Lightweight Modular Staging (LMS) is a runtime code generation approach.
1. [A Language and Compiler View on Differentiable Programming](https://openreview.net/forum?id=SJxJtYkPG)
1. [A Taxonomy of meta-programming systems](http://web.cecs.pdx.edu/~sheard/staged.html)
