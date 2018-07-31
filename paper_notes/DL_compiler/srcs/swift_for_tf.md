<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Swift for TensorFlow](#swift-for-tensorflow)
	- [Goals](#goals)
	- [Swift](#swift)
	- [Graph Program Extraction](#graph-program-extraction)
	- [The TensorFlow module](#the-tensorflow-module)
	- [Runtime Entry Points for Extraction](#runtime-entry-points-for-extraction)
	- [AD](#ad)
- [Graph Program Extraction](#graph-program-extraction)
	- [related works](#related-works)
		- [Explicit graph building APIs](#explicit-graph-building-apis)

<!-- /TOC -->

# Swift for TensorFlow

* [Design Doc](https://github.com/tensorflow/swift/blob/master/docs/WhySwiftForTensorFlow.md)
* [Why Swift](https://github.com/tensorflow/swift/blob/master/docs/WhySwiftForTensorFlow.md)

---

* The main part in Swift for TensorFlow is Graph Program Extraction, which is a compiler.
* The transformation algorithm can also be used to extract _**any computation that executes asynchronously with the host program**_ while communicating through sends and receives.
* This is useful for _**anything that represents computation as a graph**_.

- Enable the [Source Code Transformation](https://en.wikipedia.org/wiki/Automatic_differentiation#Source_code_transformation_(SCT)) techniques to AD.
  - The graph based approach is hard to translate control flow constructs in the host language to the computation graph, make it difficult to perform optimizations.
- Natural interoperability between accelerated tensor operations and arbitrary non-tensor host codes.

## Goals

1. Provide the best possible user experience for machine learning researchers, developers and production engineers.
1. Provide usable access to high performance computation.
1. Eliminate the compromises that have historically forced developers to choose between performance or usability.
1. Provide a simple, predictable, and reliable programming model that is easy to intuitively understandthe, and the compiler can reinforce with warnings, other diagnostics and potentially optimization techniques.

_**New things could be achieved if we could enhance the compiler and language**_.

* Allows an ML programmer to write simple imperative code using normal control flow.
* Have the compiler do the job of building a TensorFlow graph.
* Performance benefits of graph abstractions.
* Allow other compiler analysis to automatically detect bugs (like shape mismatches) in user code without even running it.

* The project goal is to improve usability of TensorFlow
* Graph execution: performance.
* Improved usability at _**every level of the stack**_.
* Improving the usability of high performance accelerators will enable even faster breakthroughs in ML.

## Swift

* The "scripting language feel" combined with high performance is very useful for machine learning.
* Rely on the principle: [Progressive Disclosure principle](https://www.nngroup.com/articles/progressive-disclosure/)
  * aggressively factors the cost of complexity onto the people who benefit from that complexity.
* A lot of the Swift experience is actually defined in the standard library, not the compiler itself.
* _that Swift is just "syntactic sugar for LLVM". This capability is very important_.
* _**Design choices about the user experience are not baked into the language or compiler**_.

## Graph Program Extraction

1. The compiler _**finds the tensor operations in the code**_.
1. The compiler _**desugars high-level abstractions**_ (like structs, tuples, generics, functions, variables, etc) that connect tensor operations through a process called "deabstraction".
    * the tensor operations are directly connected to each other through [SSA](https://en.wikipedia.org/wiki/Static_single_assignment_form) dataflow edges
    * the tensor operations are embedded in a control flow graph represented in the Swift Intermediate Language (SIL).
1. _**Remove the tensor operations from the host code**_:
    * A transformation called "partitioning" extracts the graph operations from the program and builds a new SIL function to represent the tensor code.
    * New calls are injected that call into the [new runtime library](https://github.com/tensorflow/swift/blob/master/docs/DesignOverview.md#runtime-entry-points-for-extraction) to start up TensorFlow.
    * Rendezvous to collect any results, and send/receive values between the host and the tensor program as it runs.
1. Once the tensor function is formed, it has some transformations applied to it, and is _**eventually emitted to a TensorFlow graph**_.

## The TensorFlow module

* One most significant design constraint is that:

  * _**we don’t want users of Swift for TensorFlow to write code that accidentally causes unnecessary copies back and forth between the host and the accelerator**_.
    * provide two primary concepts: "arrays" and "tensors".
    * "arrays" should be thought of as data in the host program, whereas "tensors" are values that are primarily managed by TensorFlow.

## Runtime Entry Points for Extraction

1. The Graph Program Extraction algorithm splits the tensor operations out to a TensorFlow graph.
1. The TensorFlow graph is serialized to a protobuf and encoded into the program’s executable.
1. The Graph Program Extraction algorithm rewrites the host code to insert calls to "start tensor program", "finish tensor program", and "terminate tensor program" runtime entry points.

>_**The most significant unimplemented piece of our compiler and runtime model is support for sending and receiving data between co-executing asynchronous host and TensorFlow programs**_.

## AD

* To develop more powerful techniques to improve user experience in failure cases:
  * enable differentiating custom data structures, recursion, and higher-order differentiation.
* As such, Swift for TensorFlow builds a stand-alone AD feature for Swift
  * completely independent of the standard TensorFlow implementation of AD.
  * completely independent of TensorFlow support in Swift.
* Have Swift AD support _**arbitrary user-defined types**_.
  * making TensorFlow's Tensor confrom to the AD system.
* Automatic differentiation in Swift is _**a compiler IR transformation**_ implemented with static analysis.
  * When differentiating a function in reverse mode, the compiler produces a separate functions that contain the corresponding "primal code" and "adjoint code", which in turn compute the partial derivatives of the model output with respect to the input parameters.

> we plan to support full-fledged control flow and discuss the need for forward-mode AD with the community

---

# Graph Program Extraction

In proof systems, it is well known that it is difficult to make a static analysis that is both [sound and complete](https://en.wikipedia.org/wiki/G%C3%B6del%27s_incompleteness_theorems), meaning that you get a choice of 1) an unsound model that sometimes produces incorrect results but handles all programs, 2) a sound model that is always correct but handles a limited set of programs. Since we are using static analysis for code generation, we require the analysis to be correct! Let’s explore the conservatively correct but incomplete model.

## related works

* [Design Doc](https://github.com/tensorflow/swift/blob/master/docs/GraphProgramExtraction.md)

Machine learning models contain a mix of two different kinds of code:

* tensor number crunching logic
* other general codes

_**All of the approaches used by machine learning frameworks are just different ways for the system to "find" the tensor logic, extract it out, and send it to an accelerator**_.

### Explicit graph building APIs

* Introduce a graph abstraction and introduce APIs for building and executing that graph

* _**Advantage**_
  * Many important performance optimizations become possible once the computation is expressed as a graph
    * support for different accelerator hardware
    * distribution across multiple accelerators

* _**Downside**_
  * significant usability is sacrificed to achieve optimized performance.
      * dynamic control flow is difficult to express
      * difficult to step through the codes, difficult to understand the bug and the ultimate fix. Because tack trace are produced through a bunch of runtime code users didn’t write.
