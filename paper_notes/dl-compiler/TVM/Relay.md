<!-- vscode-markdown-toc -->
- [Elements in Relay](#elements-in-relay)
  - [Variable](#variable)
  - [Functions](#functions)
  - [Closures](#closures)
  - [Operators](#operators)
- [Relay's Type System](#relays-type-system)
  - [Tensor Type](#tensor-type)
  - [Tuple Type](#tuple-type)
  - [Algebraic Data Types](#algebraic-data-types)
- [References](#references)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

The Relay IR is a pure, expression-oriented language.

# Elements in Relay

## Variable

Relay explicitly distinguishes between local and global variables both in the AST and in the text format.

This explicit distinction makes certain optimizations easier to implement. For example, inlining a global definition requires no analysis: simply substituting the definition suffices.

## Functions

1. Functions in Relay act similarly to procedures or functions in other programming languages and serve to generalize the concept of a named subgraph.

1. Functions in Relay are higher-order, which means that a function can be passed as an argument to a function or returned by a function, as function expressions evaluate to closures, which are values like tensors and tuples.

## Closures

A function expression evaluates to a closure. Closures are values that are represented as a pair of a local environment (storing the values for all variables defined outside the scope of the function’s body) and the function itself.

## Operators

- An operator is a primitive operation, such as _add_ or _conv2d_, **not defined in the Relay language**.
- Operators are declared in the global operator registry in C++. **Many common operators are backed by TVM’s Tensor Operator Inventory**.

# Relay's Type System

1. statically typed and type-inferred.
1. _dependent type_ for shapes.
1. treate tensor shapes as types.

Reasoning about tensor types in Relay is encoded using **type relations**.

## Tensor Type

## Tuple Type

## [Algebraic Data Types](https://tvm.apache.org/docs/reference/langref/relay_adt.html#adt-overview)

# References

1. [Expressions in Relay](https://tvm.apache.org/docs/reference/langref/relay_expr.html#dataflow-and-control-fragments)
1. [Rely's Type System](https://tvm.apache.org/docs/reference/langref/relay_type.html#adt-typing)