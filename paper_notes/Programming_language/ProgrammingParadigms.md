# Programming paradigm

1. Managing complexity is, arguably, a programmer’s main concern.
1. A major source of complexity in a program is “state”
    - state: what a program has to keep track of as it moves forward through time.

_**Approaches to managing complexity and, specifically, state are called programming paradigms.**_

Here, we compare three specific paradigms: **imperative, functional, and object-oriented**.

## Imperative

- Defines the solution to a problem as a series of steps. The computer steps through each line of code, executing it and moving on to the next step.

- Often change the state of the program on each line, assigning new variables and referring to or changing old ones.

- [_**advantages**_]
    - Intuitive for solving small problems.

- [_**disadvantage**_]
    - Quickly become unmanageable as programs become larger.
    - Complex problem cannot be solved.
    - Less efficient and less productive.
    - **Parallel programming is not possible**.

Imperative programming is divided into three broad categories: Procedural, OOP and parallel processing.

### Object-Oriented

- Deals with state by designating certain functions that operate specifically on that state.
- Objects are a combination of state, or data, with functions that work specifically on that data.
- The object-oriented approach allows only certain parts of the program to operate on certain pieces of data.

## Declarative programming paradigm

It is divided as Logic, Functional, Database.

The declarative programming is a style of building programs that expresses logic of computation _**without talking about its control flow**_.

### Functional

- Solutions are defined as a series of functions that pass values to one another, leading to a series of transformations.

- Parts of the program dealing with state, if any, tend to be isolated

- A goal of functional programming is predictability
    - The functions, if given a certain input, should always return the same output.
    - Function can be replaced with their values without changing the meaning of the program.

## Summary

- Imperative programs have no special way of dealing with state, and tend to transform state frequently.
- Functional programs tend to avoid and isolate state.
- Object-oriented programs couple state with functions that work on that state.

## References

1. [The principal programming paradigms](https://www.info.ucl.ac.be/~pvr/paradigmsDIAGRAMeng108.pdf)
1. [An Introduction to Programming Paradigms]( https://digitalfellows.commons.gc.cuny.edu/2018/03/12/an-introduction-to-programming-paradigms/)
