# [Generic Programming](https://en.wikipedia.org/wiki/Generic_programming)

Generic programming describes a programming paradigm whereby:

1. Fundamental requirements on types are abstracted from across concrete examples of algorithms and data structures and formalized as concepts.
1. Generic functions implemented in terms of these concepts, typically using language genericity mechanisms, for example:

    - parametric polymorphism in ML, Scala, Haskell and Julia
    - templates in C++ and D
    - parameterized types in the influential 1994 book Design Patterns.

# [Programming Language Support for Genericity](https://en.wikipedia.org/wiki/Generic_programming)

Genericity is implemented and supported differently in various programming languages.

1. [Forth](https://en.wikipedia.org/wiki/Forth_(programming_language)): Exposing the compiler behaviour and therefore naturally offers genericity capacities.

# [Ploymorphism](https://en.wikipedia.org/wiki/Polymorphism_(computer_science))

The provision of a single interface to entities of different types or the use of a single symbol to represent multiple different types.

- **Ad hoc polymorphism**: defines a common interface for an arbitrary set of individually specified types.
  -  function overloading or operator overloading
- **Parametric polymorphism**: when one or more types are not specified by name but by abstract symbols that can represent any type.
- **Subtyping** (also called subtype polymorphism or inclusion polymorphism): when a name denotes instances of many different classes related by some common superclass.

_**Julia is a dynamically typed language and doesn't need to make all type decisions at compile time, many traditional difficulties encountered in static parametric type systems can be relatively easily handled**_.
