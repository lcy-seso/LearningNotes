# Types

- _**What is the Type system?**_

    Type systems are generally formulated as collections of rules for checking the “con- sistency” of programs.

- _**Purpose of a type system?**_

    1. Reduce possibilities for bugs in computer programs.
        - Tyeps define interfaces between different parts of a computer program.
        - Then whether parts have been connected in a consistent way can be checked.
            - This checking can happen **statically** (at compile time), **dynamically** (at run time), or as **a combination of static and dynamic** checking
    1. Express business rules
    1. Enable certain compiler optimizations
    1. Allow for multiple dispatch
    1. Provide a form of documentation (??)

- _**Classification of Type System**_

    1. _**Untyped**_: programs simply execute ﬂat out; there is no attempt to check “consistency of shapes”.
    1. _**Typed**_: some attempt is made, either at compile time or at run-time, to check shape-consistency.

    ||_**Statically Checked**_|_**Dynamically Checked**_|
    |--|--|--|
    |_**Strongly Typed**_|ML, Haskell, Java (almost),Pascal (almost)|Lisp, Scheme|
    |_**Weakly Typed**_|C, C++|Perl|

## Static vs. Dynamic Types

_**Types are regular objects existing at runtime!**_

## Nominal and Structural Type

Most statically-typed production languages, including AdaLanguage, CeePlusPlus, JavaLanguage, and CsharpLanguage, are (for the most part) nominally typed. Many of the statically-typed FunctionalProgrammingLanguages, such as HaskellLanguage and the numerous variants of ML, are structurally typed. C++ is an interesting hybrid, actually.


## References

1. [Types and Programming Languages](http://ropas.snu.ac.kr/~kwang/520/pierce_book.pdf)
1. [Types in C/C++ and Julia](https://medium.com/@Jernfrost/types-in-c-c-and-julia-ce0fcbe0dec6)
