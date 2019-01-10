### [Back Edges](https://stackoverflow.com/questions/44494426/back-edges-in-a-graph/44494705)

Given a DFS tree of a graph, a Back Edge is an edge that connects a vertex to a vertex that is discovered before it's parent.

<p align="center">
<img src="images/1920px-Tree_edges.svg.png" width=50%>
</p>

### [Domination relationship](https://en.wikipedia.org/wiki/Control_flow_graph)

- A block M **dominates** a block N if every path from the entry that reaches block N has to pass through block M.
  - The *entry block* (through which control enters into the flow graph) **dominates** all blocks.
- In the reverse direction, block M **postdominates** block N if every path from N to the exit has to pass through block M.
  - The *exit block* (through which all control flow leaves) **postdominates** all blocks.

### [Use-define chain / Definition-use chain](https://en.wikipedia.org/wiki/Use-define_chain)

* A Use-Definition Chain (UD Chain) is a data structure that consists of a use, U, of a variable, and all the definitions, D, of that variable that can reach that use without any other intervening definitions.

* A counterpart of a UD Chain is a Definition-Use Chain (DU Chain), which consists of a definition, D, of a variable and all the uses, U, reachable from that definition without any other intervening definitions.

Both UD and DU chains are created by using a form of static code analysis known as data flow analysis.

Knowing the use-def and def-use chains for a program or subprogram is a prerequisite for many compiler optimizations, including constant propagation and common subexpression elimination.

### [Nominal typing / Nominal subtyping](https://en.wikipedia.org/wiki/Nominal_type_system)

**Nominal typing** means that two variables are type-compatible if and only if their declarations name the same type.

**Nominal subtyping** means that one type is a subtype of another if and only if it is explicitly declared to be so in its definition.

### [Abstract type](https://en.wikipedia.org/wiki/Abstract_type)

### [Object](https://en.wikipedia.org/wiki/Object_(computer_science))

- An object can be a variable, a data structure, a function, or a method, and as such, is a value in memory referenced by an identifier.

    - **[Variable](https://en.wikipedia.org/wiki/Variable_(computer_science))**: a variable or scalar is a storage location (identified by a memory address) paired with an associated symbolic name (an identifier), which contains some known or unknown quantity of information referred to as a value.
    - **[Value](https://en.wikipedia.org/wiki/Value_(computer_science))**: a value is the representation of some entity that can be manipulated by a program. **The members of a type are the values of that type**.

- **Object-based language**:  A language is usually considered object-based if it includes the basic capabilities for an object: identity, properties, and attributes.
- **Object-oriented language**: A language is considered object-oriented if it is object-based and also has the capability of polymorphism and inheritance.

---

- [Expression and Statement in Julia](https://benlauwens.github.io/ThinkJulia.jl/latest/book.html#_expressions_and_statements)
    - _**An expression**_ is a combination of values, variables, and operators.
        ```julia
        julia> 42
        42
        julia> n
        17
        julia> n + 25
        42
        ```
        - When you type an expression at the prompt, the REPL **evaluates** it.
        - An `expression` returns a value.


    - _**A statement**_ is a unit of code that has an effect, like creating a variable or displaying a value.
        ```julia
        julia> n = 17    #  assignment statement
        17
        julia> println(n)    #  print statement
        17
        ```
    - When you type a statement, the REPL **executes** it, which means that it does whatever the statement says.
    - In general, statements don’t have values.

    `while`, `for` and `if` are not `statements` but `expression` in Julia so they return a value.

---

-   [Is C actually Turing-complete?](https://cs.stackexchange.com/questions/60965/is-c-actually-turing-complete)
