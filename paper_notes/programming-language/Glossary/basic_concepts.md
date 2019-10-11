# [Expression and Statement in Julia](https://benlauwens.github.io/ThinkJulia.jl/latest/book.html#_expressions_and_statements)

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

# [Variable](https://en.wikipedia.org/wiki/Variable_(computer_science))

a storage location (identified by a memory address) paired with an associated symbolic name (an identifier), which contains some known or unknown quantity of information referred to as a value.

# [Value](https://en.wikipedia.org/wiki/Value_(computer_science))

the representation of some entity that can be manipulated by a program. **The members of a type are the values of that type**.

# [Object](https://en.wikipedia.org/wiki/Object_(computer_science))

- An object can be a variable, a data structure, a function, or a method, and as such, is a value in memory referenced by an identifier.

- **Object-based language**:  A language is usually considered object-based if it includes the basic capabilities for an object: identity, properties, and attributes.
- **Object-oriented language**: A language is considered object-oriented if it is object-based and also has the capability of polymorphism and inheritance.
