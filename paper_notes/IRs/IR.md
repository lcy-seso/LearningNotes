<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Intermedia Representation](#intermedia-representation)
    - [My Take-aways](#my-take-aways)
        - [Linear IRs vs. Graphical IRs](#linear-irs-vs-graphical-irs)
        - [About the IR design](#about-the-ir-design)
    - [Why different IRs?](#why-different-irs)
    - [Categories of IRs](#categories-of-irs)
        - [Structural organization](#structural-organization)
        - [The level of abstraction](#the-level-of-abstraction)
        - [Naming discipline](#naming-discipline)
    - [Graphical IRs](#graphical-irs)
        - [1. Syntax-related Trees (including parse tree and abstract syntax trees)](#1-syntax-related-trees-including-parse-tree-and-abstract-syntax-trees)
        - [2. Directly Acyclic Graphs](#2-directly-acyclic-graphs)
        - [3. CFG (basic block-level CFG/statement-level CFG)](#3-cfg-basic-block-level-cfgstatement-level-cfg)
        - [4. Dependency Graph](#4-dependency-graph)
        - [5. Call Graph](#5-call-graph)
    - [Linear IRs](#linear-irs)
        - [Categories of Linear IRs](#categories-of-linear-irs)
    - [Mapping values to names](#mapping-values-to-names)
        - [1. Naming Temporary Values](#1-naming-temporary-values)
        - [2. Static Single-Assignment Form](#2-static-single-assignment-form)
        - [3. Memory Model](#3-memory-model)
    - [Symbol Tables](#symbol-tables)
        - [What is the symbol table?](#what-is-the-symbol-table)
        - [Benefits of introducing symbol tables](#benefits-of-introducing-symbol-tables)
        - [Some issues related to the implementation of symbol tables](#some-issues-related-to-the-implementation-of-symbol-tables)
- [Reference](#reference)

<!-- /TOC -->

# Intermedia Representation

## My Take-aways

### Linear IRs vs. Graphical IRs

1. The structural organization of an IR ***has a strong impact on how the compiler writer thinks about analysis, optimization, and code generation***.
    - For example, treelike IRs lead naturally to passes structured as some form of tree-walk.
1. The compiler needs _**inexpensive ways**_ to perform the operations that it frequently does.
1. Broadly, usually, we have graphical IR, linear IR, and hybrid IR.

    1. The source code that serves as input to the compiler is a linear form.
    1. Linear IRs impose a clear and useful ordering on the sequence of operations, while graphical IRs expose parallelism and redundancy.
        1. Linear IRs represent the code being compiled as an ordered sequence of operations.
        1. Linear IRs can vary in their level of abstraction;
            - The source code for a program in a plain text file is a linear form, as is the assembly code for that same program.
    1. Linear IRs lend themselves to compact, human-readable representations.
1. AST is a typical graphical IR which has _**source-level abstraction**_.
    - AST is a syntax-oriented IR in which the edges show grammatical structure.
    - AST is used as a near-source-level representation, and **representation choices affect usability**.
1. CFG is another popular graphical IR. Some papers also call CFG CDFG (control data flow graph).
    - Every program can be represented as a CDFG intermediate form.
    - CFG can be built from the linear IR, but a PL's grammar will complicate the construction.

### About the IR design

1. The choice of a specific ir and a level of abstraction helps determine what operations the compiler can manipulate and optimize.
1. _**In practice, most of the improvement that compilers achieve in optimization arises from capitalizing on context. To make that improvement possible, the ir must expose the context**_.
    - Naming can hide context, as when it reuses one name for many distinct values.
    - It can also expose context, as when it creates a correspondencee between names and values.

## Why different IRs?

-   Many compiler writers consider trees and graphs as the natural representation for programs;
- Most processors that compilers target have ***linear assembly languages*** as their native language. Accordingly, some compilers use linear IRS with the rationale that those IRs expose properties of the target machine's code that the compiler should explicitly see.

Most compiler writers augment the IR with tables and sets that record additional information. We consider these tables as part of the IR.

## Categories of IRs

### Structural organization

Broadly speaking, IRs fall into three structural categories:

1.  _**graphical IRs**_
    - Encode the compiler's knowledge in a graph.
2.  _**linear IRs**_
    - Resemble pseudo-code for some abstract machine.
3.  _**hybrid IRs**_
    - Combine elements of both graphical and linear IRs.
    - A common hybrid representation uses a low-level linear IR to represent blocks of straight-line code and a graph to represent the flow of control among the blocks.

### The level of abstraction

The IR can range from a near-source representation to a low-level representation.

<p align="center">
<image src="images/IR-level-of-abstraction.png" width=50%><br>Fig 1. Level of abstracion.
</p>

The above figure shows array reference `A[i, j]` in near-source treelike IR and low-level linearized IR.

1. If the goal is to determine when two references can touch the same memory location. The source-level tree makes it easy to find and compare references, while low-level IR makes those tasks hard.
1. If the goal is to optimize the target-machine code generated for the array access, low-level IR allows the compiler to optimize details that remain implicit in the source-level tree.

### Naming discipline

1. in translating source code to a lower-level form, the compiler mush choose names from a variety of distinct values.
1. the choice of a naming scheme has a strong effect on how optimization can improve the code.
    - See the example of local value naming.
1. the choice of a naming scheme also has an impact on compile time, because it determines the sizes of many compile-time data structures.
    - As a practical matter, the costs of generating and manipulating an IR directly affect a compiler's speed.
1. the compiler writer should consider the expressiveness of the IR: its ability to accommodate all the facts that the compiler needs to record:
    1. the codes that define it
    1. _**the results of a static analysis**_
    1. _**profile data from previous executions**_
    1. maps to let the debugger understand the code and its data

## Graphical IRs

1. Graphical IRs encode relationships that may be difficult to represent in a linear IR.
1. A graphical IR can provide the compiler with an efficient way to move between logically connected points in the program, such as the definition of a variable and its use, or the source of a conditional branch and its target.

### 1. Syntax-related Trees (including parse tree and abstract syntax trees)

_**Main characteristic**_:

1. near-source representation.

_**Usage**_:

1. Parse trees are the primary IR for parsing.
1. Parse Tree/ASTs are broadly used by many compilers and interpreters since they have source-level abstractions.
1. ASTs are used by some automatic parallelization tools

In most treelike IRs, the structure of the tree corresponds to _**the syntax of the source code**_.

Parse tree vs. AST

- _**parse tree**_: is large relative to the source text with a node for each grammar symbol in the derivation.
    - it is worth considering ways to shrink this parse tree.
- _**abstract syntax trees**_: retains the essential structure of the parse tree but eliminates the extraneous nodes.
    - AST is _**more concise**_ than parsing tree.
    - AST is a near-source-level representation.
    - ASTs have been used in many practical compiler systems, including:
        - _**automatic parallelization**_ tools.
        - the S-expressions found in Lisp and Scheme implementations are essentially ASTs.

Example: AST for $a \times 2 + a \times 2 \times b$

<p align="center">
<image src="images/AST-example.png" width=30%><br> Fig 2. Example of AST.
</p>

### 2. Directly Acyclic Graphs

_**Main characteristics**_

DAG is a contraction of the AST that avoids some duplications.

_**Usage**_

1. used in systems that have constraints on the size of programs that the compiler can handle.
1. use DAG to expose redundancy.

<p align="center">
<image src="images/DAG-example.png" width=30%><br> Fig 3. Example of DAG.
</p>

1. DAG is more concise. The larger data structure is used, the more time compile-time analysis may use.
1. DAG encodes hints for parallelism.
1. DAG exposes redundancies.
    - In comparison with AST, _**indentical subtrees**_ are reused, so it encodes an explicit fact by sharing a copy of some variables.
        - if the value of $a$ cannot change between the two uses of $a$, then the compiler should generate code to evaluate $a \times 2$ once and use the results twice. This could reduce the evaluation costs.
        - however, the compiler must prove that $a$'s value cannot change.

### 3. CFG (basic block-level CFG/statement-level CFG)

_**Main characteristics**_

- Models the flow of control between the basic blocks in a program.
- Provides a graphical representation of the possible runtime control flow paths not grammatical structure.

_**Usage**_

1. Many parts of the compiler rely on a CFG explicitly or implicitly.
    1. _**instruction scheduling**_ needs a CFG to understand how the scheduled code for individual blocks flows together.
    1. _**global register allocation**_ relies on a CFG to understand how often each operation might execute and where to insert loads and stores for spilled values.
1. Analysis to support optimization generally begins with control-flow analysis and CFG construction.

The compiler typically uses a CFG in conjunction with another IR. The resulting combination is a hybrid IR.

1. the CFG represents the relationship among blocks.
1. operations inside a block are represented with another IR.

### 4. Dependency Graph

_**Main characteristics**_

1. The compiler also uses graphs to encode the flow of values from the point where a value is created, a _definition_, to any point where it is used, a _use_.
1. A dependency graph is often to be a derivative IR.

_Is this like the [use-define chain](https://en.wikipedia.org/wiki/Use-define_chain)?_

_**Usage**_

1. data-dependency graphs are often used as a _**derivative IR**_: constructed from the _**definitive IR**_, used, and then discarded.
1. play a central role in instruction scheduling.
1. used in a variety of optimizations, _**particularly transformations that reorder loops to expose parallelism**_, and to improve memory behavior;
1. The dependency graph does not fully capture the program's control flow.

<p align="center">
<image src="images/an-simple-example-of-dependency-graph.png" width=50%><br>Fig 5. An sample example of data-dependence graph.
</p>

- nodes in the data-dependence graph represent operations.
- an edge in a data-dependence graph connects two nodes, one that defines value and another that uses it.

_**The freedom in this partial order is precisely what an “out-of-order” processor exploits.**_

<p align="center">
<image src="images/dependence-graph.png"><br> Fig 4. An example of data-dependence graph.
</p>

The above figure has some more complexities:

1. connects all uses of $a$ together through a single node.
    - Typically requires sophisticated analysis of array subscripts to determine more precisely the patterns of access to arrays.
    - In more sophisticated applications of the data dependency graph, the compiler may perform an extensive analysis of array subscript values to determine when references to the same array can overlap.
1. nodes 5 and 6 both depend on themselves: they may have defined in the previous iteration.

### 5. Call Graph

_**Main characteristics**_

The call graph represents the runtime transfers of control between procedures. A call graph has a node for each procedure and an edge for each distinct procedure call site.

_**Usage**_

To address inefficiencies that arise across procedure boundaries, some compilers perform interprocedural analysis and optimization on a call graph.

## Linear IRs

1. The linear IRs used in compilers resemble the assembly code for an abstract machine.
1. If a linear ir is used as the definitive representation in a compiler, it must include a mechanism to encode transfers of control among points in the program.
    - Control flow demarcates the basic blocks in a linear ir.

### Categories of Linear IRs

1. _**one-address codes**_ model the behavior of accumulator machines and stack machines.
    - expose the machine’s use of implicit names so that the compiler can tailor the code for it.
    - the resulting code is quite _**compact**_.

    <p align="center">
    <image src="images/one-address-code.png"><br> Fig 5. Stack-Machine Code
    </p>

    1. assumes the presence of a stack of operands.
    1. most operations take their operands from the stack and push their results back onto the stack.
    1. the stack creates an implicit namespace and eliminates many names from the IR.

    - bytecodes generally implemented as a one-address code with implicit names on many operations and three-address code

1. _**two-address codes**_ model a machine that has destructive operations.

1. three-address codes
    - models the instruction format of a modern RISC machine.
    - generally implemented as a set of binary operations that have distinct name fields for two operands and one result.

    <p align="center">
    <image src="images/three-address-code.png"><br> Fig 6. Three-address code.
    </p>

## Mapping values to names

The compiler must invent the names of the intermediate results that the program produces when it executes. _**A name scheme can expose opportunities for optimizing, or it can obscure them**_.

For example, a source-level AST:

1. makes it easy to find all the references to an array `x`;
1. it also hides the details of the address calculations required to access an element of `x`;

### 1. Naming Temporary Values

Associating multiple expressions with a single temporary name obscured the flow of data and degraded the quality of optimization. The decline in code quality overshadowed any compile-time benefits.

1. The source code names tell the compiler little about the values that they hold.
1. When the compiler names each of these expressions, it can choose names in ways that specifically encode useful information about their values.
    - for example, a naming discipline reflects the computed values and ensures that textually identical expressions produce the same result.

    <p align="center">
    <image src="images/naming-leads-to-different-translations.png" width=50%>
    </p>

1. Different levels of abstraction. See below example:

    - In 5.8b, the low-level IR, each intermediate result has its name. Using distinct names exposes those results to analysis and transformation.

    <p align="center">
    <image src="images/different-levels-of-abstraction-for-an-array-subscript-reference.png" width=50%>
    </p>

### 2. Static Single-Assignment Form

- What is SSA form?

    1. _Static single-assignment form_(**SSA**) is a naming discipline that many modern compilers use to encode information about _**both the flow of control and the flow of data values**_ in the program.
    1. SSA has a _**value-based**_ name system created by renaming and the use of pseudo-operations called $\phi$-functions.
        _See an [old note for SSA](https://github.com/lcy-seso/LearningNotes/blob/master/paper_notes/programming-language/SSA/SSA.md)_.

- How to determine a program is in SSA form?
    1. each definition has a distinct name.
    1. each use refers to a single definition.

    To transfer an IR program to SSA form, the compiler inserts $\phi$-functions at points where different control-flow paths merge and it then renames variables to make the single-assignment property hold.

- SSA form was intended for code optimization.

### 3. Memory Model

1. Compiler's choice of a storage location for each value affects the information that can be represented in an IR version of a program.
1. In general, compilers work from one of the two memory models.
    1. _Register-to-register model_: the compiler keeps values in registers aggressively, ignoring any limitations imposed by the size of the machine's physical register set.
    1. _Memory-to-memory model_: the compiler assumes that all values are kept in memory locations.
1. The choice of memory model is mostly orthogonal to the choice of IR.

## Symbol Tables

### What is the symbol table?

1. As part of translation, a compiler derives information about the various entities manipulated by the program being translated.
    - compiler encounters a wide variety of names:
        1. variables,
        1. defined constants,
        1. procedures,
        1. functions,
        1. labels,
        1. structures,
        1. files.
    - compiler also generate many names.
1. The compiler must either record this information in the IR or re-derive it on demand.
1. For the sake of efficiency, most compilers record facts rather than recompute them.

These facts can be recorded directly in the IR.

1. pros: provides a uniform access method and a single implementation.
1. cons: the single access method may be inefficient.
    - for example, navigating the AST to find the approapriate declaration has its own costs.

The alternative, is to _**create a central repository for these facts and provide efficient access to it**_, called a symbol table.

_**Symbol table is an integral part of the compiler's IR**_.

### Benefits of introducing symbol tables

In some sense, the symbol table is simply an efﬁciency trick, it:

1. localizes information derived from potentially distant parts of the source code.
1. simpliﬁes the design and implementation of any code that must refer to information about variables derived earlier in compilation.
1. avoids the expense of searching the IR to find the portion that represents a variable’s declaration; using a symbol table often eliminates the need to represent the declarations directly in the ir.
1. eliminates the overhead of making each reference contain a pointer to the declaration.
1. replaces both of these with a computed mapping from the textual name to the stored information.

### Some issues related to the implementation of symbol tables

TLDR;

1. choose a data structure: hash tables
1. build a symbol table
1. handle nested scopes

# Reference

1. [Engineering A Compiler, Chapter 5, Intermedia Representation](https://github.com/concerttttt/books/blob/master/Engineering%20A%20Compiler%202nd%20Edition%20by%20Cooper%20and%20Torczon.pdf)
