<!-- vscode-markdown-toc -->
- [Basic Concepts and Conventional notations](#basic-concepts-and-conventional-notations)
  - [Type constructor $\rightarrow$](#type-constructor-rightarrow)
  - [Terms and Types](#terms-and-types)
  - [The $\lambda$ notation](#the-lambda-notation)
    - [Examples of untyped $\lambda$-term](#examples-of-untyped-lambda-term)
    - [BNF for $\lambda$-term](#bnf-for-lambda-term)
    - [Currying](#currying)
  - [Kinding](#kinding)
  - [Type judgement, its parsing and reading](#type-judgement-its-parsing-and-reading)
    - [Exercise](#exercise)
- [Reference](#reference)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

# Basic Concepts and Conventional notations

## Type constructor $\rightarrow$

the notation $\rightarrow$ is called the type constructor. By convention, is associates to the right.

$\texttt{int}\rightarrow \texttt{int}\rightarrow \texttt{int}$ is the same as $\texttt{int}\rightarrow (\texttt{int}\rightarrow \texttt{int})$.

## Terms and Types

Typically, programming languages have two different kinds of expressions: *terms* and *types*.

- A term is an expression representing a value.
- A type is an expression representing a class of similar values.

A value of a term is determined at runtime by evaluating the term. Its value may not be konwn at compile time.

Types can be determined at compile time and is used by the compiler to rule out ill-formed terms.

## The $\lambda$ notation

the pure λ-calculus has only λ-terms and only the operators of functional abstraction and functional application, nothing else.

A $\lambda$-term is defined inductively as follows. Let $Var$  be a countably infinite set of variables $x$, $y$, ...

- Any variable $x \in Var$ is a $\lambda$-term
- If $e$ is a $\lambda$-term, then so is $\lambda x.e$ (function abstraction, **$\lambda$ x is called the abstraction operator**).
    - $\lambda x.e$: a function with input parameter $x$ and body $e$.
- If $e_1$ and $e_2$ are $\lambda$-terms, then so is $e_1\cdot e_2$ (function application).
    - $e_1 \cdot e_2$: apply $e_1$ as a functon to $e_2$ as its input.
    - $\cdot$ (cdot) is the application operator, and usually can be omit: $e_1 e_2$.

### Examples of untyped $\lambda$-term

"untyped" means that there is no restrictions on how $\lambda$-term can be combined to form other $\lambda$-terms.

Every well-formed $\lambda$-term is meaningful.

1. $\textit{id} = \lambda x.x$: a function return its argument. Then the function is an identity function.
1. $\lambda x.\lambda a.a$: a function ignore its argument and returns the indentity function.
    - a function returns a function.
1. $\lambda f.fa$: a function that takes another function $f$ as its argument, and applies it to $a$.
    - a function recieves a function.

1. $\lambda v.\lambda f.fv$: a function takes an argument $v$ and returns a function $\lambda f.fv$ that calls its argument -- some function $f$ -- on $v$.

1. $\lambda f.\lambda g.\lambda x.g(f(x))$: a function take a pair of functions: $f$ and $g$ as arguments and returns its compositions $g \circ f$.
    - the composition operator could be defined this way.

### BNF for $\lambda$-term

$$e::= x | e_1 e_2 | \lambda x.e$$

The pure untyped $\lambda$-caculus has only two syntactic class, variables and $\lambda$-terms.

### Currying

We would like to allow multiple arguments to a function, we can write:

$$\lambda x_1.\lambda x_2...\lambda x_n.e$$

An abbreviation is (the right hand side is the desugaring transformation):

$$
\begin{align*}
\lambda x_1,...,x_n.e &\Rightarrow \lambda x_1 . \lambda x_n.e\\
e_0(e_1,...,e_n) &\Rightarrow e_0e_1e_2...e_n
\end{align*}
$$

This particular of of sugar is called currying.

The scope of the abstraction operator λx shown in the term λx. e is its body e. An occurrence of a variable
y in a term is said to be bound in that term if it occurs in the scope of an abstraction operator λy (with the
same variable y); otherwise, it is free. A bound occurrence of y is bound to the abstraction operator λy with
the smallest scope in which it occurs.

## Kinding

## Type judgement, its parsing and reading

- $\Gamma \vdash _{\Sigma} e:\tau$ can be viewed as a proposition that is read as "in the **context** $\Gamma$ (or under assumption $\Gamma$. The assumption $\Gamma$ specify types for the free variables of $e$), the term $e$ has type $\tau$ given **signature** $\Sigma$"
- $\vdash _{\Sigma} e:\tau$: $e$ is a closed term and $\Gamma$ is empty. like: $\vdash 3:\texttt{int}$

The typing rules will determine which terms are well-formed typed $\lambda$-calculus programs. $\Gamma$ is a type environment, a partial map from variables to types used to determine the types of the free variables in $e$. The domain of $\Gamma$ as a partial
function $Var \rightharpoonup Type$ is denoted $\texttt{dom} \Gamma$.

- $x:\tau \in \Gamma$ just means $\Gamma(x)=\tau$
- $\Gamma,x:\tau$ means binding $x$ to $\tau$

### Exercise

$$\frac{\Gamma,x:\sigma \vdash e:\tau}{\Gamma \vdash (\lambda x:\sigma .e):\sigma\rightarrow\tau}$$

In the above type judgement (type rule for $\lambda$-abstraction):
- The $\lambda$-abstraction $\lambda x:\sigma.e$ is supposed to represent a function.

---

well-formed types

function type

function type abstracting

# Reference
1. https://www.cs.cornell.edu/courses/cs6110/2019sp/lectures/lec01.pdf
