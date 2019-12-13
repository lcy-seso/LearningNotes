# Basic concepts

## Integer set

In general an integer set has the form:

$$\mathbf{S} = \{ \mathbf{\vec{s}} \in \mathcal{Z}^d | \mathbf{f}(\mathbf{\vec{s}}, \mathbf{\vec{p}})\}$$

- $\mathbf{\vec{s}}$: integer tuples contained in the integer set;
- $d$ the dimensionality;
- $\mathbf{\vec{p}}\in \mathcal{Z}^e$: a vector of $e$ parameters;
- $\mathbf{f}(\mathbf{\vec{s},\mathbf{\vec{p}}})$: Presburger formula;

## Presburger formula

A Presburger formula $p$ is defined _**recursively**_ as either:

1. a _**boolean constant**_ ($\top$, $\bot$)
1. the result of [a boolean operation](https://en.wikipedia.org/wiki/Boolean_algebra) such as negation, conjunction, disjunction or implication ($\lnot p$, $p_1 \wedge p_2$, $p_1 \vee p_2$, $p_1 \Rightarrow p_2$ )
1. a _**[quantified expression](http://www.ada-auth.org/standards/12rat/html/Rat12-3-4.html)**_ ($\forall x : p$, $\exists x : p$)

    > _A quantified expression is very much like a `for` statement except that we evaluate the expression after $\Rightarrow$ on each iteration rather than executing one or more statements. The iteration is somewhat implicit and the words loop and end loop do not appear._

1. a _**comparison**_ between different quasi-affine expressions ($e_1 \oplus e_2, \oplus \in \{\lt, \le, \ge, \gt\}$)

## Quasi-affine expression

$e$ is defined as (1) a plain integer constant (e.g., 10); (2) a parameter; (3) a set dimension or a previously introduced _**quantified variable**_;

It can also be _**constructed recursively**_ as:

1. the result of a unary negation of a quasi-affine expression ($−e$);
1. a multiplication of an integer constant with a quasi-affine expression (e.g., $10$);
1. an addition/subtraction of two quasi-affine expressions ($e_1 \oplus e_2$ , $\oplus \in \{ +, − \}$);
1. an integer division of a quasi-affine expression by a constant (e.g., $\lfloor e/10 ⌋$);
1. the result of computing a quasi-affine expression modulo;

**<span style="background-color:#ACD6FF;">A set can be approximated by computing various hulls</span>.**

# Reference

1. [What are Presburger formulas?](http://www.cs.umd.edu/projects/omega/interface_doc/node6.html)
1. [Affine hull](https://en.wikipedia.org/wiki/Affine_hull)
1. [Linear, affine, and convex sets and hulls](http://users.mat.unimi.it/users/libor/AnConvessa/insiemi_e_involucri.pdf)
1. [Affine transformations](https://eli.thegreenplace.net/2018/affine-transformations/)
