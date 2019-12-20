
## Contributions of this work

Two mehtods proposed in this paper translate a nest of _DO_ loops into a form explicitly indicating the parallel execution.

1. hyperplane method
    - applicable to multiple instruction stream computers and single instruction stream computers
    - Use Loop (1) below as the example. Hyperplane method will find that the body of loop nesting can be executed concurrently for all points $(I, J, K)$ lying in the plance defined by $2I + J + K = \text{constant}$. The constant is incremented after each execution until the loop body has been executed for all points in the iteration space.

2. coordinate method
    - applicable to single instruction stream computers

Major restrictions are that the loop body contain no I/O and no transfer of control to any statement outside the loop.

## Assumptions about the loop body

A variable which appears on the left-hand side of an assignment statement in the loop body is called a _generated_ variable.

1. It contains no I/O statements.
1. It contains no transfer of control to any statement outside the loop.
1. It contains no subroutine or function call which can modify data.
1. Any occurrence in the loop body of a generated variable $VAR$ is of the form $VAR(e^1, ..., e^{\tau})$, where each $e^i$ is an expression not containing any generated variable.

## Some key points in this paper

1. recognizing parallel processable streams in computer programs is at best a formidable task. It is impossible if $L$, $M$ and $N$ are not all known at compile time.

    Loop (1):
    ```python
    DO 99 I = 1, L
        DO 99 J = 2, M
            DO K = 2, N
                U(J,K) = (U(J+1, K) + U(J,K+1) + U(J-1,K) + U(J,K-1)) * .25
    ```

# References

1. Lamport L. [The parallel execution of do loops](https://www.cs.colostate.edu/~cs560dl/Notes/LamportCACM1974.pdf)[J]. Communications of the ACM, 1974, 17(2): 83-93.
1. Marlow S. [Parallel and concurrent programming in Haskell](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.468.1136&rep=rep1&type=pdf)[C]//Central European Functional Programming School. Springer, Berlin, Heidelberg, 2011: 339-401. Page 5: terminology: parallelism and concurrency.
    - A parallel program is one that uses a multiplicity of computational hardware (e.g. multiple processor cores) in order to perform computation more quickly.
    - Concurrency is a program-structuring technique in which there are multiple threads of control. Notionally the threads of control execute “at the same time”; that is, the user sees their eﬀects interleaved.
    - While parallel programming is concerned only with eﬃciency, concurrent programming is concerned with structuring a program that needs to interact with multiple independent external agents (for example the user, a database server, and some external clients).
    - Concurrency allows such programs to be modular; the thread that interacts with the user is distinct from the thread that talks to the database. In the absence of concurrency, such programs have to be written with event loops and callbacks—indeed, event loops and callbacks are often used even when concurrency is available, because in many languages concurrency is either too expensive, or too diﬃcult, to use.
