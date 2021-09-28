# A loop transformation theory and an algorithm to maximize parallelism

## Key points

1. DOALL loop is a loop that there are no dependences carried by loop iterations.
1. To maximize the degree of parallelism is to transform the loop nest to maximize the number of DOALL loops.
1. The optimality for parallelism.
    - $n$-deep loops whose dependences can be represented with distance vectors, have at least $n-1$ degresss of parallelism[[1](#references)].
    - the wave-front transformation produces _**the maximal degree of parallelism**_, but makes the outermost loop sequential if any are.
1. When the dependence vectors do not span the entire iteration space, it is possible to perform a transformation that makes outermost DOALL loops[[2](#references)].
1. If the loop nest depth is $n$ and the dimensionality of the space spanned by the dependences is $n_d$, it is possible to make the first $n-n_d$ rows of a transformation matrix $T$ span the _**orthogonal subspace $S$**_ of the space spanned by the dependences. This will produce the maximal number of outermost DOALL loops within the nest.
    - A vector $\vec{s}$ is in $S$ if and only if $\forall \vec{d}: \vec{s} \cdot \vec{d} = 0$. $S$ is the nullspace (kernel) of the matrix that has the dependences vecotrs as rows.

1. The code transformation step to maximum parallelism
    - step 1: transform the code into the canonical form of a fully permutable loop nest.
    - step 2: tile the loops.
    - step 3: apply the wavefront transformation to to controlling loops of the tiles.
      - tiling reduce synchronization and improve locality.

2. fine-grained corase-grained granularity parallelism.

# References

1. Irigoin F, Triolet R. [Supernode partitioning](http://www.cri.ensmp.fr/classement/doc/E-090.pdf)[C]//Proceedings of the 15th ACM SIGPLAN-SIGACT symposium on Principles of programming languages. ACM, 1988: 319-329.
