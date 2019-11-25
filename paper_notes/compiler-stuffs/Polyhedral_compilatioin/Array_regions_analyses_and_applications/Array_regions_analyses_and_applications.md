# Array regions analyses and applications


1. Compiling sequential programs for parallel homogeneous or heterogeneous machines, with either shared or distributed memory, <span style="background-color:#ACD6FF;">_**requires a knowledge about memory accesses and data flows to support parallelization and/or generation of communications**_</span>. Scientific computation-intensive applications have long been choice candidates for studying this field, <span style="background-color:#ACD6FF;">_**in particular because of their usage of the simple data structures**_</span> provided by the Fortran language, namely _**arrays**_.

1. Many optimization approaches, in particular in the polyhedral realm, are _**based on a fine study of relationships between single memory scalar or array references**_. They aim at supporting optimal program transformations, but they enforce restrictions on the input code, notably _**the absence of function calls**_. _**On the contrary, array element sets analyses rely on summarization and approximation techniques to collect information on a coarser grain scale, thus allowing function calls or less structured code, at the expense of precision**_. However, these techniques have proven their efficiency on real life applications, in particular array region analyses, which are based on convex polyhedra. They are implemented in the PIPS parallelizing compiler, and extensively used in Par4All, a source-to-source platform targeting various architectural models.

1. The define/use analysis on a whole array is insufficient for aggressive compiler optimizations such as auto parallelization and array privatization.

1. In the absence of optimization, the high-level language constructs that improve productivity can result in order-of-magnitude runtime performance degradations.

1. high-level languages and their programming models will need to provide abstractions that deliver a sufficient fraction of available performance on the underlying architectures without exposing too many low-level details.

# Reference

1. [slides](http://labexcompilation.ens-lyon.fr/wp-content/uploads/2013/06/Beatrice.pdf)
1. [Vedio](http://labexcompilation.ens-lyon.fr/polyhedral-school/videos/#beatrice)
