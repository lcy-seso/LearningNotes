<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Reading List](#reading-list)
  - [Loop Transformations in non-polyhedral framework](#loop-transformations-in-non-polyhedral-framework)
  - [Dependence representation](#dependence-representation)
  - [Polyhedral compilation](#polyhedral-compilation)
    - [Miscellanea](#miscellanea)
    - [Thesis](#thesis)
    - [Dependence analysis/test in polyhedral framework](#dependence-analysistest-in-polyhedral-framework)
    - [Polyhedral and precise data-flow analysis](#polyhedral-and-precise-data-flow-analysis)
    - [Translate dataflow program into integer set representation](#translate-dataflow-program-into-integer-set-representation)
    - [Fully automatic polyhedral compilers](#fully-automatic-polyhedral-compilers)
  - [IRs for DSL compiler](#irs-for-dsl-compiler)
  - [Skewing/Wavefront Parallelism](#skewingwavefront-parallelism)
  - [Algebraic view of code transformation](#algebraic-view-of-code-transformation)
  - [Affine loop transformation for locality](#affine-loop-transformation-for-locality)

<!-- /TOC -->
# Reading List

## Loop Transformations in non-polyhedral framework

- [ ] [Loop transformations](https://www.cri.ensmp.fr/~tadonki/PaperForWeb/tadonki_loop.pdf)
- [ ] Kelly W, Pugh W. [A unifying framework for iteration reordering transformations](https://drum.lib.umd.edu/bitstream/handle/1903/708/CS-TR-3430.pdf?sequence=4&isAllowed=y)[C]//Proceedings 1st International Conference on Algorithms and Architectures for Parallel Processing. IEEE, 1995, 1: 153-162.

    > this work first presented that striping-mining is able to be modeled as a schedule only transformation.

- [ ] Girbal S, Vasilache N, Bastoul C, et al. [Semi-automatic composition of loop transformations for deep parallelism and memory hierarchies]()[J]. International Journal of Parallel Programming, 2006, 34(3): 261-317.

  > this work show us _**some most impactful loop nest transformations**_ (<span style="background-color:#ACD6FF;">_what kind of transformation??_</span>) cannot be expressed as _**structural, incremental updates of the loop tree structure**_ (<span style="background-color:#ACD6FF;">_what does this mean??_</span>).

## Dependence representation

- [ ] [Computing dependence direction vectors and dependence cones with linear systems](https://www.cri.ensmp.fr/classement/doc/E-094.pdf)

## Polyhedral compilation

### Miscellanea

- [Polyhedral school](http://labexcompilation.ens-lyon.fr/polyhedral-school/program/)
- A lecture for polyhedral compilation https://www.cs.colostate.edu/~pouchet/
- [Polyhedral Compilation as a Design Pattern for Compilers](https://www.youtube.com/watch?v=mt6pIpt5Wk0)
    - [slides](https://pliss2019.github.io/albert_cohen_slides.pdf)
- [Cédric Bastoul's website](http://icps.u-strasbg.fr/people/bastoul/public_html/)
- [Theoretical background for Polyhedral compilation](http://www.irisa.fr/polylib/DOC/index.html).

### Thesis

- [ ] Bastoul. [Improving Data Locality in Static Control Programs](http://icps.u-strasbg.fr/people/bastoul/public_html/research/papers/Bastoul_thesis.pdf). PhD thesis, University Paris 6, Pierre et Marie Curie, France, Dec. 2004.
- [ ] Lim A W. [Improving parallelism and data locality with affine partitioning](https://suif.stanford.edu/papers/lim-thesis.ps.gz)[J]. 2002
- [ ] Grosser T. [A decoupled approach to high-level loop optimization: tile shapes, polyhedral building blocks and low-level compilers](https://tel.archives-ouvertes.fr/tel-01144563/document)[D]. , 2014.

### Dependence analysis/test in polyhedral framework

- [ ] [GCD Test](https://apps.dtic.mil/dtic/tr/fulltext/u2/a268069.pdf)
- [x] Pugh W. [The Omega test: a fast and practical integer programming algorithm for dependence analysis](http://www.cs.cmu.edu/~emc/spring06/home1_files/p4-pugh.pdf)[C]//Supercomputing'91: Proceedings of the 1991 ACM/IEEE conference on Supercomputing. IEEE, 1991: 4-13.

    > `isl` is inspried by this work.

- [ ] Palkovic M. [Enhanced applicability of loop transformations](https://pdfs.semanticscholar.org/efec/40ee2d3c61f7285912aa5611d5691b953dd5.pdf)[J]. Dissertation Abstracts International, 2007, 68(04).
- [ ] Maximizing Loop Parallelism and Improving Data Locality via Loop Fusion and Distribution?
- [ ] Wolf M E, Lam M S. [A loop transformation theory and an algorithm to maximize parallelism](https://www.cs.indiana.edu/~achauhan/Teaching/B629/2006-Fall/CourseMaterial/1991-tpds-wolf-unimodular.pdf)[J]. IEEE Transactions on Parallel & Distributed Systems, 1991 (4): 452-471.
- [ ] Loop parallelization algorithms: From parallelism extraction to code generation
- [ ] Feautrier P. [Array expansion](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.29.5704&rep=rep1&type=pdf)[C]//ACM International Conference on Supercomputing 25th Anniversary Volume. ACM, 2014: 99-111.
- [ ] Liu D, Shao Z, Wang M, et al. [Optimal loop parallelization for maximizing iteration-level parallelism](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.149.7760&rep=rep1&type=pdf)[C]//Proceedings of the 2009 international conference on Compilers, architecture, and synthesis for embedded systems. ACM, 2009: 67-76.
- [ ]  Sinharoy B, Szymanski B K. [Finding optimum wavefront of parallel computation](https://www.cs.rpi.edu/~szymansk/papers/jpaa.92.pdf)[J]. Parallel algorithms and applications, 1994, 2(1-2): 5-26.

### Polyhedral and precise data-flow analysis

- [ ] Maslov V. [Lazy array data-flow dependence analysis](https://drum.lib.umd.edu/bitstream/handle/1903/590/CS-TR-3110.pdf?sequence=4&isAllowed=y)[C]//Proceedings of the 21st ACM SIGPLAN-SIGACT symposium on Principles of programming languages. ACM, 1994: 311-325.
- [ ] Chafi H, Sujeeth A K, Brown K J, et al. [A domain-specific approach to heterogeneous parallelism](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.644.4729&rep=rep1&type=pdf)[J]. ACM SIGPLAN Notices, 2011, 46(8): 35-46.
- [ ] Libby J C, Kent K B. [A survey of data dependence analysis techniques for automated parallelization](http://www.cs.unb.ca/tech-reports/documents/TR07-188_000.pdf)[J]. TR07-188, 2007: 1-34.

### Translate dataflow program into integer set representation

- [ ] Bhaskaracharya S G, Bondhugula U. [Polyglot: a polyhedral loop transformation framework for a graphical dataflow language](https://link.springer.com/content/pdf/10.1007/978-3-642-37051-9_7.pdf)[C]//International Conference on Compiler Construction. Springer, Berlin, Heidelberg, 2013: 123-143.
- [x] Benabderrahmane M W, Pouchet L N, Cohen A, et al. [The polyhedral model is more widely applicable than you think](https://link.springer.com/content/pdf/10.1007/978-3-642-11970-5_16.pdf)[C]//International Conference on Compiler Construction. Springer, Berlin, Heidelberg, 2010: 283-303.

    > This work expands the application domain of the polyhedral model. It presents slight extensions to the polyhedral representation itself, based on the notions of _**exit**_ and _**control predicates**_ that allow to consider _**general while loops**_ and _**if conditions**_

    > Mainstream deep learning frameworks represented by TensorFlow characterize deep learning computation as a dataflow program. Whether this work also could inspire us to make an incremental extension to deep learning frameworks?

- [ ] Feautrier P. [Dataflow analysis of array and scalar references](https://www.researchgate.net/publication/2425315_Dataflow_Analysis_of_Array_and_Scalar_References)[J]. International Journal of Parallel Programming, 1991, 20(1): 23-53.
- [ ] Maydan D E. [Accurate analysis of array references](https://apps.dtic.mil/dtic/tr/fulltext/u2/a268069.pdf)[R]. STANFORD UNIV CA DEPT OF COMPUTER SCIENCE, 1992.
    > This work propose some methods to make the ordering constraints explicit for the compiler.

### Fully automatic polyhedral compilers

- General polyhedral compilers
  - [ ] _**PENCIL**_, Baghdadi R, Cohen A, Grosser T, et al. [PENCIL language specification](https://hal.inria.fr/hal-01154812/document)[J]. 2015.
  - [ ] _**PENCIL**_, Baghdadi R, Beaugnon U, Cohen A, et al. [Pencil: A platform-neutral compute intermediate language for accelerator programming](https://lirias.kuleuven.be/retrieve/342210)[C]//2015 International Conference on Parallel Architecture and Compilation (PACT). IEEE, 2015: 138-149.
  - [ ] _**Pluto**_, Bondhugula U, Hartono A, Ramanujam J, et al. [A practical automatic polyhedral parallelizer and locality optimizer](https://s3.amazonaws.com/academia.edu.documents/32503610/pldi08.pdf?response-content-disposition=inline%3B%20filename%3DA_practical_automatic_polyhedral_paralle.pdf&X-Amz-Algorithm=AWS4-HMAC-SHA256&X-Amz-Credential=AKIAIWOWYYGZ2Y53UL3A%2F20191227%2Fus-east-1%2Fs3%2Faws4_request&X-Amz-Date=20191227T040240Z&X-Amz-Expires=3600&X-Amz-SignedHeaders=host&X-Amz-Signature=57e69856de3e15a7bb196a5cadc90f7746f1be7fedce07e41ee5dd8558f636c4)[C]//Acm Sigplan Notices. ACM, 2008, 43(6): 101-113.
  - [ ] _**Polly**_, Grosser T, Groesslinger A, Lengauer C. [Polly—performing polyhedral optimizations on a low-level intermediate representation](https://grosser.es/publications/grosser-2012-Polly-Performing-polyhedral-optimizations-on-a-low-level-intermediate-representation.pdf)[J]. Parallel Processing Letters, 2012, 22(04): 1250010.

- PolyMega and TC are designed for specific domain.
    - [ ] _**PolyMage**_, Mullapudi R T, Vasista V, Bondhugula U. Polymage: Automatic optimization for image processing pipelines[C]//ACM SIGPLAN Notices. ACM, 2015, 50(4): 429-443.
    - [ ] _**TC**_, Vasilache N, Zinenko O, Theodoridis T, et al. [Tensor comprehensions: Framework-agnostic high-performance machine learning abstractions](https://arxiv.org/pdf/1802.04730.pdf)[J]. arXiv preprint arXiv:1802.04730, 2018.
- [x] [Tiramisu: A Polyhedral Compiler for Expressing Fast and Portable Code](https://arxiv.org/abs/1804.10694)

## IRs for DSL compiler

1. Chafi H, Sujeeth A K, Brown K J, et al. [A domain-specific approach to heterogeneous parallelism](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.644.4729&rep=rep1&type=pdf)[J]. ACM SIGPLAN Notices, 2011, 46(8): 35-46.

## Skewing/Wavefront Parallelism

- [ ] Belviranli M E, Deng P, Bhuyan L N, et al. [Peerwave: Exploiting wavefront parallelism on gpus with peer-sm synchronization](http://mehmet.belviranli.com/papers/ics15.pdf)[C]//Proceedings of the 29th ACM on International Conference on Supercomputing. ACM, 2015: 25-35.
- [ ] Loop skewing: the wavefront method revisited
- [ ] Tang P. [Chain-based scheduling: Part I-loop transformations and code generation](https://openresearch-repository.anu.edu.au/bitstream/1885/40801/3/TR-CS-92-09.pdf)[J]. 1992.
- [ ] Lamport L. [The parallel execution of do loops](https://www.cs.colostate.edu/~cs560dl/Notes/LamportCACM1974.pdf)[J]. Communications of the ACM, 1974, 17(2): 83-93.

## Algebraic view of code transformation

- [ ] [A Linear Algebraic View of Loop Transformations and Their Interaction](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.56.5153&rep=rep1&type=pdf)
- [ ] Dowling M L. Optimal code parallelization using unimodular transformations[J]. Parallel computing, 1990, 16(2-3): 157-171.
- [ ] Whitfield D, Soffa M L. [An approach to ordering optimizing transformations](https://dl.acm.org/citation.cfm?id=99179)[C]//ACM SIGPLAN Notices. ACM, 1990, 25(3): 137-146.
- [ ] Wolfe M. Massive parallelism through program restructuring[C]//[1990 Proceedings] The Third Symposium on the Frontiers of Massively Parallel Computation. IEEE, 1990: 407-415.
- [ ] Allen J R. [Dependence analysis for subscripted variables and its application to program transformations](https://scholarship.rice.edu/bitstream/handle/1911/19045/8314916.PDF?sequence=1&isAllowed=y)[D]. Rice University, 1983.
- [ ] Coarse-grained loop parallelization: Iteration Space Slicing vs affine transformations
- [ ] Optimizing supercompilers for supercomputers.

## Affine loop transformation for locality

- [ ] Bastoul C, Feautrier P. [More legal transformations for locality](https://hal.inria.fr/inria-00001056/document)[C]//European Conference on Parallel Processing. Springer, Berlin, Heidelberg, 2004: 272-283.
- [ ] Bondhugula U, Hartono A, Ramanujam J, et al. [A practical automatic polyhedral parallelizer and locality optimizer](http://www.ece.lsu.edu/jxr/Publications-pdf/pldi08.pdf)[C]//Proceedings of the 29th ACM SIGPLAN Conference on Programming Language Design and Implementation. 2008: 101-113.
