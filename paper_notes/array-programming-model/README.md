<!-- vscode-markdown-toc -->
- [Array type for code generation](#array-type-for-code-generation)
- [High-level IR](#high-level-ir)
  - [the LIFT IR](#the-lift-ir)
- [Data parallel language](#data-parallel-language)
- [Other references](#other-references)

<!-- vscode-markdown-toc-config
	numbering=true
	autoSave=true
	/vscode-markdown-toc-config -->
<!-- /vscode-markdown-toc -->

#  Array type for code generation

1. Pizzuti, Federico, Michel Steuwer, and Christophe Dubach. "[Position-dependent arrays and their application for high performance code generation](https://www.pure.ed.ac.uk/ws/files/99609695/Position_Dependent_Arrays_PIZZUTI_DoA220619_AFV.pdf)." Proceedings of the 8th ACM SIGPLAN International Workshop on Functional High-Performance and Numerical Computing. 2019.

# High-level IR

1. Roesch, Jared, et al. "[Relay: A new ir for machine learning frameworks](https://arxiv.org/pdf/1810.00952.pdf)." Proceedings of the 2nd ACM SIGPLAN International Workshop on Machine Learning and Programming Languages. 2018.

1. Kroll, Lars, et al. "[Arc: an ir for batch and stream programming](https://dl.acm.org/doi/pdf/10.1145/3315507.3330199)." Proceedings of the 17th ACM SIGPLAN International Symposium on Database Programming Languages. 2019.

## the LIFT IR

1. Steuwer, Michel, Toomas Remmelg, and Christophe Dubach. "[Lift: a functional data-parallel IR for high-performance GPU code generation](https://eprints.gla.ac.uk/146596/1/146596.pdf)." 2017 IEEE/ACM International Symposium on Code Generation and Optimization (CGO). IEEE, 2017.

1. Kristien, Martin, et al. "[High-level synthesis of functional patterns with Lift](https://www.pure.ed.ac.uk/ws/portalfiles/portal/99607957/High_Level_Synthesis_KRISTIEN_DoA071218_AFV.pdf)." Proceedings of the 6th ACM SIGPLAN International Workshop on Libraries, Languages and Compilers for Array Programming. 2019.

1. Myia
   
    https://github.com/mila-iqia/myia/blob/master/myia/ir/anf.py

    ```
    Myia's main intermediate representation (IR) is a graph-based version of ANF.
    Each function definition (lambda) is defined as a graph, consisting of a series
    of function applications.

    A function can be applied to a node from another function's graph; this
    implicitly creates a nested function. Functions are first-class objects, so
    returning a nested function creates a closure.
    ```

# Data parallel language

1. Blelloch, Guy E. [NESL: A nested data-parallel language](http://www.cs.cmu.edu/~guyb/papers/Nesl3.1.pdf).(version 3.1). CARNEGIE-MELLON UNIV PITTSBURGH PA SCHOOL OF COMPUTER SCIENCE, 1995.

1. [Nessie: A NESL to CUDA Compiler](https://shonan.nii.ac.jp/archives/seminar/136/wp-content/uploads/sites/172/2018/09/nessie-talk.pdf)

1. Larus, James. "[C**: A large-grain, object-oriented, data-parallel programming language](https://minds.wisconsin.edu/bitstream/handle/1793/59682/TR1126.pdf?sequence=1)." International Workshop on Languages and Compilers for Parallel Computing. Springer, Berlin, Heidelberg, 1992.


# Other references

- **PyTorch JIT**g
    - supported grammar: https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/frontend/tree_views.h#L15
    - types: https://github.com/pytorch/pytorch/blob/master/torch/csrc/jit/frontend/tree_views.h#L20-L22
