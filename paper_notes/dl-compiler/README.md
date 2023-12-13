# Reading List

## Hardware-aware DNN model optimization

1. Dao, Tri, et al. "[Flashattention: Fast and memory-efficient exact attention with io-awareness](https://arxiv.org/pdf/2205.14135.pdf)." Advances in Neural Information Processing Systems 35 (2022): 16344-16359.
1. Ivanov, Andrei, et al. "[Data movement is all you need: A case study on optimizing transformers](https://proceedings.mlsys.org/paper_files/paper/2021/file/bc86e95606a6392f51f95a8de106728d-Paper.pdf)." Proceedings of Machine Learning and Systems 3 (2021): 711-732.

## DL compiler

1. Zheng, Zhen, et al. "[AStitch: enabling a new multi-dimensional optimization space for memory-intensive ML training and inference on modern SIMT architectures](https://jamesthez.github.io/files/astitch-asplos22.pdf)." Proceedings of the 27th ACM International Conference on Architectural Support for Programming Languages and Operating Systems. 2022.
1. Zhao, Jie, et al. "[Effectively Scheduling Computational Graphs of Deep Neural Networks toward Their Domain-Specific Accelerators](https://www.usenix.org/system/files/osdi23-zhao.pdf)." 17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23). 2023.
1. Shi, Yining, et al. "[Welder: Scheduling Deep Learning Memory Access via Tile-graph](https://www.usenix.org/system/files/osdi23-shi.pdf)." 17th USENIX Symposium on Operating Systems Design and Implementation (OSDI 23). 2023.
1. Li, Yijin, et al. "[SIRIUS: Harvesting Whole-Program Optimization Opportunities for DNNs](https://proceedings.mlsys.org/paper_files/paper/2023/file/875578931a159790107a9184e39a67a4-Paper-mlsys2023.pdf)." Proceedings of Machine Learning and Systems 5 (2023).
1. Niu, Wei, et al. "[DNNFusion: accelerating deep neural networks execution with advanced operator fusion](https://dl.acm.org/doi/pdf/10.1145/3453483.3454083)." Proceedings of the 42nd ACM SIGPLAN International Conference on Programming Language Design and Implementation. 2021.
1. Smith, Gus Henry, et al. "[Pure tensor program rewriting via access patterns (representation pearl)](https://arxiv.org/pdf/2105.09377.pdf)." Proceedings of the 5th ACM SIGPLAN International Symposium on Machine Programming. 2021.

## General loop program analysis

1. Jang, Byunghyun, et al. "[Exploiting memory access patterns to improve memory performance in data-parallel architectures](https://www.researchgate.net/profile/David-Kaeli/publication/224141979_Exploiting_Memory_Access_Patterns_to_Improve_Memory_Performance_in_Data-Parallel_Architectures/links/0deec5226219d43067000000/Exploiting-Memory-Access-Patterns-to-Improve-Memory-Performance-in-Data-Parallel-Architectures.pdf)." IEEE Transactions on Parallel and Distributed Systems 22.1 (2010): 105-118.
1. Wolf, Michael E., and Monica S. Lam. "[A loop transformation theory and an algorithm to maximize parallelism](https://homes.luddy.indiana.edu/achauhan/Teaching/B629/2006-Fall/CourseMaterial/1991-tpds-wolf-unimodular.pdf)." IEEE Transactions on Parallel & Distributed Systems 2.04 (1991): 452-471.
1. Lamport, Leslie. "The parallel execution of DO loops." Communications of the ACM 17.2 (1974): 83-93.
1. Henriksen, Troels, and Cosmin Eugen Oancea. "[A T2 graph-reduction approach to fusion](https://citeseerx.ist.psu.edu/document?repid=rep1&type=pdf&doi=9540f4e66b50b2347d17004eb7c61e066edebf95)." Proceedings of the 2nd ACM SIGPLAN workshop on Functional high-performance computing. 2013.
1. Wolf, Michael E., and Monica S. Lam. "[A data locality optimizing algorithm](https://www.cs.cmu.edu/afs/cs/Web/People/745/lectures/wolf-lam-data-locality.pdf)." Proceedings of the ACM SIGPLAN 1991 conference on Programming language design and implementation. 1991.

## Polyhedral compliation

1. Zhao, Jie, et al. "[Parallelizing neural network models effectively on gpu by implementing reductions atomically](https://yaozhujia.github.io/assets/pdf/pact2022-paper.pdf))." Proceedings of the International Conference on Parallel Architectures and Compilation Techniques. 2022.
2. Zhao, Jie, and Peng Di. "[Optimizing the memory hierarchy by compositing automatic transformations on computations and data](https://01.me/files/AKG/akg-micro20.pdf)." 2020 53rd Annual IEEE/ACM International Symposium on Microarchitecture (MICRO). IEEE, 2020.
3. Zhao, Jie, et al. "[Apollo: Automatic partition-based operator fusion through layer by layer optimization](https://proceedings.mlsys.org/paper_files/paper/2022/file/e175e8a86d28d935be4f43719651f86d-Paper.pdf)." Proceedings of Machine Learning and Systems 4 (2022): 1-19.

## CUDA and Tensor core optimization

1. Huang, Jianyu, Chenhan D. Yu, and Robert A. van de Geijn. "[Implementing Strassen's algorithm with CUTLASS on NVIDIA Volta GPUs](https://arxiv.org/pdf/1808.07984.pdf)." arXiv preprint arXiv:1808.07984 (2018).

## Backup

- [TVM](rendered/TVM.pdf)
- [Swift for TensorFlow](rendered/swift_for_tensorflow.pdf)
- [Glow](rendered/Glow.pdf)
- [MLIR: A Compiler Infrastructure for the End of Moore's Law](https://arxiv.org/abs/2002.11054)


PLDI 22, Tensor compiler

- [An Asymptotic Cost Model for Autoscheduling Sparse Tensor Programs](https://arxiv.org/pdf/2111.14947.pdf)
- [DISTAL: The Distributed Tensor Algebra Compiler](https://arxiv.org/pdf/2203.08069.pdf)
- [Triton-MLIR: 从DSL到PTX](https://zhuanlan.zhihu.com/p/671434808)