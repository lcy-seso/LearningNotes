# Reading List (from 2019/05/28)

1. [Dryad: Distributed Data-Parallel Programs from Sequential Building Blocks](https://www.microsoft.com/en-us/research/wp-content/uploads/2007/03/eurosys07.pdf)
1. [Swift: A language for distributed parallel scripting](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.720.8476&rep=rep1&type=pdf)
1. [Distributed Data-Parallel Computing Using a High-Level Programming Language](http://michaelisard.com/pubs/sigmod09.pdf)
1. ~~[TensorFlow Eager: A multi-stage, Python-embedded DSL for machine learning](https://www.sysml.cc/doc/2019/88.pdf)~~
1. [Ray: A Distributed Framework for Emerging AI Applications](https://arxiv.org/pdf/1712.05889.pdf)
1. [Real-Time Machine Learning: The Missing Pieces](https://arxiv.org/pdf/1703.03924.pdf)
1. ~~[Beyond Data and Model Parallelism for Deep Neural Networks](https://arxiv.org/pdf/1807.05358.pdf)~~
1. ~~[JANUS: Fast and Flexible Deep Learning via Symbolic Graph Execution of Imperative Programs](https://arxiv.org/pdf/1812.01329.pdf)~~
1. ~~[Pydron: semi-automatic parallelization for multi-core and the cloud](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.1018.8549&rep=rep1&type=pdf)~~
1. ~~[AutoGraph: Imperative-style Coding with Graph-based Performance](https://arxiv.org/abs/1810.08061)~~
1. ~~[Chainer: a Next-Generation Open Source Framework for Deep Learning](http://learningsys.org/papers/LearningSys_2015_paper_33.pdf)~~
1. ~~[A Computational Model for TensorFlow An Introduction](https://dl.acm.org/citation.cfm?id=3088527)~~
1. [Timely Dataflow: A model](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/43546.pdf)

---

## Paper for incumbent DL systems

>_**do not read this. This is more for reference.**_

1. TensorFlow
    - Theano A CPU and GPU Math Compiler in Python
    - DistBelief Large Scale Distributed Deep Networks
    - TensorFlow: A System for Large-Scale Machine Learning (OSDI 2016)
1. PyTorch Automatic differentiation in PyTorch
1. MxNet MXNet: A Flexible and Efficient Machine Learning Library for Heterogeneous Distributed Systems (NIPS 2016)
1. TVM: An Automated End-to-End Optimizing Compiler for Deep Learning (OSDI 2018)
1. Chainer:a Next-Generation Open Source Framework for Deep Learning
1. DyNet The Dynamic Neural Network Toolkit
1. [This is not a research paper] Habana-Gaudi-Training-Platform-whitepaper

## Efforts around TensorFlow

TensorFlow's design has its root in dataflow architecture which is a long line system research.
1. A Computational Model for TensorFlow An Introduction
1. TensorFlow's controlflow design and implementation
    - Dynamic Control Flow in Large-Scale Machine Learning
    - Implementation of Control Flow in TensorFlow
1. TensorFlow Eager: A multi-stage, Python-embedded DSL for machine learning
1. AutoGraph: Imperative-style Coding with Graph-based Performance
1. Compiling machine learning programs via high-level tracing (This is JAX based on TensorFlow XLA)

## Src2src AD

1. Tangent Automatic differentiation using source-code transformation for dynamically typed array programming

## Dynamic models
1. Cavs: An Efficient Runtime System for Dynamic Neural Networks
1. Deep Learning with Dynamic Computation Graphs (ICLR 2017)
1. Static Automatic Batching in TensorFlow
1. Improving the expressiveness of deep learning frameworks with recursion (EuroSys18)

## ML system
1. Real-Time Machine Learning: The Missing Pieces (work from the same research group of the Ray work)
1. Ray: A Distributed Framework for Emerging AI Applications (OSDI 2018)
1. Project Adam: Building an Efficient and Scalable Deep Learning Training System
1. Orpheus: Efficient Distributed Machine Learning via System and Algorithm Co-design
1. Towards Federated Learning at Scale: System Design (SysML 2019, this is from Google, Federated learning is a very othoganal area)
1. PyTorch-BigGraph: A Large-scale Graph Embedding System (SysML’19 PyTorch forGNN)

## Parallelism
1. Beyond Data and Model Parallelism for Deep Neural Networks
1. PipeDream: Fast and Efficient Pipeline Parallel DNN Training
1. GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism
1. Parallax: Sparsity-aware Data Parallel Training of Deep Neural Networks (EuroSys 19, git link)

## Imperative to symbolic style

>_**This is a front-end problem. Transform imperative script into symbolic DAG description so that the computation could be distributed to clusters. Most of these work prototype under Python. If we are going to translate JPL script into symbolic computation description, we can dive into details of these work.**_

1. Dryad: Distributed Data-Parallel Programs from Sequential Building Blocks
1. Pydron: semi-automatic parallelization for multi-core and the cloud
1. JANUS: Fast and Flexible Deep Learning via Symbolic Graph Execution of Imperative Programs
1. AutoGraph: Imperative-style Coding with Graph-based Performance
1. Swift: A language for distributed parallel scripting
1. [Distributed Data-Parallel Computing Using a High-Level Programming Language](http://michaelisard.com/pubs/sigmod09.pdf) (sigmod 09)

## Distributed ML
1. Horovod fast and easy distributed deep learning in TensorFlow
1. TicTac Accelerating Distributed Deep Learning with Communication Scheduling
1. NeuGraph: Parallel Deep Neural Network Computation on Large Graphs (ATC 19)
1. Scalable Deep Learning on Distributed Infrastructures: Challenges, Techniques and Tools (A survey for reference)
1. Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis (A survey)

## Severing
1. SERF: Efficient Scheduling for Fast Deep Neural Network Serving via Judicious Parallelism
1. DeepCPU: Serving RNN-based Deep Learning Models 10x Faster
1. Low Latency RNN Inference with Cellular Batching (EuroSys 2018)
1. PRETZEL: Opening the Black Box of Machine Learning Prediction Serving Systems (OSDI)
1. Optimize CNN Model Inferences on CPUs (ATC 19)
1. A Case for Managed and Model-less Inference Serving (HotOS'19)

## Code-gen/automatic kernel optimization/IR/DL compiler (not well classified)
1. [On Optimizing Operator Fusion Plans for Large-Scale Machine Learning in SystemML](http://www.vldb.org/pvldb/vol11/p1755-boehm.pdf)
1. Polyhedral Optimization of TensorFlow Computation Graphs
1. A modern compiler infrastructure for deep learning systems with adjoint code generation in a domain-specific IR (authors of this project worked for S4T)
1. Halide
    - Automatically Scheduling Halide Image Processing Pipelines
    - Halide A Language and Compiler for Optimizing Parallelism, Locality, and Recomputation in Image Processing Pipelines
    - Differentiable Programming for Image Processing and Deep Learning in Halide
1. [Glow Graph Lowering Compiler Techniques for Neural Networks](https://arxiv.org/pdf/1805.00907.pdf) (from facebook for PyTorch's ecosystem)
1. nGraph
    - Intel [nGraph An Intermediate Representation Compiler and Executor for Deep Learning](https://arxiv.org/pdf/1801.08058.pdf)
    - nGraph-HE: A Graph Compiler for Deep Learning on Homomorphically Encrypted Data

## Dataflow architecture
1. Timely Dataflow: A model
1. Dataflow computers: their history and future
1. Reducing control overhead in dataflow architectures
