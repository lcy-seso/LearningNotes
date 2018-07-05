# [TVM](https://tvm.ai/about)

## Goals

* Automatically generate deployable codes that are performance-competitive with state-of-art vendor-specific libraries.
* Through automatically generate codes, address the problem that handcrafting operator kernels for the massive space of back-end specific operators and operator combinations.

## Motivations

- Current deep learning framework rely on a computation graph representation.
- Challenges and Goals 
  1. High-level dataflow rewriting.
      * kernel fusion
      * data layout optimization
  2. Memory reuse across threads.
      * cooperation among threads on shared memory
  3. Tensorized computation intrinsics.
  4. Latency Hiding.

## Contributions

- TVM <span style="background-color:#A3D1D1;"> separate the algorithm description, schedule, and hardware interface </span>.
- TVM presents two stage optimization
  1. computation graph level optimization
      1. operator fusion
      1. data layerout transformation
  1. <span style="background-color:#A3D1D1;"> _**tensor level optimization**_ </span>
      1. _**Tensor Expression Language**_ takes cues from Halide.
          * descirbe both  <span style="background-color:#A3D1D1;">the users' intended compute description</span> and <span style="background-color:#A3D1D1;">the abstractions that the hardware exposes</span>.
          * commutative reduction operator
          * high-order scan operator : _to form recurrent computation_
      2. introduce _**schedule primitives**_ to decouple computation description and schedule.
          * adopt useful primitives from Halide and _introduce new ones (?)_ to tackle the chanllenges introduced by GPU and specialized hardware accelerators.
      3. Nested parallelism with cooperation
          * traditional solution for parallelism: _**shared-nothing nested parallelism (fork-join parallelism)**_
          * introduce the concept _**memory scope**_ so that a stage canbe marked as shared.
            *  the shared task needs to compute the dependencies of all the working threads.
            *  use persist threads
            *  memory synchronization barriers need to be properly inserted.
      4. Tensorization: (1) inputs are *ndarrays*; (2) dicdate different data layerout.
          1. chanllenges: 
              1. DL workloads have high [arithmetic intensity](https://en.wikipedia.org/wiki/Roofline_model#Arithmetic_intensity).
              2. cannot resort to a fixed set of primitives.
          2. separate the hardware interface from the schedule
              * _**declare the behavior of each new hardware intrinsic**_.
          3. introduce _**a tensorize schedule primitive**_
              * replace a unit of computation with the corresponding tensor intrinsics.
      5. Latency hiding: decoupled-access/execute philosophy
          1. assume the hardware pipline consists of memory and compute stages that can execute concurrently.
          2. use FIFO queues to implement explicit denpendency tracking.
          3. introduce _**virtual thread schedule primitive**_: programming at low-level is diffcult and painstaking.
