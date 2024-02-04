A persistent kernel has a governing loop in it that only ends when signaled, otherwise it runs indenfinitely。A persistent kernel是一种kernel设计策略，我们可以把一个persistent kernel类比为一个生产者-消费者模式，something（host code）生产数据，persistent kernel消费数据生产结果。这个生产者-消费者模型can run indefinitely直到没有数据需要消费。consumer simply waits in a loop。

# Reference

1. Some stack overflow answers:
   - [What's the differences between the kernel fusion and persistent thread?](https://stackoverflow.com/questions/58402802/whats-the-differences-between-the-kernel-fusion-and-persistent-thread)
   - [CUDA How Does Kernel Fusion Improve Performance on Memory Bound Applications on the GPU?](https://stackoverflow.com/questions/53305830/cuda-how-does-kernel-fusion-improve-performance-on-memory-bound-applications-on/53311373#53311373)
   - [Doubling buffering in CUDA so the CPU can operate on data produced by a persistent kernel](https://stackoverflow.com/questions/33150040/doubling-buffering-in-cuda-so-the-cpu-can-operate-on-data-produced-by-a-persiste/33158954#33158954)