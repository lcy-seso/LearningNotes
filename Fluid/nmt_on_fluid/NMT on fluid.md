# Let us first look into how Fluid and PaddlePaddle implement the `dynamicRNN` and attention.

---

# What is `dynamicRNN` in Fluid?

- The users can **customize the computations in one time-step**, and the framework take the responsibility to iterate over the input sequence.
- The learnable parameters are shared among time steps.

---
# RNN attention and beam search needs `dynamicRNN`

- `dynamicRNN` is used in RNN encoder-decoder with attention models:
    - In RNN with attention, decoder needs to compute the context vector through the attention mechanism.
    - how to compute attention can be customized by users
        - for example, dot-attention, bi-linear attention, cosine attention, and so on.
    - the computation for RNN with attention is serial, not paralleled. **BUT this is not the fact for ConvS2S and the transformer**.
- beam search is a while loop.

---

- Externally, the `dynamicRNN` is implemented by using the `while_op`.
- If `memory` is defined in `dynamicRNN`, it is a recurrent model.
- Otherwise, it is just like `while` in programming language, which iterate over the input sequence.
    - This means the input for `while_op` must be a sequence (LoDTensor in Fluid).

---

# Important elements `for dynamicRNN`
1. **step input**: the LoDTensor upon which the step function is iteratively called.
1. **step function**: the user customized computation in one time step, which will be iteratively invoked in every time step.
1. **memory**: is used to form the "recurrent" input.
1. **external memory**
     - The external memory is first proposed in [NTM](https://arxiv.org/abs/1410.5401).
     - In NMT models, encoder outputs (encoder vectors) are external memory for decoder.
     - Every time step, any operator defined in the step function can read all the contexts of the external memory.

---

- The current user interface of dynamic RNN：

<p align="center">
<img src="images/user_interface.png" width=100%>
</p>

- This example does not contain an external memory.

---

# Memory in Dynamic RNN
- `memory` is used to form the "recurrent" input.
- Logically, `memory` in dynamic RNN acts like the reference variable in C++.
    - It “points to” the output of an operator, let's denote it as operator A.
    - `memory` can be boot with a Tensor/LoDTensor, by default, memory is initialized to zeros.
    - `memory` is forwarded after A.
    - when memory is forward, it retrieve the output Tensor/LoDTensor of A.
    - the output of memory canbe the input for another operator. This forms the "recurrent" connection.

---

- `while_op` **cannot work itself**. It depends on:
    - related operators (Here only lists most important ones, not all):
        - `lod_rank_table` operator
        - `lod_tensor_to_array` operator
        - `array_to_lod_tensor` operator
        - `shrink_memory` operator
    - related data structure
        - `TensorArray`
        - `LoDRankTable`

- One fact: Fluid does not need any padding to support variable-length sequence.
- All the above data structure and operators are all for one purpose: **Batch computation without padding for variable-length sequence**.

---

# How `dynamicRNN` implement batch computation?

---
- Followings are the original inputs for RNN encoder-decoder.
<p align="center">
<img src="images/raw_input.png" width=100%>
</p>

- source word sequences above are outputs of encoder RNN.
- target word sequences above are word embeddings for target language.
- one rectangle in the above picture is a dense vector in CPU/GPU memory.

---

- RNN can be regarded as an expanded feed-forward network.
- The depth of the feed-forward network equals to length of the sequence.
- Without padding, each sequence in a mini-batch may have different lengths, leading to difference steps of forward computation and backward computation for each sequence.

---

# Fluid's and old PaddlePaddle's solution:
1. Sort the mini-batch by its length first, the longest sequence in a mini-batch because the first one.
    - `lod_rank_table`
    - `LoDRankTable`
        - this data structure canbe regarded as sorting by index.
        - LoDRankTable stores the sorted index information.
2. Construct batch input for each time step. The batch input for each time step will become smaller and smaller as step count increases.
    - `lod_tensor_to_array`
    - `TensorArray`
3. For final outputs of `dynamicRNN`, reorder it to its original order according to the `LoDRankTable` created in step 1.
    - `array_to_lod_tensor`

---

# Let's take an example

<p align="center">
<img src="images/sorted_input.png" width=100%>
</p>

---

<p align="center">
<img src="images/1.png" width=100%>
</p>

- input Tensor in batch 5 ~ 7 become smaller.

---


<p align="center">
<img src="images/1.png" width=100%>
</p>

- let's think about what happens in batch 5 ~ 7 for `memory` defined in `dynamicRNN`.
    - `memory` retrieve the output tensor of an operator.
    - in batch 5 ~ 7, ends of sequences in the mini-batch is reached, for these sequences, we do not need an "recurrent" connection for the next time step.
    - `shrink_memory` operator is used to shrink batch input for `memory` in `dynamicRNN`.

---

# batch 1 ~ 2

<p align="center">
<img src="images/2.png" width=100%>
</p>

---

# batch 3 ~ 4

<p align="center">
<img src="images/3.png" width=100%>
</p>

---

# batch 5 ~ 7
<p align="center">
<img src="images/4.png" width=100%>
</p>

---

# How to compute attention
- take the first time step (batch 1 in previous pictures) for example:
    - attention calculate a context vector which is the adaptive weighted sum of the encoded vectors.
    - attention itself is a feed-forward network.

<p align="center">
<img src="images/attention.png" width=80%>
</p>

---

1. decoder states are first expanded (repeated) as long as encoder vectors.
    - **In old PaddlePaddle, this repeating may consume a lot of memory**
    - This is because the memory will not actually be free, there are 2 ~ 3 operator, the memory consumption of whose outputs are proportional to `source sequence length x target sequence length x hidden size`.
1. batch calculation of the similarities between the decoder state and each encoder vectors.
1. normalize the similarities leads to attention weight.
1. calculate the weighted sum of encoder vectors according to attention weight and return the context vectors.
