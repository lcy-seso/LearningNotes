# [A Structured Self-attentive Sentence Embedding](https://arxiv.org/pdf/1703.03130.pdf)
 
---
# background: attention

- $H_{n\times d}=(\mathbf{h}_1, \mathbf{h}_2, ..., \mathbf{h}_n)$ is sentence encoding usually given by LSTM or GRU
    - $n$ is number of words
    - $d$ is the dimension of LSTM 
- $a_{n\times 1}=\text{softmax}(\mathbf{W}_{s2} \tanh(\mathbf {W}_{s1}H))$ is attention weight
- $m_{1 \times d}=\text{sum}(a \cdot H)$ is the vector representation of the input sentence

---

# motivations
- carrying the semantices along all time steps of a recurrent model is relarively hard and not necessary
- extracting different aspects of the sentence into multiple vector representations
- in some NLP task, such as text classification, the model is only given one single sentence, no extra information is given to caculate attention
- self-attention can be used to replace max or average pooling
- in this paper, self-attention is performed on top of an LSTM

---
# Some points

- one $m$ is to reflect an aspect, or a component of the semantices in a sentence.
- multiple $m$ that focus on different parts of the sentence is necessary to represent one sentence (especially long sentence)
- multiple hops of attention is required

---

# model architecture (PART I)
<p align="center">
<img src="images/self_attention_sentence_representation.png" width=30%>
</p>

- in essence, self-attention is a feed-forward model with two fully connected layers.
- $A = \text{softmax}\lbrace W_{s2}\text{tanh}(W_{s1}H^T)\rbrace$ is the weight matrix.
- $M = AH$ is the sentence embedding matrix.

---

# model architecture (PART II)
- Problems:
    - the embedding matrix $M$ can suffer from redundancy problems
    - the attention mechanism always provides similar summation weights for all $r$ hops
- Solution: a penalization term to encourage the diversity of summation weights over different hops
    $$P = \lVert (AA^T - I)\rVert_{F}^2$$
    
    - $\lVert \centerdot \rVert _{F}$ stands for Forbenius norm of a matrix

---
# for reference
- Forbenius Norm of matirx $A_{m\times n}$:
$$\lVert A \rVert_F = \sqrt{\sum_{i=1}^m\sum_{j=1}^n a_{i,j}^2}$$
- also equal to square root of the [matrix trace](http://mathworld.wolfram.com/MatrixTrace.html) of $AA^H$, where $A^H$ is the [conjugate transpose](http://mathworld.wolfram.com/ConjugateTranspose.html).
$$\lVert A \rVert_F = \sqrt{\text{Tr}(AA^H)}$$

---

# About the penaltiy (PART I)
- the penalty measures the redundancy.
- an element of $AA^T$(except the diagonal elements) is the dot product of two distribution:
$$0\lt a_{i,j}=\sum_{k=1}^n a_k^i a_k^j \lt 1$$
- two extreme situation:
    - no overlap between two distributions: $a_{i,j} = 0$
    - two distribution are identical and focus on one single word: $a_{i,j} = 1$
---
# About the penaltiy (PART II)
- substracting $I$ from $AA^T$ to force the elements on the diagonal of $AA^T$ to approximate 1.
- this encourages each distribution to focus on <font color=#DC143C>as few number of words as possible</font>