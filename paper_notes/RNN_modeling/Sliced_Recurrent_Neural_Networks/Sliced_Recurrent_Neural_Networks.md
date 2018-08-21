# [Sliced Recurrent Neural Networks](https://arxiv.org/abs/1807.02291)

## Model Structure

2 hyperparameters of SRNN

1. slice number $n$
1. slicing times $k$

The input sequence is $X = [x_1, x_2, ..., x_t]$ whose length is $T$.

1. Slice $X$ into $n$ subsequences of equal length.
1. Repeat the above process $k$ times until a pre-defined minimum sequence length is obtained.
1. Apply RNN function to each subsequence.

<p align="center">
<img src="images/SRNN.png" width=70%>
</p>

## My Comments

Personally, I don't think this work is interesting, for the following reasons:

1. SRNN cannot be applied to sequence labeling tasks. How to use it to sequence to sequence models are not clear and not studied.
    * It is only evaluated in text classification (sentiment classification). Text classification is a simple task in the NLP field. Sometimes it does not require "understanding the semantics of the language" (A good sentiment analysis model does need to understand the semantics, which is also the core challenge in NLP field. Whether SRNN shows some advantages over modeling semantics or not requires more careful evaluation). The model can achieve high accuracy by overfitting or capturing some statistical significance of training data.
    * SRNN even cannot be directly used in an RNN LM.
    * The evaluation is not enough.
1. SRNN cannot be stacked for multiple layers which are very important in RNN modeling. If there is only one RNN unit, the state transition between the current and previous state is shallow.
1. SRNN is not novel. How it works is hugely like recursive neural networks which are proposed by Socher several years ago. I don't think it makes new contributions. The work is not reliable.
