<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Hierarchical Multiscale Recurrent Neural Networks](#hierarchical-multiscale-recurrent-neural-networks)
	- [Takeaways](#takeaways)
	- [Problem proposed in this paper](#problem-proposed-in-this-paper)
	- [Approach](#approach)
	- [Model](#model)
	- [Some claims made in this paper](#some-claims-made-in-this-paper)
- [References](#references)

<!-- /TOC -->

# Hierarchical Multiscale Recurrent Neural Networks

## Takeaways



## Problem proposed in this paper

Learn the hierarchical multiscale structure from temporal data _**without explicit boundary information**_.

## Approach

1. [**key**] <span style="background-color:#B3D9D9">Introduce _**a parametrized binary boundary detector**_</span> at each layer.
    - turned on only at the time steps where a segment of the corresponding abstraction level is completely processed.
1. Implement three operations: **UPDATE**, **COPY**, **FLUSH**.
    - **UPDATE**: similar to update rule of the LSTM.
    - **COPY**: _**simply copies**_ cell and hidden states of the previous time step which is unlike the _**leaky integration**_ in LSTM/GRU.
    - **FLUSH**: executed when a boundary is detected, where it first ejects the summarized representation of the current segment to the upper layer and then reinitializes the states to start processing the next segment.
1. Use _**straight-through estimator**_ to train this model to learn to select a proper operation at each time step and to detect the boundaries.
    - This contains discrete variables.

## Model



## Some claims made in this paper

1. However, considering the fact that _**non-stationarity is prevalent in temporal data**_, and that many entities of abstraction such as words and sentences are in variable length, we claim that _**it is important for an RNN to dynamically adapt its timescales**_ to the particulars of the input entities of various length.
1. It has been a challenge for an RNN to discover the latent hierarchical structure in temporal data without explicit boundary information.
1. Although the LSTM has a _**self loop for the gradients that helps to capture the long-term dependencies**_ by mitigating the vanishing gradient problem, in practice, it is still limited to a few hundred time steps due to the leaky integration by which the contents to memorize for a long-term is gradually diluted at every time step.

# References

1. [Hierarchical Multiscale Recurrent Neural Networks](https://arxiv.org/pdf/1609.01704.pdf)
