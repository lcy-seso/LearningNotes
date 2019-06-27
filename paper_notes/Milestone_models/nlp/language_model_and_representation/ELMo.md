# [Deep contextualized word representation](https://arxiv.org/abs/1802.05365)

Deep contextualized word representation models:
1. complex characteristics of word use
1. how these uses vary across linguistic contexts.

Word vectors in this papers are _**learned functions of the internal states of a deep bi-directional language model**_.

ELMo

1. Each token is assigned a representation that is a function of the entire input sentence.
1. ELMo representations are deep.
    - learn a linear combination of the vectors stacked above each input work for each end task.
    - combining the internal sates in this manner allows for every rich word representation.
