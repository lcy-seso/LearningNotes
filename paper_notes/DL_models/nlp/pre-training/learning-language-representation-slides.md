<!-- $theme: default -->

# What you can cram into a single vector ?

<p align="center">
Ying Cao
</p>

---

## NLP’s quest for learning language representations

<p align="center">
<img src="images/intro.png">
</p>

---

<font size=5>

## Backgroud (I)

### Transductive transfer learning vs. inductive transfer learning

Imagine you have a training data, but only a subset of it has labels.

For example, you are trying to classify whether an image has a flower in it or not:

- You have 100,000 images,
- 1,000 images that you know definitively contain a flower;
- and another 1,000 that you know don't contain a flower;
- The other 98,000 you have no idea about -- maybe they have flowers, maybe they don't.

</font>

---

<font size=5>

## Backgroud (II)

1. _**Transductive Transfer Learning**_ (转导推理，aks. semi-supervised learning)
    - the other 98,000 images don't have labels, but they tell me something about the problem space.
    - still use the unlabeled data in training to help improve accuracy.
1. _**Inductive Transfer Learning**_ （归纳推理）
    - only looks at the 2,000 labeled examples, builds a classifier on this.
1. _**Active Learning**_ (主动学习)
    - when looking through the 98,000 examples, select a subset and request labels from an oracle.
    - the algorithm might say "of those 98,000, can you label this set of 50 images for me? That will help me build a better classifier".

</font>

---

---

An ideal language representation should model: _syntax_, _semantic_ and _polysemy_.

- An example of polysemy:

    >I read the book yesterday.
    >Can you read the letter now?

---
