# [Dynamic Routing between Capsule](https://arxiv.org/pdf/1710.09829.pdf)

> Hinton: “The pooling operation used in convolutional neural networks is a big mistake and the fact that it works so well is a disaster.”

> Internal data representation of a convolutional neural network does not take into account important spatial hierarchies between simple and complex objects.

Also see this: [Understanding Hinton’s Capsule Networks](https://medium.com/ai%C2%B3-theory-practice-business/understanding-hintons-capsule-networks-part-i-intuition-b4b559d1159b)


## Motivations

* This paper focus on vision problem: digits recognization on MNIST.
* It makes some new assumptions to guide builsding a visual system.
* **Capsules** is the elementary building block and **dynamic routing** is the algorithm to train such a network.
  1. This paper assumes that s single fixation gives us much more than just a single indentified object and its properties.
  1. It assumes that **the visual system creates a parse tree-like structure on each fixation**.
  1. Ignore the issue of how these single-fixation parse trees are coordinated over multiple fixations.

### What is the building block

* Each layer will be divided into many small groups of neurons call "**Capsules**".
* Eeach node in the parse tree will conrespond to an active capsule.
* Using **an interative routing process**, each active capsule will choose a capsule in the layer **above** to be its parent in the tree.

### What is the **Capsule**

* A capsule is a group of neurons.
* Capsule's activity vector represents the instantiation parameters of a specific type of entiry.
  * such as position, size, orientation, deformation, velocity, albedo, hue, texture, etc.
  * one very special property is **the existence of the instatntiated entiry** in the image.
    * obvious way [**Not used in this parper**]: ~~use a separate logitstic unit to model this existence.~~

>* **Use the overall length of the vector of instantiation parameters to represent the existence of the entity**.
>* **Force the orientation of the vector to represent the properties of the entity**.
>    * ensure that the length of the vector output of a capsule cannot exceed 1.
>    * this is done by applying a non-linearity that leaves the orientation of the vector unchanged but scales down its magnitude.

**Why length and orientation?**

### What is **Dynamic Routing**

* Dynamic routing is used to **ensure that the output of the capsule gets sent to an appropriate parent in the layer above**.
  1. Intially, the output is routed to all possible parents.
  1. For each possible parent, the capsule computes a "prediction vector".
      * If this prediction vector has a large scalar product with the output of a possible parent, there is top-down feedback which increases the couping coefficient for that parent and decreases it for other parents. :arrow_right: "**routing-by-agreement**"

> **Routing-by-agreement is claimed to be far more effective than the very primitive form of routing implemented by max-pooling.**