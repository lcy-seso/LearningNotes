

## Motivations

## Model

<p align="center">
<img src="images/SSD.png" width=100%>
</p>

<p align="center">
<img src="images/SSD2.png">
</p>

1. SSD is based on:
    * A _**feed-forward convolutional network**_ that produces a _**fixed-size**_ collection of bounding boxes and scores for the presence of object class instances in those boxes.
    * Followed by a _**non-maximum suppression**_ step to produce the final detections.
    * Uses VGG16 as the base network.
1. Auxiliary structures:
    * **Multi-scale feature maps for detection.**
      * Convolution layers at the end of base network, sizes of which decrease prograssively.
    * **Convolutional predictors for detection**
      * Each added feature layer can produce a fixed set of detection predictions using a set of convolutional filters.
    * **Default boxes and aspect ratios**
      * Associates a set of default bounding boxes with each feature map cell for multiple feature maps.
      * The default boxes tile the feature map in a convolutional manner. The position of each box relative to the default box shapes in the corresponding cell is fixed.
      * At each feature map cell, predicts:
          1. the offsets relative to the default box shapes in the cell.
          1. the per-class scores.

## Training

* Ground gruth information needs to be assigned to specific outputs in the fixed set of detector outputs.
* Choose:
    1. the set of default boxes and scales for detection.
    1. the hard negative mining.
    1. data augmentation strategies.
