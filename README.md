# PyTorch Architectures

PyTorch implementations of research paper model architectures

## Motivation

Increasingly machine learning research papers have provided accompanying code on Github.
However, these repositories can often be rather monolithic making it difficult to re-purpose components of the
architecture for a new use-case.
In contrast, `torcharch` focuses strictly on architecture *components* rather than complete end-to-end models.
The goal is to share generic designs that can be easily re-used.


## Implemented Papers

The following list has either some or all of its architecture components implemented

[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/pdf/1406.4729.pdf)

* `conv.SpatialPyramidPooling`

[Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144.pdf)

* `conv.FeaturePyramid`
