# PyTorch Architectures

PyTorch implementations of research paper model architectures

## Module Structure

The `torcharch` package is divided into the following submodules. 
Both `models` an `modules` are further divided into broad categories (i.e. convolutional, recurrent, etc)

* **models:** Complete architecture implementations
* **modules:** Helper components of various architectures
* **data:** Utility functions and full dataset loaders
* **training:** Execute trainings and save model weights

## Implemented Papers

The following list has either some or all of its architecture components and/or datasets implmeneted

[Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition](https://arxiv.org/pdf/1406.4729.pdf)

* `modules.conv.SpatialPyramidPooling`
