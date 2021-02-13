# PRISM vs other methods

## Table of Contents
* [Research Sample](#Research_Sample)
* [Targeted vs UnTargeted](#Targeted_vs_UnTargeted)
* [UnTargeted Methods](#UnTargeted_Methods)
  * [PRISM](#PRISM)
  * [DeConvNet](#DeConvNet)
  * [Guided Backpropagation](#Guided_Backpropagation)
* [Targeted Methods](#Targeted_Methods)
  * [Grad-CAM](#Grad-CAM)
  * [Guided Grad-CAM](#Guided_Grad-CAM)
  * [Excitation Backpropagation](#Excitation_Backpropagation)
  * [Linear approximation](#Linear_approximation)

RISE?

## Research Sample
For the comparison we have chosen 2 images. One representing Border Collie dog and second with Border Collie and Siberian Husky specimens.

## Targeted vs UnTargeted
Distinction between Targeted and UnTargeted methods means that some of them aim to highlight features with repect to chosen class. UnTargeted methods try to explain what model has seen in the image and does not specify how important it has been to the certain classes. In other words they are agnostic to the classification.

## UnTargeted Methods

### PRISM

The first comes of course the Principal Image Sections Mapping (PRISM). It relies on extracting the important features that contributed to the model output. While output for a single image does not provide worldbreaking changes, if we expand process to several images we the PRISM starts to shine. It clearly highlights with different color recognized features in all considered images thus gives us insight, through set difference, into which features are discriminative for classes.

![comparable output - PRISM](./output_PRISM.jpg)

### DeConvNet

First, we will compare results to the DeconvNet proposed by Zeiler et al. This was actually one of the first of techniques prposed for explaining DNNs. The central idea of Zeiler et al. is to visualize layer activations of a ConvNet by running them through a "DeconvNet" - a network that undoes the convolutions and pooling operations of the ConvNet until it reaches the input space. Deconvolution is defined as convolving an image with the same filters transposed, and unpooling is defined as copying inputs to the spots in the (larger) output that were maximal in the ConvNet (i.e., an unpooling layer uses the switches from its corresponding pooling layer for the reconstruction). Any linear rectifier in the ConvNet is simply copied over to the DeconvNet.

DeconvNet exactly corresponds to simply backpropagating through the ConvNet, except for the linear rectifier. So again, this can be implemented by modifying the gradient of the rectifier: Instead of propagating the error back to every positive input, propagate back all positive error signals. Note that this is equivalent to applying the linear rectifier to the error signal.

![comparable output - DeConvNet](./output_deconvnet.jpg)

```raw
Zeiler, Matthew D., and Rob Fergus. "Visualizing and understanding convolutional networks." European conference on computer vision. Springer, Cham, 2014.
```

### Guided Backpropagation

Springenberg et al. proposes to change a tiny detail for backpropagating through the linear rectifier, the nonlinearity used in all the layers of the network except for the final output layer.

The guided-backpropagation method determines the parts of a particular input image that resulted in a strongly winning class activation. The guided backpropagation process, which is a mixture of backpropagation based on the input data and a deconvolution of the gradient, hides the influence of negative gradients which decrease the activation of the target neuron while highlighting regions that strongly affect the target neuron.

This results in a map displayplying pixels that contributed the most to the overall model output. Nonetheless, in Guided Backpropagation visualizations classes are still indistinguishable.

![comparable output - Guided Backpropagation](./output_guidedBP.jpg)

```raw
Springenberg, J., et al. "Striving for Simplicity: The All Convolutional Net." ICLR (workshop track). 2015.
```

## Targeted Methods

### Gradient

![comparable output - Gradient](./output_gradient.jpg)


### Grad-CAM

![comparable output - Grad-CAM](./output_grad-cam.jpg)

```
Selvaraju, Ramprasaath R., et al. "Grad-cam: Visual explanations from deep networks via gradient-based localization." Proceedings of the IEEE international conference on computer vision. 2017.
```

### Guided Grad-CAM

Grad-CAM may be combined with existing pixel-space visualizations, most notably with Guided Backpropagation, to create a high-resolution class-discriminative visualization.

![comparable output - Guided Grad-CAM](./output_guided-grad-cam.jpg)

```raw
Selvaraju, Ramprasaath R., et al. "Grad-cam: Why did you say that?." arXiv preprint arXiv:1611.07450 (2016).
```
### Excitation Backpropagation

![comparable output - Excitation Backpropagation](./output_excitationBP.jpg)

### Linear Approximation

![comparable output -Linear Approximation](./output_linear-approx.jpg)
