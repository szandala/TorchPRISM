# PRISM vs other methods

## Table of Contents
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

## Research sample
For the comparison we have chosen 2 images. One representing Border Collie dog and second with Border Collie and Siberian Husky specimens.

## Targeted vs UnTargeted
Distinction between Targeted and UnTargeted methods means that some of them aim to highlight features with repect to chosen class. UnTargeted methods try to explain what model has seen in the image and does not specify how important it has been to the certain classes. In other words they are agnostic to the classification.


## UnTargeted Methods

### PRISM

The first comes of course the Principal Image Sections Mapping (PRISM). It relies on extracting the important features that contributed to the model output. While output for a single image does not provide worldbreaking changes, if we expand process to several images we the PRISM starts to shine. It clearly highlights recognized features.

![comparable output - PRISM](./output_PRISM.jpg)

### DeConvNet

First, we will compare results to the DeconvNet proposed by Zeiler et al. This was actually one of the first of techniques prposed for explaining DNNs.

The central idea of Zeiler et al. is to visualize layer activations of a ConvNet by running them through a "DeconvNet" - a network that undoes the convolutions and pooling operations of the ConvNet until it reaches the input space. Deconvolution is defined as convolving an image with the same filters transposed, and unpooling is defined as copying inputs to the spots in the (larger) output that were maximal in the ConvNet (i.e., an unpooling layer uses the switches from its corresponding pooling layer for the reconstruction). Any linear rectifier in the ConvNet is simply copied over to the DeconvNet.

DeconvNet exactly corresponds to simply backpropagating through the ConvNet, except for the linear rectifier. So again, this can be implemented by modifying the gradient of the rectifier: Instead of propagating the error back to every positive input, propagate back all positive error signals. Note that this is equivalent to applying the linear rectifier to the error signal.

![comparable output - DeConvNet](./output_deconvnet.jpg)

```raw
Zeiler, Matthew D., and Rob Fergus. "Visualizing and understanding convolutional networks." European conference on computer vision. Springer, Cham, 2014.
```

### Guided Backpropagation

Springenberg et al. proposes to change a tiny detail for backpropagating through the linear rectifier $y(x) = max(x, 0) = x \cdot [x &gt; 0]$, the nonlinearity used in all the layers of the network except for the final output layer . Here, $[\cdot]$ is the indicator function in a notation promoted by Knuth.

The gradient of the rectifier's output writes its input is defined as follows: $\frac{dy}{dx} y(x) = [x &gt; 0]$. So when backpropagating an error signal $\delta_i$ through the rectifier, we retain $\delta_{i-1} = \delta_i \cdot [x &gt; 0]$. Springenberg et al. propose an additional limitation: In addition to propagating the error back to every positive input, only propagate back positive error signals: $\delta_{i-1} = \delta_i \cdot [x &gt; 0] \cdot [\delta_i &gt; 0]$. They term this "guided backpropagation", because the gradient is guided not only by the input from below, but also by the error signal from above.

**Nonetheless, in Guided Backpropagation visualizations classes are still indistinguishable.**

![comparable output - Guided Backpropagation](./output_guidedBP.jpg)

```raw
Springenberg, J., et al. "Striving for Simplicity: The All Convolutional Net." ICLR (workshop track). 2015.
```

## Targeted Methods

### Grad-CAM

![comparable output - Grad-CAM](./output_grad-cam.jpg)

### Guided Grad-CAM

![comparable output - Guided Grad-CAM](./output_guided-grad-cam.jpg)
```raw
Selvaraju, Ramprasaath R., et al. "Grad-cam: Why did you say that?." arXiv preprint arXiv:1611.07450 (2016).
```
### Excitation Backpropagation

![comparable output - Excitation Backpropagation](./output_excitationBP.jpg)

### Linear Approximation

![comparable output -Linear Approximation](./output_linear-approx.jpg)
