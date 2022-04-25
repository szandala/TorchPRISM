# PRISM - **Pr**incipal **I**mage **S**ections **M**apping

![PRISM logo](https://raw.githubusercontent.com/szandala/TorchPRISM/master/PRISM_logo.png)

A novel tool that utilizes Principal Component Analysis to display discriminative featues detected by a given convolutional neural network.
It complies with virtually all CNNs.

# Table of Contents
* [Usage](#Usage)
* [`prism` arguments](#`prism`-arguments)
* [Other arguments](#Other-arguments)
* [Demo](#Demo)
* [Results](#Results)
* [Read more](#Read-more)
* [Citation](#Citation)

## Usage

For user's convenience we have prepared an argument-feedable excutable `prism`.
In order to use it, please prepare virtual env:
```sh
python3 -m venv venv
source ./venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
./prism
```

## `prism` arguments

| Argument | Description | Result |
| :---: | :---: | :---: |
| none | Default PRISM exection with Gradual Extrapolation applied | ![Vanilla result](https://raw.githubusercontent.com/szandala/TorchPRISM/assets/results/PRISM_vanilla.jpg) |
| --no-gradual-extrapolation | Skipping Gradual Extrapolation | ![Disabled Gradual Extrapolation](https://raw.githubusercontent.com/szandala/TorchPRISM/assets/results/PRISM_no-ge.jpg) |
| --inclusive | Quantize colours and show only common for all images in batch | ![Only common features](https://raw.githubusercontent.com/szandala/TorchPRISM/assets/results/PRISM_inclusive.jpg) |
| --inclusive & --no-gradual-extrapolation | Quantize colours and show only common for all images in batch. **Skip GE!** | ![Inclusive, no GE](https://raw.githubusercontent.com/szandala/TorchPRISM/assets/results/PRISM_no-ge_inclusive.jpg) |
| --exclusive | Quantize colours and show only unique features for images in batch | ![Only unique features](https://raw.githubusercontent.com/szandala/TorchPRISM/assets/results/PRISM_exclusive.jpg) |
| --exclusive & --no-gradual-extrapolation | Quantize colours and show only unique features for images in batch. **Skip GE!** | ![Exclusive, no GE](https://raw.githubusercontent.com/szandala/TorchPRISM/assets/results/PRISM_no-ge_exclusive.jpg) |
| --exclusive & --inclusive | Quantize original PRISM output **Skip GE!** | ![Quantized vanilla result without GE](https://raw.githubusercontent.com/szandala/TorchPRISM/assets/results/PRISM_inclusive_exclusive.jpg) |
| --split-rgb |Split PRISM output into separate RGB channels. | ![RGB split](https://raw.githubusercontent.com/szandala/TorchPRISM/assets/results/PRISM_RGB.jpg) |
| --split-rgb & --no-gradual-extrapolation | Also split into RGB, but without Gradual Extrapolation. **Skip GE!** Note it can also go with `--inclusive` or `--exclusive`| ![RGB split, no GE](https://raw.githubusercontent.com/szandala/TorchPRISM/assets/results/PRISM_no-ge_RGB.jpg) |


## Other arguments

| Argument | Description | Default |
| :---: | :---: | :---: |
| --input=`/path/to/...` | Path from where to take images. Note it is a `glob`, so value `./samples/**/*.jpg` will mean: `jpg` images from ALL subfolders of `samples` | `./samples/*.jpg` |
| --model=`model-name` | Model to be used with PRISM. Note that Gradual Extrapolation may not behave properly for some models outside *vgg* family. | vgg16 |
| --help | Print help details and exit |  |

## Demo

[Simplest snippet](https://github.com/szandala/TorchPRISM/blob/master/SoftwareX_snippet/snippet.py) of working code.

```python
import sys
sys.path.insert(0, "../")
from torchprism import PRISM
from torchvision import models
from utils import load_images, draw_input_n_prism

# load images into batch
input_batch = load_images()

model = models.vgg11(pretrained=True)
model.eval()
PRISM.register_hooks(model)

model(input_batch)
prism_maps_batch = PRISM.get_maps()

drawable_input_batch = input_batch.permute(0, 2, 3, 1).detach().cpu().numpy()
drawable_prism_maps_batch = prism_maps_batch.permute(0, 2, 3, 1).detach().cpu().numpy()

draw_input_n_prism(drawable_input_batch, drawable_prism_maps_batch)
```
First we have to import PRISM and torch models., as well as functions for preparing input images as simple torch batch and function to draw batches. Next we have to load the model, in this case a pretrained vgg11 has been chosen and then we have to call the first PRISM method to register required hooks in the model.
With such a prepared model we can perform the classification and, since the actual output is not needed, we can just ignore it. Model execution is followed by using the second PRISM method to calculate features maps for the processed batch. Finally we have to prepare both input and PRISM output so they can be drawn and as the last step we call a method that displays them using e.g. matplotlib.

## Results

The results allow us to see the discriminative features found by the model.
On the sample images below we can see wolves

![Snippet result](https://raw.githubusercontent.com/szandala/TorchPRISM/master/SoftwareX_snippet/PRISM_result.png)

We can notice that all wolves have similar colors - features, found on their bodies. Furthermore the coyote also shows almost identical characteristics except the mouth element. wolves have a black stain around their noses, while coyote does not.

## Read more
- [IEEE Access: Gradual Extrapolation](https://ieeexplore.ieee.org/document/9468713)
- [ICCS-2021: Attention Focus](https://www.iccs-meeting.org/archive/iccs2021/papers/127430415.pdf)
- [ICCS-2022: PRISM](https://docs.google.com/document/d/1jFEyjZdj8HFdPmgQMhqpLgs4Yi_1t7X1/edit?usp=sharing&ouid=111364369145250786679&rtpof=true&sd=true)
*Accepted, not yet published*
- [PP RAI'22: Summary](https://docs.google.com/document/d/1_-TKex_0BW2pV3BO4Uwk6gF3Cer2bB5qUdgxWns0e-4/edit?usp=sharing)

## Citation
Currently in the form of pre-print, but hope soon as publication.

```raw
@misc{szandala2021torchprism,
      title={TorchPRISM: Principal Image Sections Mapping, a novel method for Convolutional Neural Network features visualization},
      author={Tomasz Szandala},
      year={2021},
      eprint={2101.11266},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
