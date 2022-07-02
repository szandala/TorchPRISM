import torch
import numpy as np
import torch.nn as nn
import os, glob
from torchvision import transforms
from torchvision import models
from torchprism import PRISM
import matplotlib.pyplot as plt
from PIL import Image
import json


with open("classes.json") as json_file:
    CLASSES = json.load(json_file)

CLASSES_IDs = { int(k): v.split(",")[0].replace(" ", "_") for k,v in CLASSES.items() }
CLASSES_NAMEs = { v.split(",")[0].replace(" ", "_").lower(): int(k) for k,v in CLASSES.items() }

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def read_images_2_batch():
    image_files = glob.glob("./samples/*.jpg")
    image_files.sort()

    input_images = [Image.open(f) for f in image_files]
    input_batch = torch.stack([transform(image) for image in input_images])

    return image_files, input_images, input_batch

def prepare_network(arch):
    model = models.__dict__[arch](pretrained=True)
    model.eval()
    PRISM.register_hooks(model)
    return model

def print_output(output, image_files_names):
    listed_output = [ { CLASSES_IDs[i]: val for i, val in enumerate(o.tolist()) } for o in output ]
    classification = []
    for i, name in enumerate(image_files_names):
        print(f"\n{name}:")
        classes = []
        for k, v in sorted(listed_output[i].items(), key=lambda o: o[1], reverse=True)[:5]:
            # print(f"{k}: {v:.2f}")
            classes.append(f"{k}: {v:.2f}")
        classification.append(classes)
    return classification

def normalize_image(image):
    # change image tensor from -1,1 to 0,1
    return (image - image.min()) / (image.max() - image.min())

if __name__ == "__main__":
    arches = [
        # "vgg16",
        # "vgg11",
        # "vgg16",
        # "vgg19",
        # "resnet18",
        # "resnet50",
        # "resnet101",
        # "googlenet",
        # "alexnet",
        # "mobilenet_v2",
        # "squeezenet1_0"
    ]

    for arch in arches:
        with torch.no_grad():
            print(arch)
            PRISM.prune_old_hooks(None)
            model = prepare_network(arch)

            image_files_names, input_images, input_batch = read_images_2_batch()

            if torch.cuda.is_available():
                print("Running on GPU")
                input_batch = input_batch.to("cuda")
                model.to("cuda")


            output = model(input_batch)
            percentage = nn.Softmax(dim=1)
            # print(f"SHAPE {output.shape}")
            classification = print_output(percentage(output), image_files_names)

            prism_maps = PRISM.get_maps().permute(0, 2, 3, 1).detach().cpu().numpy()

            columns = input_batch.shape[0]
            fig, ax = plt.subplots(nrows=2, ncols=columns)
            input_batch = input_batch.permute(0, 2, 3, 1).detach().cpu().numpy()

            if columns == 1:
                ax[0].imshow(normalize_image(input_batch[0]))
                ax[0].set_title("\n".join(classification[0]), fontsize=3)
                ax[0].axis('off')
                ax[1].imshow(prism_maps[0])
                ax[1].axis('off')
            else:
                for column in range(columns):
                    ax[0][column].imshow(normalize_image(input_batch[column]))
                    ax[0][column].set_title("\n".join(classification[column]), fontsize=3)
                    ax[0][column].axis('off')

                for column in range(columns):
                    ax[1][column].imshow(prism_maps[column])
                    ax[1][column].axis('off')

            fig.suptitle(f'PRISM', fontsize=10)
            fig.tight_layout()
            fig.subplots_adjust(top=0.99)
            plt.savefig(f"results/PRISM_{arch}.jpg", format='jpg', bbox_inches="tight", dpi=500)
