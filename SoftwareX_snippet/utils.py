import torch
import os, glob
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2

_crop = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224))
])
_normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def load_images():
    image_files = glob.glob("../samples/*.jpg")
    image_files.sort()

    input_images = [ cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in image_files ]
    input_batch = torch.stack([_normalize(_crop(image)) for image in input_images])

    return input_batch

def _normalize_image(image):
    # change image tensor from -1,1 to 0,1
    return (image - image.min()) / (image.max() - image.min())

def draw_input_n_prism(drawable_input_batch, drawable_prism_maps_batch):
    plt.title(f"PRISM")

    columns = drawable_input_batch.shape[0]
    fig, ax = plt.subplots(nrows=2, ncols=columns)
    if columns == 1:
        ax[0].imshow(_normalize_image(drawable_input_batch[0]))
        ax[0].set_title("\n".join(classification[0]), fontsize=3)
        ax[0].axis('off')
        ax[1].imshow(drawable_prism_maps_batch[0])
        ax[1].axis('off')
    else:
        for column in range(columns):
            ax[0][column].imshow(_normalize_image(drawable_input_batch[column]))
            ax[0][column].axis('off')

        for column in range(columns):
            print(column)
            ax[1][column].imshow(drawable_prism_maps_batch[column])
            ax[1][column].axis('off')

    fig.tight_layout()
    plt.savefig(f"PRISM_result.png", format='png', bbox_inches="tight", dpi=500)
