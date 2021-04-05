import torch
import numpy as np
import torch.nn as nn
import os, glob, sys
from torchvision import transforms
from torchvision import models
from torchprism import PRISM
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import json


with open("classes.json") as json_file:
    CLASSES = json.load(json_file)

CLASSES_IDs = { int(k): v.split(",")[0].replace(" ", "_") for k,v in CLASSES.items() }
CLASSES_NAMEs = { v.split(",")[0].replace(" ", "_").lower(): int(k) for k,v in CLASSES.items() }

crop = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224))
])

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

def read_images_2_batch():
    image_files = glob.glob("./samples/e/*.jpg")
    image_files.sort()

    input_images = [ cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB) for f in image_files ]
    input_batch = torch.stack([normalize(crop(image)) for image in input_images])

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
    return ((image - image.min()) / (image.max() - image.min()))




# def kmeans_color_quantization(image, clusters=8, rounds=1):
#     h, w = image.shape[:2]
#     samples = np.zeros([h*w,3], dtype=np.float32)
#     count = 0

#     for x in range(h):
#         for y in range(w):
#             samples[count] = image[x][y]
#             count += 1

#     compactness, labels, centers = cv2.kmeans(samples,
#             clusters,
#             None,
#             (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10000, 0.0001),
#             rounds,
#             cv2.KMEANS_RANDOM_CENTERS)

#     centers = np.uint8(centers)
#     res = centers[labels.flatten()]
#     return res.reshape((image.shape))

def quantize(img):

    # img = ((img - img.min()) / (img.max() - img.min()))
    # img[img<0.1] = 0.0
    # img[img>=0.1] = 0.1
    # img[img>=0.2] = 0.2
    # img[img>=0.3] = 0.3
    # img[img>=0.4] = 0.4
    # img[img>=0.5] = 0.5
    # img[img>=0.6] = 0.6
    # img[img>=0.7] = 0.7
    # img[img>=0.8] = 0.8
    # img[img>=0.9] = 0.9
    # return img
    # return np.around(img, decimals=1)
    # return (0.2 * np.round(img*2 / 0.2))/2
    # print(0.25 * np.round(img / 0.25))

    return 0.25 * np.round(img / 0.25)

def find_common(images):
    color_sets = []
    for img in images:
        color_table = set()
        # [np.unique(row, axis=1) for row in img]
        for row in img:
            for pixel in row:
                color_table.add(pixel.tobytes())
        color_sets.append(color_table)
    common_colors = set.intersection(*color_sets)
    print(len(common_colors))


if __name__ == "__main__":
    arches = ["vgg16"]
        # "vgg11",
        # "vgg16",
        # "vgg19",
        # "resnet18",
        # "resnet50",
        # "resnet101",
        # "googlenet",
        # "alexnet",
        # "mobilenet_v2",
        # "squeezenet1_0"]

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

            # find_common(quantize(prism_maps))
            # sys.exit(0)

            plt.title(f"PRISM")
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

            fig.tight_layout()
            plt.savefig(f"PRISM_{arch}.png", format='jpg', bbox_inches="tight", dpi=500)
