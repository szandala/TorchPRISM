from torchvision import models
import torch
import glob
import os
from torchvision import transforms
import matplotlib.pyplot as plt
import cv2
from torchray.benchmark.datasets import IMAGENET_CLASSES
import torch.nn as nn

crop = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224))
])

normalize = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def read_images_2_batch(folder):
    image_files = glob.glob(f"./samples/imagenet_images/{folder}/*.jpg")
    image_files.sort()

    input_images = [cv2.cvtColor(cv2.imread(f), cv2.COLOR_BGR2RGB)
                    for f in image_files]
    input_batch = torch.stack([normalize(crop(image))
                               for image in input_images])

    return input_batch

def get_categories(model, input_batch):
    percentage = nn.Softmax(dim=1)
    output = percentage(model(input_batch))

    listed_all_outputs = [[(i, val)
                           for i, val in enumerate(o.tolist())] for o in output]
    best_fits = [sorted(sing_out, key=lambda o: o[1])[-1]
                 for sing_out in listed_all_outputs]

    return [IMAGENET_CLASSES[o[0]].split(",")[0].strip().replace(" ", "_").replace("'", "")     for o in best_fits]

if __name__ == "__main__":

    readed = [l.strip() for l in open("readed.txt").readlines()]

    # print(os.listdir("./samples/imagenet_images/"))
    # print(IMAGENET_CLASSES)
    # folders = sorted(os.listdir("./samples/imagenet_images/"))
    # for obj in IMAGENET_CLASSES:
    #     obj = obj.split(",")[0].strip().replace(" ", "_").replace("'", "")
    #     if obj not in folders:
    #         print(obj)

    model = models.vgg16(pretrained=True)
    model.eval()

    for folder in sorted(os.listdir("./samples/imagenet_images/")):

        if folder in readed:
            continue

        print(f"Working on: {folder}...")

        input_batch = read_images_2_batch(folder)
        # print("Fail here")
        category_names = get_categories(model, input_batch)
        # print("Fail here 2")
        to_save = ", ".join(category_names)
        # print("Fail here 3")

        open("classified.txt", "a").write(f"{folder}: {to_save}\n")
        open("readed.txt", "a").write(f"{folder}\n")
