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
