import torch
from torchvision import transforms
from PIL import Image
import numpy as np

def get_agnostic_image(img):
    model=torch.load('preprocess/agnostic_model',map_location=torch.device('cpu'))
    imageSize = (256,256)
    transform=transforms.Compose([
        transforms.Resize(imageSize),
        transforms.ToTensor(),
        transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
    ])
    img_tensor = transform(img)
    agnostic_image = model(torch.unsqueeze(img_tensor,0))[0]
    agnostic_image = agnostic_image.permute(1, 2, 0).detach().numpy()
    agnostic_image = ((agnostic_image + 1) / 2) * 255  # Scale values to [0, 255]
    agnostic_image = agnostic_image.astype(np.uint8)
    return Image.fromarray(agnostic_image)
