from PIL import Image
import sys
from torchvision import transforms
import numpy as np
import torch

sys.path.append('preprocess')
# Get code for face segmentation
# sys.path.append('preprocess/face_segment_models/face_parsing')
# sys.path.append('preprocess/face_segment_models/face_parsing/rtnet')

from preprocess.skelton import get_skelton_image
from preprocess.agnostic import get_agnostic_image
from preprocess.face_segment import get_face_segment_image

from models.generator import UnetGenerator,UnetSkipConnectionBlock

person_image_path='wdiv.png'
cloth_image_path='cloth.png'

person_image=Image.open(person_image_path)
cloth_image=Image.open(cloth_image_path)

skelton_image=get_skelton_image(person_image)
agnostic_image=get_agnostic_image(person_image)
face_segment_image=get_face_segment_image(person_image)

imageSize = (256,256)
transform=transforms.Compose([
    transforms.Resize(imageSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

cloth_tensor = transform(cloth_image)
skelton_tensor = transform(skelton_image)
agnostic_tensor = transform(agnostic_image)
face_segment_tensor = transform(face_segment_image)

input = torch.cat([cloth_tensor,skelton_tensor,face_segment_tensor,agnostic_tensor],dim=0)

model=torch.load('g29',map_location=torch.device('cpu'))

cloth_image.show()
skelton_image.show()
face_segment_image.show()
agnostic_image.show()

output = model(torch.unsqueeze(input,0))[0]
output = output.permute(1, 2, 0).detach().numpy()
output = ((output + 1) / 2) * 255  # Scale values to [0, 255]
output = output.astype(np.uint8)
output = Image.fromarray(output)
output.show()