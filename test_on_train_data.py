import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
from PIL import Image
from models.generator import UnetGenerator,UnetSkipConnectionBlock
    
model=torch.load('g20',map_location=torch.device('cpu'))

if isinstance(model, torch.nn.DataParallel):
    generator = model.module


class VitonDataset(Dataset):
    def __init__(self, data_dir, transform, types_folders, diff_cloth = False):
        self.data_dir = data_dir
        self.transform = transform
        self.types_folders = types_folders
        self.diff_cloth=diff_cloth

        self.data = pd.read_csv(data_dir+'/train_pairs.txt', 
                                sep=" ", header=None)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images_list=[]
        for folder in self.types_folders:
            folder_path = os.path.join(self.data_dir, folder)
            if folder=='cloth' and self.diff_cloth:
                image_path = os.path.join(folder_path, self.data[1][idx])
            else:
                image_path = os.path.join(folder_path, self.data[0][idx])
            image = Image.open(image_path)
            image = self.transform(image)
            images_list.append(image)

        inp=torch.cat(images_list)
        folder_path = os.path.join(self.data_dir, 'image')
        image_path = os.path.join(folder_path, self.data[0][idx])
        image = Image.open(image_path)
        out = self.transform(image)
        return inp,out
    
from torchvision import transforms

data_directory = 'dataset'
imageSize = (256,256)
types_folders=['cloth','skeleton','face_segment','agnostic-v3.2']
batch_size=9

transform=transforms.Compose([
    transforms.Resize(imageSize),
    transforms.ToTensor(),
    transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
])

dataset = VitonDataset(data_dir=data_directory, 
                           transform=transform, types_folders=types_folders,diff_cloth=True)
dataloader =  DataLoader(dataset, batch_size=batch_size, shuffle=True)

def convert(image):
    return image.permute(1,2,0).detach().cpu().numpy()

batch_inp,batch_out=next(iter(dataloader))
gen_images=model(batch_inp)

fig, axs = plt.subplots(3, 9)

# Flatten the 4x4 subplot array to simplify indexing
axs = axs.ravel()

# Loop through the 16 images and display them on each subplot
for i in range(len(gen_images)):
    axs[3*i].imshow(convert(batch_out[i]))
    axs[3*i].axis('off')  # Turn off axis labels
    axs[3*i+1].imshow(convert(batch_inp[i,:3,:,:]))
    axs[3*i+1].axis('off')  # Turn off axis labels
    axs[3*i+2].imshow(convert(gen_images[i]))
    axs[3*i+2].axis('off')  # Turn off axis labels

plt.show()

