from torch.utils.data import Dataset
import pandas as pd
from torchvision import transforms
from PIL import Image
import os
import torch

class VitonDataset(Dataset):
    def __init__(self, data_dir, transform, types_folders, imageSize):
        self.data_dir = data_dir
        self.transform = transform
        self.types_folders = types_folders
        self.imageSize=imageSize

        self.data = pd.read_csv(data_dir+'/train_pairs.txt', 
                                sep=" ", header=None)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images_list=[]
        resize=transforms.Compose([
            transforms.Resize(self.imageSize),
            transforms.ToTensor()
        ])
        for folder in self.types_folders:
            folder_path = os.path.join(self.data_dir, folder)
            image_path = os.path.join(folder_path, self.data[0][idx])
            image = Image.open(image_path)
            image = resize(image)
            images_list.append(image)

        images=torch.cat(images_list)
        images=self.transform(images)
        return images[:-3],images[-3:]