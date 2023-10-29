from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os
import torch

class VitonDataset(Dataset):
    def __init__(self, data_dir, transform, types_folders):
        self.data_dir = data_dir
        self.transform = transform
        self.types_folders = types_folders

        self.data = pd.read_csv(data_dir+'/train_pairs.txt', 
                                sep=" ", header=None)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        images_list=[]
        for folder in self.types_folders:
            folder_path = os.path.join(self.data_dir, folder)
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