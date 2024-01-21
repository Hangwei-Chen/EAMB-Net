import os
from torchvision import transforms
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import default_loader

IMAGE_NET_MEAN = [0.485, 0.456, 0.406]
IMAGE_NET_STD = [0.229, 0.224, 0.225]
normalize = transforms.Normalize(
            mean=IMAGE_NET_MEAN,
            std=IMAGE_NET_STD)

class PARADataset(Dataset):
    def __init__(self, path_to_csv, images_path,if_train):
        self.df = pd.read_csv(path_to_csv)
        self.images_path =  images_path
        if if_train:
            self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize])
        else:
            self.transform = transforms.Compose([
            transforms.Resize((512, 512)),
            transforms.ToTensor(),
            normalize])

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]
        y = np.array([row['aestheticScore_mean']/5])
        image_id = row['imageName']
        session_id = row['sessionId']
        image_path = os.path.join(self.images_path, session_id,image_id)
        image = default_loader(image_path)
        x = self.transform(image)
        return x, y.astype('float32')

