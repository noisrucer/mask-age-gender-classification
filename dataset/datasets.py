import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import Dataset

import os
import pandas as pd
from PIL import Image
import numpy as np
#  from base import BaseDatasetByImageFolder
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset by Image Folder
#  class FaceMaskDataset_practice(BaseDatasetByImageFolder):
#      def __init__(self, data_dir, transform=None):
#          super().__init__(data_dir, transform)

# Custom Dataset
class FaceMaskDataset(Dataset):
    def __init__(self, data_dir, csv_path, transform=None, training_mode=True):
        # csv_path includes full path. We don't use self.data_dir!
        self.data_dir = data_dir
        self.csv_path = csv_path
        self.transform = transform

        self.df = pd.read_csv(csv_path)
        self.training_mode = training_mode


    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        data_row = self.df.iloc[idx]
        col_path = 'path' if self.training_mode else 'ImageID'
        col_label = 'age' if self.training_mode else 'ans'
        #  real_age = data_row['real_age']

        img_full_path = data_row[col_path]
        if self.data_dir != "none":
            img_full_path = os.path.join(self.data_dir, img_full_path)

        img = Image.open(img_full_path)
        label = data_row[col_label]

        if self.transform:
            #  img_np = np.array(img)
            img = self.transform(img)

        #  img = img.to(device)
        #  label = label.to(device)

        return img, label
