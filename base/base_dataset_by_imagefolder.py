import torch
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os

class BaseDatasetByImageFolder(object):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        self.dataset = ImageFolder(root=data_dir, transform=self.transform)
        
    def __call__(self):
        return self.dataset
    
        