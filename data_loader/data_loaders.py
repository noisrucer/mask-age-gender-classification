#  from base import BaseDataLoader
from base import BaseDataLoaderOriginal
from dataset.datasets import FaceMaskDataset
from torchvision import transforms
import os
import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from transform import Transform
from utils.util import load_pickle
from torch.utils.data import Sampler

class FaceMaskDataLoader(BaseDataLoaderOriginal):
    def __init__(self, data_dir, csv_path, batch_size, shuffle=True,
                 validation_split=0.0, num_workers=1, img_resize=224, training_mode=True, KFold_samplers_path=None, Fold_num=None):

        trsfm = Transform(img_resize=img_resize, training_mode=training_mode)()

        self.dataset = FaceMaskDataset(data_dir, csv_path, transform=trsfm, training_mode=training_mode)
        print("Dataset loaded! ")
        print("Dataset Length: {}".format(len(self.dataset)))
        print("-"*50)

        if training_mode:
            KFold_train_val_indices = load_pickle(KFold_samplers_path)
            Fold_num = 'Fold' + str(Fold_num)
            train_indices = np.random.permutation(KFold_train_val_indices[Fold_num]['train'])
            val_indices = np.random.permutation(KFold_train_val_indices[Fold_num]['val'])
        else:
            train_indices = None
            val_indices = None

        #  train_sampler = Sampler(train_indices)
        #  val_sampler = Sampler(val_indices)

        super().__init__(self.dataset, batch_size, shuffle,
                         validation_split, num_workers, train_indices=train_indices, val_indices=val_indices, training_mode=training_mode)
