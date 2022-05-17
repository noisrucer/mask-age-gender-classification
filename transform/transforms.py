import albumentations as A
from albumentations.pytorch import ToTensorV2

from PIL import Image
from PIL import ImageEnhance
import numpy as np
import torch
from torchvision.transforms import ToTensor, Resize, AutoAugment, Compose, Normalize

class Transform:
    def __init__(self, img_resize=224, training_mode=True):
        '''
        Initialize transform according to transform_num

        Parameters:
            transform_num 0: General transform applied to all gender/age/mask
            transform_num 1: transform for gender only
            transform_num 2: transform for age only
            transform_num 3: transform for mask only
        '''

        #  assert transform_choice in [0,1,2,3], "{} is an invalid transform choice"
        #  self.transform_choice = transform_choice
        #  trsfm_options = ['general', 'gender', 'age', 'mask']

        self.img_resize = img_resize
        if training_mode:
            self.transform = getattr(self, '_all_transform')()
        else:
            self.transform = getattr(self, '_test_transform')()


    def __call__(self):
        return self.transform

    def _test_transform(self):
        trsfm = Compose([
            Resize(224, Image.BILINEAR),
            ToTensor(),
            Normalize(
                mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)
            )
        ])

        return trsfm

    def _all_transform(self):
        pytorch_trsfm = Compose([
            Resize(224, Image.BILINEAR),
            ToTensor(),
            Normalize(
                mean=(0.548, 0.504, 0.479), std=(0.237, 0.247, 0.246)
            )
        ])

        album_trsfm  = A.Compose([
           A.Crop(30, 30, 354, 472),
           A.Resize(self.img_resize, self.img_resize),
           A.GaussianBlur(p=0.3),
           A.ColorJitter(),
           A.RandomFog(p=0.3),
           A.PixelDropout(dropout_prob=0.1),
           A.RandomGridShuffle(grid=(3,3), p=0.2),
           A.HorizontalFlip(p=0.5),
            A.OpticalDistortion(p=0.2),
           A.Normalize(
                mean=[0.548, 0.504, 0.479], std=[0.237, 0.247, 0.246],
            ),
            ToTensorV2()
        ])

        def trsfm(image):
            if np.random.rand() < 0.7:
                image = ImageEnhance.Sharpness(image).enhance(25)

            image = album_trsfm(image=np.array(image))['image']
            #  image = Image.fromarray(image)
            #  image = pytorch_trsfm(image)
            return image

        return trsfm





