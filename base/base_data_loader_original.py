import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from torchsampler import ImbalancedDatasetSampler
import torchvision
import torch.utils.data

class ImbalancedDatasetSampler2(ImbalancedDatasetSampler):
    def _get_labels(self, dataset):
        if self.callback_get_label:
            return self.callback_get_label(dataset)
        elif isinstance(dataset, torchvision.datasets.MNIST):
            return dataset.train_labels.tolist()
        elif isinstance(dataset, torchvision.datasets.ImageFolder):
            return [x[1] for x in dataset.imgs]
        elif isinstance(dataset, torchvision.datasets.DatasetFolder):
            return dataset.samples[:][1]
        elif isinstance(dataset, torch.utils.data.Subset):
            return dataset.classes
        elif isinstance(dataset, torch.utils.data.Dataset):
            return dataset.get_labels()
        else:
            raise NotImplementedError


class BaseDataLoaderOriginal(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, train_indices=None, val_indices=None, training_mode=True, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle  # might change due to validation_split
        self.batch_size = batch_size
        self.n_samples = len(dataset)
        self.train_indices = train_indices
        self.val_indices = val_indices

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }

        # For evaluation phase, set sampler to None to get the full dataset
        if training_mode == True:
            self.train_dataset = Subset(dataset, train_indices)
            self.val_dataset = Subset(dataset, val_indices)
        if training_mode == False:
            self.train_dataset = dataset

        self.init_kwargs.update({
            'dataset': self.train_dataset
        })

        super().__init__(**self.init_kwargs, pin_memory=True)


    def split_validation(self):
        if self.validation_split is None:
            return None

        self.init_kwargs.update({
            'dataset': self.val_dataset
        })
        device = torch.device('cuda')
        return DataLoader(**self.init_kwargs)

