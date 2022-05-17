import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from collections import Counter

class BaseDataLoader(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle  # might change due to validation_split
        self.batch_size = batch_size
        self.n_samples = len(dataset)
        self.dataset = dataset
        #  self.train_sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        if validation_split == 0.0:
            print("Provide validation_split!!!!")

        if validation_split != 0.0:
            dataset_targets = [e[1] for e in dataset]
            train_idx, valid_idx = train_test_split(np.arange(len(dataset_targets)), test_size=validation_split, shuffle=True, stratify=dataset_targets)
            self.train_dataset = Subset(dataset, train_idx)
            self.valid_dataset = Subset(dataset, valid_idx)
        else:
            self.train_dataset = dataset
        #  n_val = int(self.n_samples * validation_split)
        #  n_train = self.n_samples - n_val
        #  self.train_dataset, self.valid_dataset = random_split(dataset, [n_train, n_val])
        print("train valid split done")

        #####WeightedRandomSampler########
        #  target_indices = self.train_dataset.indices
        #  #  target_class = np.array([self.dataset[i][1] for i in target_indices])
        #  target_class = np.array([dataset_targets[i] for i in target_indices])
        #  print("target_class")
        #  class_count = np.array([
        #      len(np.where(target_class == t)[0]) for t in np.unique(target_class)
        #  ])
        #  print("class_count: {}".format(class_count))
        #
        #  weight = 1. / class_count
        #  print("weight: {}".format(weight))
        #
        #  samples_weight = np.array([weight[t] for t in target_class])
        #  samples_weight = torch.from_numpy(samples_weight)

        self.init_kwargs = {
            #  'dataset': dataset,
            'batch_size': batch_size,
            #  'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }
        super().__init__(
            dataset=self.train_dataset, **self.init_kwargs, shuffle=False
        )

    def _split_sampler(self, val_split):
        if val_split == 0.0:
            return None, None
        n_val = int(self.n_samples * val_split)
        n_train = self.n_samples - n_val

        full_idx = np.random.permutation(np.arange(self.n_samples))
        train_idx = full_idx[:n_train]
        val_idx = np.delete(full_idx, np.arange(n_train))

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        # shuffle is redundant with our sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, val_sampler

    def split_validation(self):
        if self.validation_split == 0.0:
            return None

        return DataLoader(dataset=self.valid_dataset, shuffle=True, **self.init_kwargs)
