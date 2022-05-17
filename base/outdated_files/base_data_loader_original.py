import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, Subset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
from sklearn.model_selection import train_test_split


class BaseDataLoaderOriginal(DataLoader):
    def __init__(self, dataset, batch_size, shuffle, validation_split, num_workers, collate_fn=default_collate):
        self.validation_split = validation_split
        self.shuffle = shuffle  # might change due to validation_split
        self.batch_size = batch_size
        self.n_samples = len(dataset)
        #  self.train_sampler, self.valid_sampler = self._split_sampler(self.validation_split)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers
        }

        #  if validation_split != 0.0:
        if validation_split != 0.0:
            dataset_targets = [e[1] for e in dataset]
            train_idx, valid_idx = train_test_split(np.arange(len(dataset_targets)), test_size=validation_split, shuffle=True, stratify=dataset_targets)
            self.train_dataset = Subset(dataset, train_idx)
            self.valid_dataset = Subset(dataset, valid_idx)
            self.init_kwargs.update({'dataset': self.train_dataset})

        super().__init__(**self.init_kwargs)

    def _split_sampler(self, val_split):
        if val_split == 0.0:
            return None, None
        n_val = int(self.n_samples * val_split)
        n_train = self.n_samples - n_val

        #  full_idx = np.random.permutation(np.arange(self.n_samples))
        full_idx = np.arange(self.n_samples)
        train_idx = full_idx[:n_train]
        val_idx = np.delete(full_idx, np.arange(n_train))

        train_sampler = SubsetRandomSampler(train_idx)
        val_sampler = SubsetRandomSampler(val_idx)

        # shuffle is redundant with our sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, val_sampler

    def split_validation(self):
        if self.validation_split is None:
            return None

        self.init_kwargs.update({'dataset': self.valid_dataset})
        return DataLoader(**self.init_kwargs)






#####WeightedRandomSampler########
#  train_dataset = random
#  target_indices = torch.tensor(list(self.train_sampler))
#  target_indices = target_list[torch.randperm(len(target_list))]

#  target_class = np.array([dataset[i] for i in target_indices])
#  class_count = np.array([
    #  len(np.where(target_class == t)[0]) for t in np.unique(target_class)
#  ])
#  print("class_count: {}".format(class_count))

#  weight = 1. / class_count
#  print("weight: {}".format(weight))

#  samples_weight = np.array([weight[t] for t in target_class])
