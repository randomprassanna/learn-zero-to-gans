import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split

dataset = datasets.MNIST('data', train = True) #without converting to tensor or keepi PIL format

dataset[0][0]

#data augmentation and normalisation
stats = ((0.1307 ), (0.3081)) #these will be used as mean and std dev of mnist data

train_transformation = transforms.Compose([transforms.RandomCrop(28,padding=4,padding_mode='reflect'),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize(*stats,inplace=True)])

val_transformation = transforms.Compose([transforms.ToTensor(),transforms.Normalize(*stats)])



train_ds, val_ds = random_split(dataset, [50000,10000])


class MapDataset(torch.utils.data.Dataset):
    """
    Given a dataset, creates a dataset which applies a mapping function
    to its items (lazily, only when an item is called).

    Note that data is not cloned/copied from the initial dataset.
    """

    def __init__(self, dataset, map_fn):
        self.dataset = dataset
        self.map = map_fn

    #     def __getitem__(self, index):
    #         return self.map(self.dataset[index])

    def __getitem__(self, index):
        if self.map:
            x = self.map(self.dataset[index][0])
        else:
            x = self.dataset[index][0]  # image
        y = self.dataset[index][1]  # label
        return x, y

    def __len__(self):
        return len(self.dataset)


train_sds = MapDataset(train_ds, train_transformation)
val_sds = MapDataset(val_ds, val_transformation)


train_loader = DataLoader(train_sds, batch_size=128, shuffle=True, num_workers=3, pin_memory=True)
val_loader = DataLoader(val_sds, batch_size=128, shuffle=True, num_workers=1, pin_memory=True)


def main():
    for i,l in train_loader:
        print(i.shape)
        print(l.shape)
        break

if __name__ == '__main__':
    main()


