import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torchvision  
from torchvision import transforms, datasets

#Loading data from folder
def load_data():
    images_dir = os.path.dirname(os.path.dirname(__file__)) + "/FruitFinder/data/" 
    fruit_dataset = datasets.ImageFolder(images_dir,transform=transforms.ToTensor())
    return fruit_dataset

#Split the dataset into train/test
def split_data(dataset,valid_percent):
    num_train = len(dataset)
    indicies = list(range(num_train))
    split = int(np.floor(valid_percent*num_train))

    train_idx, valid_idx = indicies[split:], indicies[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset, 100, train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset, 100, valid_sampler)

    for i, (inputs, targets) in enumerate(train_loader):
        print(i, inputs, targets)
    return train_loader, valid_loader

data = load_data()
train, test = split_data(data, 20)
print(train, test)