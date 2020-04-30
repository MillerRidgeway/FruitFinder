import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchvision

from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
from torchvision import transforms, datasets

#Loading data from folder
def load_data():
    transform = transforms.Compose(
        [transforms.Resize((258,320)),
         transforms.ToTensor(),
         transforms.Normalize(mean=[0.5032, 0.5149, 0.4498], std=[0.2390, 0.2399, 0.2564])])

    fruit_dataset = datasets.ImageFolder("data/",transform=transform)
    return fruit_dataset

#Split the dataset into train/test
def split_data(fruit_dataset, valid_percent, batch_size, num_workers, dist = True):
    num_train = len(fruit_dataset)
    split = int(valid_percent*num_train)
    # We need this not to be "random" across nodes
    torch.manual_seed(42)
    trainset, valset = torch.utils.data.dataset.random_split(fruit_dataset,
                                                         [num_train - split, split])

    print("Dataset split into: ", [len(trainset), len(valset)])


    # Create distributed samplers
    train_sampler = None
    if(dist):
        train_sampler = torch.utils.data.distributed.DistributedSampler(trainset)

    #  Create loaders
    train_loader = torch.utils.data.DataLoader(trainset,
                                            batch_size=batch_size, 
                                            shuffle=(train_sampler is None), 
                                            num_workers=num_workers, 
                                            pin_memory=True, 
                                            sampler=train_sampler)

    val_loader = torch.utils.data.DataLoader(valset,
                                            batch_size=batch_size, 
                                            shuffle=False, 
                                            num_workers=num_workers, 
                                            pin_memory=True)
    print ("Loaders created. ", train_loader, val_loader)    

    return train_loader, val_loader