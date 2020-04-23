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
    [transforms.Resize((224, 224)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    
    fruit_dataset = datasets.ImageFolder("data/",transform=transform)
    return fruit_dataset

#Split the dataset into train/test
def split_data(dataset,valid_percent, batch_size, num_workers, pin_memory = True, dist = True):
    num_train = len(dataset)
    split = int(np.floor(valid_percent*num_train))

    train_idx, valid_idx = torch.utils.data.dataset.random_split(dataset,
                                                         [num_train - split, split])

    sampler = None
    if(dist):
        sampler = torch.utils.data.distributed.DistributedSampler(train_idx)  

    train_loader = torch.utils.data.DataLoader(dataset, 
                                                batch_size = batch_size, 
                                                num_workers = num_workers,
                                                shuffle=(sampler is None),
                                                sampler = sampler,
                                                pin_memory=pin_memory)
    valid_loader = torch.utils.data.DataLoader(dataset, 
                                                 batch_size = batch_size,
                                                 num_workers = num_workers,
                                                 pin_memory=pin_memory,
                                                 shuffle=False)

    return train_loader, valid_loader

#Utility function to show images
def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()