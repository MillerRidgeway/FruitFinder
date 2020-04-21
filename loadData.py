import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, Dataset, SubsetRandomSampler
import torchvision  
from torchvision import transforms, datasets

#Loading data from folder
def load_data():
    #images_dir = os.path.dirname(os.path.dirname(__file__)) + "/FruitFinder/data/" 
    transform = transforms.Compose(
    [transforms.Resize((256,256)),
     transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])    
    fruit_dataset = datasets.ImageFolder("data/",transform=transform)
    return fruit_dataset

#Split the dataset into train/test
def split_data(dataset,valid_percent):
    num_train = len(dataset)
    indicies = list(range(num_train))
    split = int(np.floor(valid_percent*num_train))

    train_idx, valid_idx = indicies[split:], indicies[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(dataset, 4, train_sampler)
    valid_loader = torch.utils.data.DataLoader(dataset, 4, valid_sampler)

    return train_loader, valid_loader

def imshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

def run():
    data = load_data()
    train, test = split_data(data, 20)
    
    data_iter = iter(train)
    images, labels = data_iter.next()
    imshow(torchvision.utils.make_grid(images))

run()