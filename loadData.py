import os
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision  
from torchvision import transforms, datasets

#Loading data from folder
imageDir = os.path.dirname(os.path.dirname(__file__)) + "/FruitFinder/data/"
fullDataset = datasets.ImageFolder(imageDir,transform=transforms.ToTensor)