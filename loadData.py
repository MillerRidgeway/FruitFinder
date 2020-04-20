import torch
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import transforms, datasets
import glob

#Loading data from folder
imageDir = glob.glob("../data/*/total/")
#fullDataset = datasets.ImageFolder(imageDir,transform=transforms.ToTensor)