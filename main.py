import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from neural_net import Net
from load_data import *

def main():
    #Load and split data into train and test
    data = load_data()
    train, test = split_data(data, 20)
    
    #Check to make sure the data was loaded properly
    data_iter = iter(train)
    images, labels = data_iter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % labels[j] for j in range(4)))

    #Create a CNN to train on
    net = Net()

main()