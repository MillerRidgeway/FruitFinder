import time
import sys
import torch
import os

import torch.nn as nn
import torch.nn.parallel
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

from utils import *
from validate import *
from train import *
from load_data import *

#Params
# Batch Size for training and testing
batch_size = 128

# Percentage of dataset to reserve for validation
valid_percent = 0.2

# Number of additional worker processes for dataloading
workers = 2

# Number of epochs to train for
num_epochs = 5

# Starting Learning Rate
starting_lr = 0.1

#Model init
model = models.resnet18(pretrained=False)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), starting_lr, 
                            momentum=0.9, weight_decay=1e-4)

#Load Data
fruit_dataset = load_data()
train_loader, valid_loader = split_data(fruit_dataset, 0.2, batch_size, workers, dist=False)

# Training loop
best_prec1 = 0

start = time.time()
for epoch in range(num_epochs):
    # Adjust learning rate according to schedule
    adjust_learning_rate(starting_lr, optimizer, epoch)

    # train for one epoch
    print("\nBegin Training Epoch {}".format(epoch+1))
    train(train_loader, model, criterion, optimizer, epoch, cuda = False)

    # evaluate on validation set
    print("Begin Validation @ Epoch {}".format(epoch+1))
    prec1 = validate(valid_loader, model, criterion, cuda = False)

    # remember best prec@1 and save checkpoint if desired
    # is_best = prec1 > best_prec1
    best_prec1 = max(prec1, best_prec1)

    print("Epoch Summary: ")
    print("\tEpoch Accuracy: {}".format(prec1))
    print("\tBest Accuracy: {}".format(best_prec1))
    
print("Total elapsed training time: ", time.time() - start)

