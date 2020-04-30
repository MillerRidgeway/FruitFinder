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

from subprocess import Popen, DEVNULL, run
from torch.multiprocessing import Pool, Process
from load_data import *
from utils import *
from train import *
from validate import *

# List of machines from lab 325 to use
machines = ['turbot','brill','flounder','sardine','eel']

def dist_train(world_size, rank, auto=False):
    print("dist_train",world_size,rank)
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

    # Number of distributed processes
    #world_size = 2

    # Distributed backend type
    dist_backend = 'nccl'

    # Url used to setup distributed training
    dist_url = "file://" + os.path.join(os.getcwd(), 'dist_store')

    #Process group init
    print("Init process group")
    dist.init_process_group(backend=dist_backend, init_method=dist_url,
                            rank=rank, world_size=world_size)

    if rank == 0 and auto:
        #Launch nodes (this is potentially a very uncontrolled way of doing this)
        proc = [None] * (world_size - 1)

        for i in range(world_size - 1):
            print('Launching node: ' + str(i + 1))
            proc[i] = Popen(["ssh",machines[i],"cd",os.getcwd(),"; ipython dist_test.py",str(world_size),str(i + 1)],)
                         #stdout=DEVNULL,
                         #stderr=DEVNULL,
                         #stdin=DEVNULL)

    #Model init
    print("Model init")
    model = models.resnet18(pretrained=False).cuda()
    model = torch.nn.parallel.DistributedDataParallel(model)

    #Define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = torch.optim.SGD(model.parameters(), starting_lr,
                                momentum=0.9, weight_decay=1e-4)

    #Load data and split for train/test
    print("Load data for training")
    fruit_dataset = load_data();
    train_loader, valid_loader = split_data(fruit_dataset, valid_percent, batch_size, workers, dist=True)

    # Training loop
    best_prec1 = 0

    start = time.time()
    for epoch in range(num_epochs):
        # Adjust learning rate according to schedule
        adjust_learning_rate(starting_lr, optimizer, epoch)

        # train for one epoch
        print("\nBegin Training Epoch {}".format(epoch+1))
        train(train_loader, model, criterion, optimizer, epoch, cuda = True)

        # evaluate on validation set
        print("Begin Validation @ Epoch {}".format(epoch+1))
        prec1 = validate(valid_loader, model, criterion, cuda = True)

        # remember best prec@1 and save checkpoint if desired
        # is_best = prec1 > best_prec1
        best_prec1 = max(prec1, best_prec1)

        print("Epoch Summary: ")
        print("\tEpoch Accuracy: {}".format(prec1))
        print("\tBest Accuracy: {}".format(best_prec1))

    train_time = time.time() - start
    print("Total elapsed training time: ", train_time)

    # Gather metrics for run
    metrics = {'model': 'resnet18',
               'num_epochs': num_epochs,
               'batch_size': batch_size,
               'world_size': world_size,
               'train_time': train_time,
               'best_prec1': best_prec1}

    return model, metrics

def dist_kill():
    '''WARNING!!! This is a sledgehammer! It will kill all ipython procs!'''
    for machine in machines:
        run(["ssh",machine,"killall -9 ipython"])


if __name__ == "__main__":
    world_size = int(sys.argv[1])

    if len(sys.argv) > 2:
        rank = int(sys.argv[2])
        model, metrics = dist_train(world_size, rank, auto=False)
    else:
        model, metrics = dist_train(world_size, 0, auto=True)
