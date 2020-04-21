import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim

from neural_net import *
from load_data import *

def train(path):
    #Load and split data into train and test
    data = load_data()
    train, valid = split_data(data, 20)
    
    #Check to make sure the data was loaded properly
    data_iter = iter(train)
    images, labels = data_iter.next()
    imshow(torchvision.utils.make_grid(images))
    print(' '.join('%5s' % labels[j] for j in range(4)))

    #Create a CNN to train on
    net = Net()

    #Define an optimizer for the CNN
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    #Train the network with backprop
    for epoch in range(2):  # loop over the dataset multiple times
        running_loss = 0.0
        for i, data in enumerate(train, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:    # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

    print('Finished Training')

    #Save the model
    torch.save(net.state_dict(), path)

    # #Load and eval
    # print("Made it to eval")
    # net = Net()
    # net.load_state_dict(torch.load(path))
    # print("Made it to loading")
    # correct = 0
    # total = 0
    # with torch.no_grad():
    #     for data in valid:
    #         images, labels = data
    #         outputs = net(images)
    #         _, predicted = torch.max(outputs.data, 1)
    #         total += labels.size(0)
    #         correct += (predicted == labels).sum().item()
    #         print(correct/total)

    # print('Accuracy of the network on the 10000 test images: %d %%' % (
    #     100 * correct / total))

PATH = "./model/sample_model.pth"
train(PATH)