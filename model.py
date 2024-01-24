import torch
import torch.nn as nn


# MLP model for MNIST dataset
class ALL_CNN_MNIST(torch.nn.Module):
    def __init__(self):
        super(ALL_CNN_MNIST,self).__init__()
        self.out1 = nn.Linear(28*28, 256)
        self.act = nn.ReLU()
        self.out2 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = x.flatten(1)
        x = self.out1(x)
        x = self.act(x)
        x = self.out2(x)
        return x


