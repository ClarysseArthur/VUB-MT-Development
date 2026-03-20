import math
import os
import shutil
import time
import urllib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


class create_dirs:
    """ Creates directories for Checkpoints and saving trained models """

    def __init__(self, ct):
        self.ct = ct
        self.dircp = 'checkpoint.pth_{}.tar'.format(self.ct)
        self.dirout = 'Mul_trained_RKM_{}.tar'.format(self.ct)

    def create(self):
        if not os.path.exists('cp/'):
            os.makedirs('cp/')

        if not os.path.exists('out/'):
            os.makedirs('out/')

    def save_checkpoint(self, state, is_best):
        if is_best:
            torch.save(state, 'cp/{}'.format(self.dircp))

class Net1(nn.Module):
    def __init__(self, opt):
        super(Net1, self).__init__()
        self.capacity = opt.capacity
        self.x_fdim = opt.x_fdim

        self.conv1 = nn.Conv2d(1, self.capacity, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(self.capacity, self.capacity * 2, kernel_size=4, stride=2, padding=1)
        self.fc1 = nn.Linear(self.capacity * 2 * 7 * 7, self.x_fdim)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


class Net2(nn.Module):
    def __init__(self, opt):
        super(Net2, self).__init__()
        self.y_fdim = opt.y_fdim

        self.fc1 = nn.Linear(10, 16)          # was 15
        self.fc2 = nn.Linear(16, self.y_fdim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.fc2(x)
        return x


class Net3(nn.Module):
    def __init__(self, opt):
        super(Net3, self).__init__()
        self.capacity = opt.capacity
        self.x_fdim = opt.x_fdim

        self.fc1 = nn.Linear(self.x_fdim, self.capacity * 2 * 7 * 7)
        self.conv2 = nn.ConvTranspose2d(self.capacity * 2, self.capacity, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(self.capacity, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = x.view(x.size(0), self.capacity * 2, 7, 7)   # simplified
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = torch.sigmoid(self.conv1(x))
        return x


class Net4(nn.Module):
    def __init__(self, opt):
        super(Net4, self).__init__()
        self.y_fdim = opt.y_fdim

        self.fc1 = nn.Linear(self.y_fdim, 32)   # was 64
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = self.fc2(x)
        return x