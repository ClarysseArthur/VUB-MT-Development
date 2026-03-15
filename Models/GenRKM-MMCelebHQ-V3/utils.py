import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import autograd
import time
import os
import shutil


N = 500                 # Samples
mb_size = 100           # Mini-batch size
capacity = 32
x_fdim1 = 128
learning_rate = 1e-4    # for optimizer
max_epochs = 5000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


# Feature-Maps - Network Architecture
# Image Encoder
class NetImEn(nn.Module):
    def __init__(self):
        super(NetImEn, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=c*2, out_channels=c * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=c*4, out_channels=c * 8, kernel_size=4, stride=2, padding=1)
        self.fc1 = torch.nn.Linear(c*8*8*8, x_fdim1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)
        x = x.view(x.size(0), -1)  # flatten batch of multi-channel feature maps to a batch of feature vectors
        x = self.fc1(x)
        return x

# Sketch Encoder
class NetSkEn(nn.Module):
    def __init__(self):
        super(NetSkEn, self).__init__()
        c = capacity

        self.conv1 = nn.Conv2d(1,   c,   kernel_size=4, stride=2, padding=1, padding_mode="reflect")
        self.conv2 = nn.Conv2d(c,   2*c, kernel_size=4, stride=2, padding=1, padding_mode="reflect")
        self.fc1 = nn.Linear(2 * c * 32 * 32, x_fdim1)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# Label Encoder
class NetLaEn(nn.Module):
    def __init__(self):
        super(NetLaEn, self).__init__()
        self.fc1 = torch.nn.Linear(40, 60)
        self.fc2 = torch.nn.Linear(60, 70)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        return x


# Pre-Image Maps - Network Architecture
# Image Decoder
class NetImDe(nn.Module):
    def __init__(self):
        super(NetImDe, self).__init__()
        c = capacity
        self.fc1 = nn.Linear(in_features=x_fdim1, out_features=c*8*8*8)
        self.conv4 = nn.ConvTranspose2d(in_channels=c*8, out_channels=c * 4, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(in_channels=c*4, out_channels=c * 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c, out_channels=3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # x = F.relu(self.fc2(x))
        x = self.fc1(x)
        if x.dim() == 1:
            x = x.view(1, capacity * 8, 8, 8)
        else:
            x = x.view(x.size(0), capacity * 8, 8, 8)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = torch.sigmoid(self.conv1(x))
        return x

# Sketch Decoder
class NetSkDe(nn.Module):
    def __init__(self):
        super(NetSkDe, self).__init__()
        c = capacity
        self.c = c  # store for forward reshape

        self.fc1   = nn.Linear(in_features=x_fdim1, out_features=2 * c * 32 * 32)
        self.conv2 = nn.ConvTranspose2d(in_channels=2 * c, out_channels=c,     kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(in_channels=c,     out_channels=1,     kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        # x: [B, x_fdim1] or [x_fdim1]
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.2, inplace=True)

        # Works for both 1D and 2D inputs:
        x = x.view(-1, 2 * self.c, 32, 32)

        x = F.leaky_relu(self.conv2(x), negative_slope=0.2, inplace=True)
        x = torch.sigmoid(self.conv1(x))
        return x

# Label Decoder
class NetLaDe(nn.Module):
    def __init__(self):
        super(NetLaDe, self).__init__()
        self.fc1 = torch.nn.Linear(70, 60)
        self.fc2 = torch.nn.Linear(60, 40)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = torch.tanh(self.fc2(x))
        return x