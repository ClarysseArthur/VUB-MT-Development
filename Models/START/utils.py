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

# Hyper-parameters =================
N = 500  # Samples
mb_size = 100  # Mini-batch size
h_dim = 15  # No. of Principal components
capacity = 32
x_fdim1 = 128
learning_rate = 1e-4  # for optimizer
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


def convert_to_imshow_format(image):
    image = image.numpy()
    # convert from CHW to HWC
    return image.transpose(1, 2, 0)


# Feature-maps - network architecture
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
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


class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        self.fc1 = torch.nn.Linear(40, 60)
        self.fc2 = torch.nn.Linear(60, 70)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        return x


# Pre-image maps - network architecture
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
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


class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.fc1 = torch.nn.Linear(70, 60)
        self.fc2 = torch.nn.Linear(60, 40)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = torch.tanh(self.fc2(x))
        return x


net1 = Net1().to(device)
net2 = Net2().to(device)
net3 = Net3().to(device)
net4 = Net4().to(device)
