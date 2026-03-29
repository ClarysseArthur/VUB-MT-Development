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
    def __init__(self, opt):
        super(NetImEn, self).__init__()
        c = opt.capacity
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=c, out_channels=c * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(c * 2)
        self.conv3 = nn.Conv2d(in_channels=c * 2, out_channels=c * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(c * 4)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(c * 4 * 4 * 4, opt.x_fdim)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


# Sketch Encoder
# class NetSkEn(nn.Module):
#     def __init__(self, opt):
#         super(NetSkEn, self).__init__()
#         self.conv1 = nn.Conv2d(1, opt.capacity, kernel_size=4, stride=2, padding=1, padding_mode="reflect")
#         self.conv2 = nn.Conv2d(opt.capacity, 2 * opt.capacity, kernel_size=4, stride=2, padding=1, padding_mode="reflect")
#         self.fc1 = nn.Linear(2 * opt.capacity * 32 * 32, opt.y_fdim)

#     def forward(self, x):
#         x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
#         x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
#         x = x.view(x.size(0), -1)
#         x = self.fc1(x)
#         return x

class NetSkEn(nn.Module):
    def __init__(self, opt):
        super(NetSkEn, self).__init__()
        c = opt.capacity
        # Slightly larger first kernel for sparse line drawings / sketches
        self.conv1 = nn.Conv2d(1, c, kernel_size=7, stride=2, padding=3)
        self.conv2 = nn.Conv2d(c, 2 * c, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * c)
        self.conv3 = nn.Conv2d(2 * c, 4 * c, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(4 * c)
        self.pool = nn.AdaptiveAvgPool2d((4, 4))
        self.drop = nn.Dropout(0.3)
        self.fc1 = nn.Linear(4 * c * 4 * 4, opt.y_fdim)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.drop(x)
        x = self.fc1(x)
        return x


# Label Encoder
class NetLaEn(nn.Module):
    def __init__(self, opt):
        super(NetLaEn, self).__init__()

        h1 = 256
        h2 = 256
        h3 = 128

        self.fc1 = nn.Linear(40, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        self.skip = nn.Linear(h1, h2) if h1 != h2 else nn.Identity()
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.out = nn.Linear(h3, opt.z_fdim)

    def forward(self, x):
        x = x.double()
        x1 = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.2)
        x1 = self.drop1(x1)
        x2 = F.leaky_relu(self.bn2(self.fc2(x1)), negative_slope=0.2)
        x2 = x2 + self.skip(x1)   # residual connection
        x2 = self.drop2(x2)
        x3 = F.leaky_relu(self.bn3(self.fc3(x2)), negative_slope=0.2)
        z = self.out(x3)
        return z


# Pre-Image Maps - Network Architecture
# Image Decoder
class NetImDe(nn.Module):
    def __init__(self, opt):
        super(NetImDe, self).__init__()
        self.c = opt.capacity
        self.fc1 = nn.Linear(opt.x_fdim, 4 * self.c * 16 * 16)
        self.conv3 = nn.ConvTranspose2d(4 * self.c, 2 * self.c, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(2 * self.c)
        self.conv2 = nn.ConvTranspose2d(2 * self.c, self.c, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(self.c)
        self.conv1 = nn.ConvTranspose2d(self.c, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = x.view(-1, 4 * self.c, 16, 16)
        x = F.leaky_relu(self.bn3(self.conv3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.conv2(x)), negative_slope=0.2)
        x = torch.sigmoid(self.conv1(x))
        return x


# Sketch Decoder
# class NetSkDe(nn.Module):
#     def __init__(self, opt):
#         super(NetSkDe, self).__init__()
#         self.c = opt.capacity
#         self.fc1 = nn.Linear(opt.y_fdim, 2 * self.c * 32 * 32)
#         self.conv2 = nn.ConvTranspose2d(2 * self.c, self.c, kernel_size=4, stride=2, padding=1)
#         self.conv1 = nn.ConvTranspose2d(self.c, 1, kernel_size=4, stride=2, padding=1)

#     def forward(self, x):
#         x = self.fc1(x)
#         x = F.leaky_relu(x, negative_slope=0.2, inplace=True)
#         x = x.view(-1, 2 * self.c, 32, 32)
#         x = F.leaky_relu(self.conv2(x), negative_slope=0.2, inplace=True)
#         x = torch.sigmoid(self.conv1(x))
#         return x

class NetSkDe(nn.Module):
    def __init__(self, opt):
        super(NetSkDe, self).__init__()
        self.c = opt.capacity
        self.fc1 = nn.Linear(opt.y_fdim, 4 * self.c * 16 * 16)
        self.deconv3 = nn.ConvTranspose2d(4 * self.c, 2 * self.c, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(2 * self.c)
        self.deconv2 = nn.ConvTranspose2d(2 * self.c, self.c, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(self.c)
        self.deconv1 = nn.ConvTranspose2d(self.c, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = x.view(-1, 4 * self.c, 16, 16)
        x = F.leaky_relu(self.bn3(self.deconv3(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn2(self.deconv2(x)), negative_slope=0.2)
        x = torch.sigmoid(self.deconv1(x))
        return x


# Label Decoder
class NetLaDe(nn.Module):
    def __init__(self, opt):
        super(NetLaDe, self).__init__()

        h1 = 128
        h2 = 256
        h3 = 256

        self.fc1 = nn.Linear(opt.z_fdim, h1)
        self.bn1 = nn.BatchNorm1d(h1)
        self.fc2 = nn.Linear(h1, h2)
        self.bn2 = nn.BatchNorm1d(h2)
        self.fc3 = nn.Linear(h2, h3)
        self.bn3 = nn.BatchNorm1d(h3)
        self.skip = nn.Linear(h1, h2) if h1 != h2 else nn.Identity()
        self.drop1 = nn.Dropout(0.1)
        self.drop2 = nn.Dropout(0.1)
        self.out = nn.Linear(h3, 40)

    def forward(self, x):
        x1 = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.2)
        x2 = F.leaky_relu(self.bn2(self.fc2(x1)), negative_slope=0.2)
        x2 = x2 + self.skip(x1)
        x2 = self.drop1(x2)
        x3 = F.leaky_relu(self.bn3(self.fc3(x2)), negative_slope=0.2)
        x3 = self.drop2(x3)
        x = self.out(x3)   # raw logits
        return x