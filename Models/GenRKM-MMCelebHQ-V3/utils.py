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
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=opt.capacity, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(opt.capacity)
        self.conv2 = nn.Conv2d(in_channels=opt.capacity, out_channels=opt.capacity * 2, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(opt.capacity * 2)
        self.conv3 = nn.Conv2d(in_channels=opt.capacity * 2, out_channels=opt.capacity * 4, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(opt.capacity * 4)
        self.conv4 = nn.Conv2d(in_channels=opt.capacity * 4, out_channels=opt.capacity * 8, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(opt.capacity * 8)
        self.fc1 = nn.Linear(opt.capacity * 8 * 8 * 8, opt.x_fdim)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = self.bn1(x)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = self.bn2(x)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = self.bn3(x)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)
        x = self.bn4(x)
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

        self.conv1 = nn.Conv2d(1, c, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(c)
        self.conv2 = nn.Conv2d(c, 2 * c, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(2 * c)
        self.conv3 = nn.Conv2d(2 * c, 4 * c, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(4 * c)
        self.conv4 = nn.Conv2d(4 * c, 8 * c, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(8 * c)

        self.fc1 = nn.Linear(8 * c * 8 * 8, opt.y_fdim)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = self.bn1(x)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = self.bn2(x)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = self.bn3(x)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)
        x = self.bn4(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


# Label Encoder
class NetLaEn(nn.Module):
    def __init__(self, opt):
        super(NetLaEn, self).__init__()
        self.fc1 = nn.Linear(40, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, opt.z_fdim)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc3(x), negative_slope=0.2)
        return x


# Pre-Image Maps - Network Architecture
# Image Decoder
class NetImDe(nn.Module):
    def __init__(self, opt):
        super(NetImDe, self).__init__()
        self.c = opt.capacity
        self.fc1 = nn.Linear(opt.x_fdim, self.c * 8 * 8 * 8)
        self.conv4 = nn.ConvTranspose2d(self.c * 8, self.c * 4, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(self.c * 4, self.c * 2, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(self.c * 2, self.c, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.ConvTranspose2d(self.c, 3, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = x.view(-1, self.c * 8, 8, 8)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
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

        self.fc1 = nn.Linear(opt.y_fdim, 8 * self.c * 8 * 8)
        self.deconv4 = nn.ConvTranspose2d(8 * self.c, 4 * self.c, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(4 * self.c, 2 * self.c, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(2 * self.c, self.c, kernel_size=4, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(self.c, 1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc1(x)
        x = F.leaky_relu(x, negative_slope=0.2)
        x = x.view(-1, 8 * self.c, 8, 8)
        x = F.leaky_relu(self.deconv4(x), negative_slope=0.2)
        x = F.leaky_relu(self.deconv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.deconv2(x), negative_slope=0.2)
        x = torch.sigmoid(self.deconv1(x))
        return x


# Label Decoder
class NetLaDe(nn.Module):
    def __init__(self, opt):
        super(NetLaDe, self).__init__()
        self.fc1 = nn.Linear(opt.z_fdim, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, 40)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = F.leaky_relu(self.fc2(x), negative_slope=0.2)
        x = self.fc3(x)
        return x