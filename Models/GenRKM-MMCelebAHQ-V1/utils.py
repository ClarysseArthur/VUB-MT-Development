import math
import os
import shutil
import time
import urllib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image

# Hyper-parameters =================
capacity = 32
x_fdim = 64
y_fdim = 64

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


""" Note: Do not change the architecture, since it is used to initialize the pre-trained models while generation.
    However, while training from scratch, one can ofcourse define new architectures """


# Feature-maps - network architecture
# Map the images to the latent representation, defines the mapping from image to feature space
# After learning 𝜙1, we can make the kernel function as follows: k(x, x’) = 𝜙1(x) ^T 𝜙1(x’)
# At iitialisation, the features are random, they have to be learned
# 𝜙1(x) ← x
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        c = capacity
        self.conv1 = nn.Conv2d(in_channels=1,   out_channels=c, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(in_channels=c,   out_channels=c * 2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(in_channels=c*2, out_channels=c * 4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(in_channels=c*4, out_channels=c * 8, kernel_size=4, stride=2, padding=1)
        self.fc1 = torch.nn.Linear(c * 8 * 16 * 16, x_fdim)

    def forward(self, x):
        x = F.leaky_relu(self.conv1(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv2(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.conv4(x), negative_slope=0.2)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x

# 𝜙2(y) ← y
class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()                        # The init of the NN: Defining the layers
        self.fc1 = torch.nn.Linear(40, 32)                  # Input size 10, output size 15 (updated to input 40)
        self.fc2 = torch.nn.Linear(32, y_fdim)              # Input size 15, output size y_fdim

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)   # Input goes trough first layer, then activation function LeakyReLU
        x = self.fc2(x)                                     # Input goes trough second layer, no activation function here
        return x                                            # Return the output


# Pre-image maps - network architecture
class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()
        c = capacityc = capacity
        self.fc1 = nn.Linear(in_features=x_fdim, out_features=c * 8 * 16 * 16)
        self.deconv4 = nn.ConvTranspose2d(in_channels=c*8, out_channels=c * 4, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(in_channels=c*4, out_channels=c * 2, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(in_channels=c*2, out_channels=c, kernel_size=4, stride=2, padding=1)
        self.deconv1 = nn.ConvTranspose2d(in_channels=c, out_channels=1, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = x.view(x.size(0), capacity * 8, 16, 16)
        x = F.leaky_relu(self.deconv4(x), negative_slope=0.2)
        x = F.leaky_relu(self.deconv3(x), negative_slope=0.2)
        x = F.leaky_relu(self.deconv2(x), negative_slope=0.2)
        x = torch.sigmoid(self.deconv1(x))
        return x


class Net4(nn.Module):
    def __init__(self):
        super(Net4, self).__init__()
        self.fc1 = torch.nn.Linear(y_fdim, 32)
        self.fc2 = torch.nn.Linear(32, 40)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        x = torch.sigmoid(self.fc2(x))
        return x


# MNIST

class FastMNIST(datasets.MNIST):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Scale data to [0,1]
        self.data = self.data.unsqueeze(1).double().div(255)

        # Put both data and targets on GPU in advance
        self.data, self.targets = self.data, torch.nn.functional.one_hot(self.targets).double()

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        return img, target


def get_mnist_dataloader(args, path_to_data='mnist'):
    """MNIST dataloader with (28, 28) images."""

    all_transforms = transforms.Compose([transforms.ToTensor()])
    train_data = FastMNIST(path_to_data, train=True, download=True, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=args.mb_size, shuffle=args.shuffle, pin_memory=False,
                              num_workers=0)
    _, c, x, y = next(iter(train_loader))[0].size()
    return train_loader, c * x * y, c


def final_compute(args, net1, net2, kPCA, device=torch.device('cuda')):
    """ Function to compute embeddings of full dataset. """
    args.shuffle = False
    xt, _, _ = get_mnist_dataloader(args=args)  # loading data without shuffle
    xtr = net1(xt.dataset.train_data[:args.N, :, :, :].to(args.device))
    ytr = net2(xt.dataset.targets[:args.N, :].to(args.device))

    h, s = kPCA(xtr, ytr)
    return torch.mm(torch.t(xtr), h), torch.mm(torch.t(ytr), h), h, s

# SKETCHES
class MMCelebAHQ(Dataset):
    def __init__(self, *args, **kwargs):
        self.transform = kwargs['transform']

        dataset_source = "../MMCelebA-HQ/DATASET/"

        self.images = []
        self.sketches = []
        self.labels = []

        for i in range(args[0].N):
            image = self.transform(np.array(Image.open(dataset_source + 'image/' + str(i) + '.jpg').convert('RGB')).astype(np.double))
            sketch = self.transform(np.array(Image.open(dataset_source + 'sketch/' + str(i) + '.jpg').convert('1')).astype(np.double))
            label = np.array(open(dataset_source + 'label/' + str(i) + '.txt').read().split(',')).astype(np.double)

            label[label == -1] = 0
            label = torch.tensor(label, dtype=torch.double).T

            self.images.append(image)
            self.sketches.append(sketch)
            self.labels.append(label) # Already multi-hot encoded


    def __getitem__(self, i):
        image = self.images[i]
        sketch = self.sketches[i]
        label = self.labels[i]

        return sketch, label

    def __len__(self):
        return len(self.images)

def get_mmcelebahq_dataloader(args):
    all_transforms = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda t: t.double() / 255.0 if t.max() > 1 else t.double())]) # Define transform from img (RGB array) to tensor
    train_data = MMCelebAHQ(args, transform=all_transforms)
    train_loader = DataLoader(train_data, batch_size=args.mb_size, shuffle=args.shuffle, pin_memory=False, num_workers=0)
    _, c, x, y = next(iter(train_loader))[0].size()

    return train_loader, c * x * y, c

def mmcelebahq_final_compute(args, net1, net2, kPCA, device=torch.device('cuda')):
    """ Function to compute embeddings of full dataset. """
    print(args.N)
    args.shuffle = False
    xt, _, _ = get_mmcelebahq_dataloader(args=args)  # loading data without shuffle
    xtr = net1(torch.stack(xt.dataset.sketches)[:args.N, :, :, :].to(args.device))
    ytr = net2(torch.stack(xt.dataset.labels)[:args.N, :].to(args.device))

    h, s = kPCA(xtr, ytr)
    return torch.mm(torch.t(xtr), h), torch.mm(torch.t(ytr), h), h, s