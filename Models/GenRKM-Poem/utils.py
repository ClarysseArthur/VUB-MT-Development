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
x_fdim1 = 64
y_fdim = 100
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
class NetTextEn(nn.Module):
    def __init__(self, vocab=200_000, d=32, hidden1=256, hidden2=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.fc1 = nn.Linear(15*d, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, x_fdim1)

    def forward(self, x):
        x = x.long()
        e = self.emb(x)              # (B,15,d)
        e = e.reshape(x.size(0), -1) # (B,15*d)
        # e = e.double()
        e = F.leaky_relu(self.fc1(e), 0.2)
        e = F.leaky_relu(self.fc2(e), 0.2)
        return self.fc3(e)


class NetSentEn(nn.Module):
    """
    Input:  (B, 4) one-hot float/int
    Output: (B, y_fdim)
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 32)

    def forward(self, x):
        if x.dim() == 1:  # (4,)
            x = x.unsqueeze(0)
        if x.size(-1) != 4:
            raise ValueError(f"NetSentEn expected last dim=4, got {tuple(x.shape)}")

        # x = x.float()
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.fc2(x)
        return x

class NetLenEn(nn.Module):
    """
    Input:  (B, 15) one-hot float/int
    Output: (B, y_fdim)
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(15, 32)
        self.fc2 = nn.Linear(32, 64)

    def forward(self, x):
        if x.dim() == 1:  # (15,)
            x = x.unsqueeze(0)
        if x.size(-1) != 15:
            raise ValueError(f"NetLentEn expected last dim=15, got {tuple(x.shape)}")

        # x = x.float()
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = self.fc2(x)
        return x


class NetTextDe(nn.Module):
    def __init__(self, emb_layer, d=32, hidden1=256, hidden2=256):
        super().__init__()
        self.fc1 = nn.Linear(x_fdim1, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 15*d)
        self.emb = emb_layer  # tie weights
        self.d = d

    def forward(self, z):
        # z = z.float()
        z = F.leaky_relu(self.fc1(z), 0.2)
        z = F.leaky_relu(self.fc2(z), 0.2)
        out = self.fc3(z).view(-1, 15, self.d)   # (B,15,d)
        logits = out @ self.emb.weight.T         # (B,15,V)
        return logits


class NetSentDe(nn.Module):
    """
    Input:  (B, y_fdim)
    Output: (B, 4) logits
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(32, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)

        # z = z.double()
        z = F.leaky_relu(self.fc1(z), 0.2)
        logits = self.fc2(z)       # (B, 4)
        return logits              # use softmax + CE, or BCEWithLogitsLoss if multi-label

class NetLenDe(nn.Module):
    """
    Input:  (B, y_fdim)
    Output: (B, 15) logits
    """
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 15)

    def forward(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)

        # z = z.double()
        z = F.leaky_relu(self.fc1(z), 0.2)
        logits = self.fc2(z)       # (B, 15)
        return logits              # use softmax + CE, or BCEWithLogitsLoss if multi-label