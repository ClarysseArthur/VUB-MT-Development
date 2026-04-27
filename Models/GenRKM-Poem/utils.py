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


def convert_to_imshow_format(image):
    image = image.numpy()
    # convert from CHW to HWC
    return image.transpose(1, 2, 0)


# Feature-maps - network architecture
class NetTextEn(nn.Module):
    def __init__(self, vocab=200_000, d=64, hidden1=512, hidden2=256, pad_idx=None, opt=None):
        super().__init__()
        assert opt is not None, "opt must be provided"

        self.d = d
        self.emb = nn.Embedding(vocab, d, padding_idx=pad_idx)
        self.fc1 = nn.Linear(15 * d, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, opt.x_fdim)
        self.drop = nn.Dropout(0.1)

    def forward(self, x):
        x = x.long()
        e = self.emb(x).reshape(x.size(0), -1)
        e = self.drop(F.leaky_relu(self.bn1(self.fc1(e)), 0.2))
        e = self.drop(F.leaky_relu(self.bn2(self.fc2(e)), 0.2))
        return self.fc3(e)


class NetSentEn(nn.Module):
    def __init__(self, hidden1=32, hidden2=64, opt=None):
        super().__init__()
        assert opt is not None, "opt must be provided"

        self.fc1 = nn.Linear(4, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, opt.y_fdim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.float()
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return self.fc3(x)


class NetLenEn(nn.Module):
    def __init__(self, hidden1=64, hidden2=64, opt=None):
        super().__init__()
        assert opt is not None, "opt must be provided"

        self.fc1 = nn.Linear(15, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, opt.z_fdim)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        x = x.float()
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.leaky_relu(self.fc2(x), 0.2)
        return self.fc3(x)


class NetTextDe(nn.Module):
    def __init__(self, emb_layer, hidden1=256, hidden2=512, opt=None):
        super().__init__()
        assert opt is not None, "opt must be provided"

        self.emb = emb_layer
        self.d = emb_layer.embedding_dim

        self.fc1 = nn.Linear(opt.x_fdim, hidden1)
        self.bn1 = nn.BatchNorm1d(hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.bn2 = nn.BatchNorm1d(hidden2)
        self.fc3 = nn.Linear(hidden2, 15 * self.d)
        self.drop = nn.Dropout(0.1)

    def forward(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        z = self.drop(F.leaky_relu(self.bn1(self.fc1(z)), 0.2))
        z = self.drop(F.leaky_relu(self.bn2(self.fc2(z)), 0.2))
        out = self.fc3(z).view(-1, 15, self.d)
        logits = out @ self.emb.weight.T
        return logits, out


class NetSentDe(nn.Module):
    def __init__(self, hidden1=64, hidden2=32, opt=None):
        super().__init__()
        assert opt is not None, "opt must be provided"

        self.fc1 = nn.Linear(opt.y_fdim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 4)

    def forward(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        z = F.leaky_relu(self.fc1(z), 0.2)
        z = F.leaky_relu(self.fc2(z), 0.2)
        return self.fc3(z)


class NetLenDe(nn.Module):
    def __init__(self, hidden1=64, hidden2=64, opt=None):
        super().__init__()
        assert opt is not None, "opt must be provided"

        self.fc1 = nn.Linear(opt.z_fdim, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 15)

    def forward(self, z):
        if z.dim() == 1:
            z = z.unsqueeze(0)
        z = F.leaky_relu(self.fc1(z), 0.2)
        z = F.leaky_relu(self.fc2(z), 0.2)
        return self.fc3(z)