import torch

import numpy as np
import torch.nn.functional as F

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

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
    all_transforms = transforms.Compose([transforms.ToTensor()])

    train_data = FastMNIST(path_to_data, train=True, download=True, transform=all_transforms)
    test_data = FastMNIST(path_to_data, train=False, download=True, transform=all_transforms)

    train_loader = DataLoader(
        train_data,
        batch_size=args.mb_size,
        shuffle=args.shuffle,
        pin_memory=False,
        num_workers=0
    )

    test_loader = DataLoader(
        test_data,
        batch_size=args.mb_size,
        shuffle=False,
        pin_memory=False,
        num_workers=0,
        drop_last=True
    )

    sample, _ = train_data[0]
    c, x, y = sample.size()

    return train_loader, test_loader, c * x * y, c


def final_compute(args, net1, net2, kPCA, device=torch.device('cuda')):
    """ Function to compute embeddings of full dataset. """
    args.shuffle = False
    xt, xte, _, _ = get_mnist_dataloader(args=args)  # loading data without shuffle
    xtr = net1(xt.dataset.train_data[:args.N, :, :, :].to(args.device))
    ytr = net2(xt.dataset.targets[:args.N, :].to(args.device))

    h, s = kPCA(xtr, ytr)
    return torch.mm(torch.t(xtr), h), torch.mm(torch.t(ytr), h), h, s

def final_compute_1V(args, net1, kPCA, device=torch.device('cuda')):
    """ Function to compute embeddings of full dataset. """
    args.shuffle = False
    xt, _, _, _ = get_mnist_dataloader(args=args)  # loading data without shuffle
    xtr = net1(xt.dataset.train_data[:args.N, :, :, :].to(args.device))

    h, s = kPCA(xtr)
    return torch.mm(torch.t(xtr), h), h, s