import torch

import numpy as np
import pandas as pd
import torch.nn.functional as F


from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

class MMCelebAHQ(Dataset):
    def __init__(self, *args, **kwargs):
        self.im_transform = kwargs['transform']
        self.sk_transform = kwargs['transform']

        dataset_source = "../../Datasets/MMCelebAHQ/DATASET/"

        self.images = []
        self.sketches = []
        self.labels = []

        print("info", args[0].start_iter, args[0].start_iter + args[0].N, 1)

        for i in range(args[0].start_iter, args[0].start_iter + args[0].N, 1):
            image = self.im_transform(np.array(Image.open(dataset_source + 'image/' + str(i) + '.jpg').convert('RGB')).astype(np.double))
            sketch = self.sk_transform(np.array(Image.open(dataset_source + 'sketch/' + str(i) + '.jpg').convert('1')).astype(np.double))
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

        return image, sketch, label

    def __len__(self):
        return len(self.images)

def get_mmcelebahq_dataloader(args):
    transform = transforms.Compose([
        transforms.ToTensor(), # Convert the PIL image to a tensor
        transforms.Lambda(lambda t: F.interpolate(t.unsqueeze(0), size=(128, 128), mode="bilinear", align_corners=False).squeeze(0)), # Convert the image from 256x256 -> 128x128
        transforms.Lambda(lambda t: t.double() / 255.0 if t.max() > 1 else t.double()) # Make the values real [0, 1]
    ])

    train_data = MMCelebAHQ(args, transform=transform)
    train_loader = DataLoader(train_data, batch_size=args.mb_size, shuffle=args.shuffle, pin_memory=False, num_workers=0)

    args.start_iter = args.N + 1 # Skip the samples from the training set
    test_data = MMCelebAHQ(args, transform=transform)
    test_loader = DataLoader(test_data, batch_size=args.mb_size, shuffle=False, pin_memory=False, num_workers=0, drop_last=True)

    _, c, x, y = next(iter(train_loader))[0].size()

    return train_loader, test_loader, c * x * y, c

def mmcelebahq_final_compute(args, net1, net2, net3, kPCA, device=torch.device('cuda')):
    """ Function to compute embeddings of full dataset. """
    args.shuffle = False
    xt, _, _, _= get_mmcelebahq_dataloader(args=args)  # loading data without shuffle

    xtr = net1(torch.stack(xt.dataset.images)[:args.N, :, :, :].to(args.device))
    ytr = net2(torch.stack(xt.dataset.sketches)[:args.N, :, :, :].to(args.device))
    ztr = net3(torch.stack(xt.dataset.labels)[:args.N, :].to(args.device))

    h, s = kPCA(xtr, ytr, ztr)
    return torch.mm(torch.t(xtr), h), torch.mm(torch.t(ytr), h), torch.mm(torch.t(ztr), h), h, s


def get_classes():
    return ('_o_Clock_Shadow','Arched_Eyebrows','Attractive','Bags_Under_Eyes','Bald','Bangs','Big_Lips','Big_Nose','Black_Hair','Blond_Hair','Blurry','Brown_Hair','Bushy_Eyebrows','Chubby','Double_Chin','Eyeglasses','Goatee','Gray_Hair','Heavy_Makeup','High_Cheekbones','Male','Mouth_Slightly_Open','Mustache','Narrow_Eyes','No_Beard','Oval_Face','Pale_Skin','Pointy_Nose','Receding_Hairline','Rosy_Cheeks','Sideburns','Smiling ','Straight_Hair','Wavy_Hair','Wearing_Earrings','Wearing_Hat','Wearing_Lipstick','Wearing_Necklace','Wearing_Necktie','Young')

def get_labels_df():
    import pandas as pd
    classes = get_classes()
    labels_df = pd.DataFrame(columns=classes)

    for i in range(1, 40 + 1):
        label = np.array(open("../../Datasets/MMCelebAHQ/DATASET/label/" + str(i) + '.txt').read().split(',')).astype(np.double)
        label[label == -1] = 0
        labels_df.loc[i] = label

    return labels_df
