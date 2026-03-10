import torch
import tiktoken

import numpy as np
import torch.nn.functional as F


from PIL import Image
from torchvision import datasets, transforms
from sklearn.preprocessing import OneHotEncoder
from torch.utils.data import DataLoader, Dataset

class Poem(Dataset):
    def __init__(self, *args, **kwargs):
        dataset_source = "../../Datasets/Poem/"
        txt_encoder = tiktoken.get_encoding("o200k_base") # Text tokenizer
        PAD = txt_encoder.eot_token

        stm_encoder = OneHotEncoder(categories=[[-1, 0, 1, 2]], sparse_output=False, dtype=int)
        stm_encoder.fit(np.array([[-1],[0],[1],[2]]))

        len_encoder = OneHotEncoder(categories=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]], sparse_output=False, dtype=int)
        len_encoder.fit(np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10],[11],[12],[13],[14],[15]]))

        self.text = []
        self.sentiment = []
        self.length = []

        for i in range(args[0].N):
            text = open(dataset_source + 'TEXT/' + str(i) + '.txt').read()
            sentiment = int(open(dataset_source + 'SENTIMENT/' + str(i) + '.txt').read())

            txt_enc = np.array(txt_encoder.encode(text), dtype=np.float32)

            if txt_enc.size < 16:
                txt_enc = np.pad(txt_enc, (0, 15-txt_enc.size), constant_values=PAD)
                txt_len = len(txt_enc)
            else:
                txt_enc = txt_enc[:15]
                txt_len = 15

            txt_enc = txt_enc

            stm_enc = stm_encoder.transform(np.array([[sentiment]])).astype(np.float32)
            stm_enc = stm_enc.squeeze(0)

            len_enc = len_encoder.transform(np.array([[txt_len]])).astype(np.float32)
            len_enc = len_enc.squeeze(0)

            self.text.append(txt_enc)
            self.sentiment.append(stm_enc)
            self.length.append(len_enc)

    def __getitem__(self, i):
        text = self.text[i]
        sentiment = self.sentiment[i]
        length = self.length[i]

        return text, sentiment, length

    def __len__(self):
        return len(self.text)

def get_poem_dataloader(args):
    train_data = Poem(args)
    train_loader = DataLoader(train_data, batch_size=args.mb_size, shuffle=args.shuffle, pin_memory=False, num_workers=0)
    _, c = next(iter(train_loader))[0].size()

    return train_loader, 15, 4

def poem_final_compute(args, net1, net2, net3, kPCA, device=torch.device('cuda')):
    """ Function to compute embeddings of full dataset. """
    print(args.N)
    args.shuffle = False
    xt, _, _ = get_poem_dataloader(args=args)  # loading data without shuffle
    xtr = net1(torch.stack([torch.as_tensor(t, dtype=torch.float32) for t in xt.dataset.text])[:args.N, :].to(args.device))
    ytr = net2(torch.stack([torch.as_tensor(s, dtype=torch.float32) for s in xt.dataset.sentiment])[:args.N, :].to(args.device))
    ztr = net3(torch.stack([torch.as_tensor(l, dtype=torch.float32) for l in xt.dataset.length])[:args.N, :].to(args.device))

    h, s = kPCA(xtr, ytr, ztr)
    return torch.mm(torch.t(xtr), h), torch.mm(torch.t(ytr), h), torch.mm(torch.t(ztr), h), h, s