import random
import torch
import tiktoken

enc = tiktoken.get_encoding("o200k_base")
PAD = enc.eot_token

class O200SemanticTokenAugmenter:
    def __init__(self, max_len=15, pad_token=PAD, n_min=1, n_max=3, seed=None):
        self.enc = tiktoken.get_encoding("o200k_base")
        self.max_len = max_len
        self.pad_token = pad_token
        self.n_min = n_min
        self.n_max = n_max

        if seed is not None:
            random.seed(seed)

    def augment_ids(self, token_ids):
        if torch.is_tensor(token_ids):
            device = token_ids.device
            out = token_ids.clone().detach().long()

        else:
            device = None
            out = torch.tensor(token_ids, dtype=torch.long)

        valid_positions = [i for i, tok in enumerate(out.tolist()) if int(tok) != self.pad_token]

        if len(valid_positions) == 0:
            return out.to(device) if device is not None else out

        n_changes = random.randint(self.n_min, self.n_max)
        n_changes = min(n_changes, len(valid_positions))

        positions = random.sample(valid_positions, n_changes)

        for pos in positions:
            old_id = int(out[pos])
            new_id = old_id

            while new_id == old_id or new_id == self.pad_token:
                new_id = random.randint(0, self.enc.n_vocab - 1)

            out[pos] = new_id

        # Keep fixed length
        if out.numel() < self.max_len:
            pad = torch.full(
                (self.max_len - out.numel(),),
                self.pad_token,
                dtype=torch.long,
                device=out.device,
            )
            out = torch.cat([out, pad], dim=0)
        else:
            out = out[:self.max_len]

        return out.to(device) if device is not None else out

    def augment_batch(self, batch_token_ids):
        return torch.stack([self.augment_ids(row) for row in batch_token_ids], dim=0)