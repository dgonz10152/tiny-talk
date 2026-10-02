import numpy as np
import torch


def load_tokens(path):
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy(data[i + 1 : i + block_size + 1].astype(np.int64))
            for i in ix
        ]
    )
    return x.to(device), y.to(device)
