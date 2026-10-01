import torch


def load_text(path):
    with open(path, encoding="utf-8") as f:
        return f.read()


def train_val_split(data, train_frac=0.9):
    n = int(train_frac * len(data))
    return data[:n], data[n:]


def get_batch(data, batch_size, block_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)
