import argparse
import os
from dataclasses import asdict

import torch

from tiny_talk.config import GPTConfig, TrainConfig, get_device
from tiny_talk.data import get_batch, load_text, train_val_split
from tiny_talk.model import GPT
from tiny_talk.tokenizer import BPETokenizer


@torch.no_grad()
def estimate_loss(model, splits, tcfg: TrainConfig, device):
    out = {}
    model.eval()
    for name, data in splits.items():
        losses = torch.zeros(tcfg.eval_iters)
        for k in range(tcfg.eval_iters):
            X, Y = get_batch(data, tcfg.batch_size, model.cfg.block_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[name] = losses.mean().item()
    model.train()
    return out


def train(text, gcfg: GPTConfig, tcfg: TrainConfig, device):
    tokenizer = BPETokenizer()
    tokenizer.train(text, gcfg.vocab_size)
    # Training stops early if the text runs out of pairs to merge.
    gcfg.vocab_size = tokenizer.vocab_size

    data = torch.tensor(tokenizer.encode(text, allow_special=True), dtype=torch.long)
    train_data, val_data = train_val_split(data)
    splits = {"train": train_data, "val": val_data}

    model = GPT(gcfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.learning_rate)

    for i in range(tcfg.max_iters + 1):
        if i % tcfg.eval_interval == 0 or i == tcfg.max_iters:
            losses = estimate_loss(model, splits, tcfg, device)
            print(
                f"Step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
        if i == tcfg.max_iters:
            break

        xb, yb = get_batch(train_data, tcfg.batch_size, gcfg.block_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Train the GPT.")
    parser.add_argument("--data", default="data/input.txt")
    parser.add_argument("--out", default="checkpoints/model.pt")
    parser.add_argument("--max-iters", type=int, default=TrainConfig.max_iters)
    parser.add_argument("--eval-interval", type=int, default=TrainConfig.eval_interval)
    parser.add_argument("--vocab-size", type=int, default=GPTConfig.vocab_size)
    args = parser.parse_args()

    tcfg = TrainConfig(max_iters=args.max_iters, eval_interval=args.eval_interval)
    gcfg = GPTConfig(vocab_size=args.vocab_size)
    torch.manual_seed(tcfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    model, tokenizer = train(load_text(args.data), gcfg, tcfg, device)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "config": asdict(gcfg),
            "merges": list(tokenizer.merges.items()),
        },
        args.out,
    )
    print(f"Saved checkpoint to {args.out}")


if __name__ == "__main__":
    main()
