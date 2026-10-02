import argparse
import functools
import math
import os
from dataclasses import asdict

import torch

from tiny_talk.config import GPTConfig, TrainConfig, get_device
from tiny_talk.data import get_batch, load_tokens
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


def lr_factor(step, tcfg: TrainConfig):
    """Linear warmup, then cosine decay to min_lr_ratio, as a multiple of the peak lr."""
    if step < tcfg.warmup_iters:
        return (step + 1) / tcfg.warmup_iters
    decay_steps = max(1, tcfg.max_iters - tcfg.warmup_iters)
    progress = min(1.0, (step - tcfg.warmup_iters) / decay_steps)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return tcfg.min_lr_ratio + (1 - tcfg.min_lr_ratio) * cosine


def configure_optimizer(model, tcfg: TrainConfig):
    """AdamW that decays only 2D weights (matmuls, embeddings), not biases or norms."""
    params = list(model.parameters())
    return torch.optim.AdamW(
        [
            {"params": [p for p in params if p.dim() >= 2]},
            {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0},
        ],
        lr=tcfg.learning_rate,
        weight_decay=tcfg.weight_decay,
    )


def train(splits, gcfg: GPTConfig, tcfg: TrainConfig, device):
    train_data = splits["train"]
    model = GPT(gcfg).to(device)
    optimizer = configure_optimizer(model, tcfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, functools.partial(lr_factor, tcfg=tcfg)
    )

    for i in range(tcfg.max_iters + 1):
        if i % tcfg.eval_interval == 0 or i == tcfg.max_iters:
            losses = estimate_loss(model, splits, tcfg, device)
            print(
                f"Step {i}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, lr {scheduler.get_last_lr()[0]:.2e}"
            )
        if i == tcfg.max_iters:
            break

        xb, yb = get_batch(train_data, tcfg.batch_size, gcfg.block_size, device)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if tcfg.grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        optimizer.step()
        scheduler.step()

    return model


def main():
    parser = argparse.ArgumentParser(description="Train the GPT.")
    parser.add_argument(
        "--data",
        default="data/tinystories",
        help="dir with train.bin, val.bin and merges.json",
    )
    parser.add_argument("--out", default="checkpoints/model.pt")
    parser.add_argument("--max-iters", type=int, default=TrainConfig.max_iters)
    parser.add_argument("--eval-interval", type=int, default=TrainConfig.eval_interval)
    args = parser.parse_args()

    tcfg = TrainConfig(max_iters=args.max_iters, eval_interval=args.eval_interval)
    tokenizer = BPETokenizer.load(os.path.join(args.data, "merges.json"))
    gcfg = GPTConfig(vocab_size=tokenizer.vocab_size)
    torch.manual_seed(tcfg.seed)
    device = get_device()
    print(f"Using device: {device}")

    splits = {
        name: load_tokens(os.path.join(args.data, f"{name}.bin"))
        for name in ("train", "val")
    }
    model = train(splits, gcfg, tcfg, device)

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
