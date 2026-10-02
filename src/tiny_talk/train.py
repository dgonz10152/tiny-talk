import argparse
import dataclasses
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


def best_path(out):
    return os.path.splitext(out)[0] + "_best.pt"


def save_checkpoint(path, state):
    # Write then rename, so an interrupted save never clobbers the last good file.
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(state, path + ".tmp")
    os.replace(path + ".tmp", path)


def train(splits, gcfg: GPTConfig, tcfg: TrainConfig, device, out, merges, resume=None):
    """Train, checkpointing to out (and out's _best.pt on val improvement) each eval.

    Args:
        resume: A checkpoint dict from a previous run to continue from
    """
    train_data = splits["train"]
    model = GPT(gcfg).to(device)
    optimizer = configure_optimizer(model, tcfg)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, functools.partial(lr_factor, tcfg=tcfg)
    )
    start, best_val = 0, float("inf")
    if resume is not None:
        model.load_state_dict(resume["model"])
        optimizer.load_state_dict(resume["optimizer"])
        scheduler.load_state_dict(resume["scheduler"])
        torch.set_rng_state(resume["rng"].cpu())
        start, best_val = resume["step"], resume["best_val"]

    for i in range(start, tcfg.max_iters + 1):
        is_eval = i % tcfg.eval_interval == 0 or i == tcfg.max_iters
        # A resumed run already evaluated and saved its starting step.
        if is_eval and not (resume is not None and i == start):
            losses = estimate_loss(model, splits, tcfg, device)
            improved = losses["val"] < best_val
            best_val = min(best_val, losses["val"])
            print(
                f"Step {i}: train loss {losses['train']:.4f}, "
                f"val loss {losses['val']:.4f}, lr {scheduler.get_last_lr()[0]:.2e}"
                + (" (best)" if improved else "")
            )
            state = {
                "model": model.state_dict(),
                "config": asdict(gcfg),
                "merges": merges,
                "train_config": asdict(tcfg),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "step": i,
                "best_val": best_val,
                # Taken after eval so a resumed run draws the same batches.
                "rng": torch.get_rng_state(),
            }
            save_checkpoint(out, state)
            if improved:
                save_checkpoint(best_path(out), state)
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
    parser.add_argument("--resume", help="checkpoint to continue training from")
    parser.add_argument("--max-iters", type=int)
    parser.add_argument("--eval-interval", type=int)
    args = parser.parse_args()

    tokenizer = BPETokenizer.load(os.path.join(args.data, "merges.json"))
    resume = None
    if args.resume:
        resume = torch.load(args.resume, map_location="cpu")
        if "optimizer" not in resume:
            parser.error(f"{args.resume} has no optimizer state, so can't resume")
        if dict(resume["merges"]) != tokenizer.merges:
            parser.error(f"{args.resume} was trained with a different merges.json")
        gcfg = GPTConfig(**resume["config"])
        tcfg = TrainConfig(**resume["train_config"])
    else:
        gcfg = GPTConfig(vocab_size=tokenizer.vocab_size)
        tcfg = TrainConfig()
    overrides = {"max_iters": args.max_iters, "eval_interval": args.eval_interval}
    tcfg = dataclasses.replace(
        tcfg, **{k: v for k, v in overrides.items() if v is not None}
    )

    torch.manual_seed(tcfg.seed)
    device = get_device()
    print(f"Using device: {device}")
    if resume is not None:
        print(f"Resuming from {args.resume} at step {resume['step']}")

    splits = {
        name: load_tokens(os.path.join(args.data, f"{name}.bin"))
        for name in ("train", "val")
    }
    merges = list(tokenizer.merges.items())
    train(splits, gcfg, tcfg, device, args.out, merges, resume)
    print(f"Checkpoints: {args.out} (latest), {best_path(args.out)} (best val)")


if __name__ == "__main__":
    main()
