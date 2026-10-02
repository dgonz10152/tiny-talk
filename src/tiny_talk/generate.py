import argparse

import torch

from tiny_talk.config import GPTConfig, get_device
from tiny_talk.model import GPT
from tiny_talk.tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Sample text from a trained checkpoint."
    )
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--max-new-tokens", type=int, default=1000)
    args = parser.parse_args()

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    tokenizer = BPETokenizer()
    tokenizer.merges = dict(ckpt["merges"])
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Training docs start with <|endoftext|>, so seed with it.
    eot = tokenizer.special_tokens["<|endoftext|>"]
    context = torch.tensor([[eot]], dtype=torch.long, device=device)
    print(tokenizer.decode(model.generate(context, args.max_new_tokens)[0].tolist()))


if __name__ == "__main__":
    main()
