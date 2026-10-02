import argparse
import codecs

import torch

from tiny_talk.config import GPTConfig, get_device
from tiny_talk.model import GPT
from tiny_talk.tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser(
        description="Sample text from a trained checkpoint."
    )
    parser.add_argument(
        "prompt", nargs="?", default="", help="text to start generation from"
    )
    parser.add_argument("--checkpoint", default="checkpoints/model.pt")
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument(
        "--top-k", type=int, default=50, help="0 samples from the full vocab"
    )
    args = parser.parse_args()
    if args.temperature <= 0:
        parser.error("--temperature must be > 0")
    if args.top_k < 0:
        parser.error("--top-k must be >= 0")

    device = get_device()
    ckpt = torch.load(args.checkpoint, map_location=device)
    tokenizer = BPETokenizer()
    tokenizer.merges = dict(ckpt["merges"])
    model = GPT(GPTConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    # Training docs start with <|endoftext|>, so the prompt follows one.
    # The prompt comes from the person running the CLI, so special tokens
    # like <|user_start|> are allowed.
    eot = tokenizer.special_tokens["<|endoftext|>"]
    ids = [eot, *tokenizer.encode(args.prompt, allow_special=True)]
    context = torch.tensor([ids], dtype=torch.long, device=device)

    # A token can end mid UTF-8 character, so decode the byte stream
    # incrementally instead of decoding each token on its own.
    utf8 = codecs.getincrementaldecoder("utf-8")(errors="replace")
    print(args.prompt, end="", flush=True)
    for tok in model.generate(
        context,
        args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k or None,
    ):
        tok = tok.item()
        if tok == eot:
            break
        print(utf8.decode(tokenizer.decode_bytes([tok])), end="", flush=True)
    print(utf8.decode(b"", final=True))


if __name__ == "__main__":
    main()
