import argparse
import os
import sys
from itertools import islice
from pathlib import Path

import numpy as np
from datasets import load_dataset

from tiny_talk.config import GPTConfig
from tiny_talk.tokenizer import BPETokenizer


def tokenize_docs(texts, tokenizer):
    """Tokenize documents, each prefixed with <|endoftext|>, into one uint16 array."""
    eot = tokenizer.special_tokens["<|endoftext|>"]
    # Dataset text is untrusted, so special token strings in it stay plain text.
    ids = [tok for text in texts for tok in (eot, *tokenizer.encode(text))]
    return np.array(ids, dtype=np.uint16)


def main():
    parser = argparse.ArgumentParser(
        description="Train the BPE tokenizer on TinyStories and write "
        "train.bin, val.bin and merges.json."
    )
    parser.add_argument("--num-docs", type=int, default=250_000)
    parser.add_argument(
        "--num-val-docs", type=int, default=None, help="default: whole split"
    )
    parser.add_argument("--vocab-size", type=int, default=GPTConfig.vocab_size)
    parser.add_argument("--out-dir", default="data/tinystories")
    args = parser.parse_args()
    if args.vocab_size > 2**16:
        parser.error("--vocab-size must fit in uint16 (<= 65536)")

    splits = {}
    for name, hf_split, limit in [
        ("train", "train", args.num_docs),
        ("val", "validation", args.num_val_docs),
    ]:
        ds = load_dataset("roneneldan/TinyStories", split=hf_split, streaming=True)
        splits[name] = [row["text"] for row in islice(ds, limit)]
        if not splits[name]:
            parser.error(f"{name} split is empty")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = BPETokenizer()
    # Joining on <|endoftext|> keeps merges from spanning documents.
    tokenizer.train("<|endoftext|>".join(splits["train"]), args.vocab_size)
    tokenizer.save(out_dir / "merges.json")
    print(f"tokenizer: vocab size {tokenizer.vocab_size}")

    for name, split in splits.items():
        ids = tokenize_docs(split, tokenizer)
        ids.tofile(out_dir / f"{name}.bin")
        print(f"{name}: {len(split):,} docs, {len(ids):,} tokens")

    # ponytail: the abandoned streaming read deadlocks pyarrow's thread pool
    # teardown at exit, so skip teardown once everything is written.
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
