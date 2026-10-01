# tiny-talk

A tiny GPT trained on a byte-level BPE tokenizer, built with PyTorch.

## Setup

```sh
uv sync
```

Put training text in `data/` (gitignored), e.g. `data/input.txt` (Tiny Shakespeare) and `data/taylorswift.txt`.

## Usage

```sh
uv run tiny-talk-train                 # trains the tokenizer and model on data/input.txt, saves checkpoints/model.pt
uv run tiny-talk-generate              # samples 1000 tokens from the checkpoint
uv run python -m tiny_talk.tokenizer   # trains the BPE tokenizer on data/taylorswift.txt
uv run pytest
```

## Layout

- `src/tiny_talk/config.py` - model and training hyperparameters
- `src/tiny_talk/tokenizer.py` - BPE tokenizer
- `src/tiny_talk/model.py` - GPT transformer
- `src/tiny_talk/data.py` - text loading and batching
- `src/tiny_talk/train.py` / `generate.py` - CLI entry points
