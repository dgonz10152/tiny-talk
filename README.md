# tiny-talk

A tiny GPT trained on TinyStories with a byte-level BPE tokenizer, built with PyTorch.

## Setup

```sh
uv sync
```

## Usage

```sh
uv run tiny-talk-prepare     # streams 250k TinyStories stories, trains the BPE tokenizer, writes data/tinystories/{train,val}.bin + merges.json
uv run tiny-talk-train       # trains on data/tinystories; checkpoints every eval to checkpoints/model.pt (+ model_best.pt)
uv run tiny-talk-train --resume checkpoints/model.pt   # continues an interrupted run
uv run tiny-talk-generate "Once upon a time"   # streams a story until <|endoftext|> (--temperature, --top-k)
uv run pytest
```

## Layout

- `src/tiny_talk/config.py` - model and training hyperparameters
- `src/tiny_talk/tokenizer.py` - BPE tokenizer with chat special tokens (`<|endoftext|>`, `<|system_start|>`/`<|system_end|>`, `<|user_start|>`/`<|user_end|>`, `<|assistant_start|>`/`<|assistant_end|>`)
- `src/tiny_talk/prepare.py` - downloads the dataset, trains the tokenizer, tokenizes
- `src/tiny_talk/model.py` - GPT transformer
- `src/tiny_talk/data.py` - token loading and batching
- `src/tiny_talk/train.py` / `generate.py` - CLI entry points
