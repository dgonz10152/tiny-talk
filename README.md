# tiny-talk

A simple, from-scratch project for training a small language model on short stories.
It trains a ~15M parameter GPT on [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories), a dataset of simple children's stories, using a byte-level BPE tokenizer written by hand.
Everything is plain PyTorch and small enough to read in one sitting and train on a laptop (CUDA, Apple MPS, or CPU).

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

## Architecture

The pipeline has three stages, each a CLI entry point: `prepare` tokenizes TinyStories into `train.bin` / `val.bin`, `train` fits the GPT below, and `generate` samples stories from a checkpoint.

```
Input tokens
    |
[Token Embedding + Learned Position Embedding]
    |
[Transformer Block x8:]
    |--- LayerNorm
    |--- Causal Multi-Headed Attention (6 heads)
    |    |--- Scaled Dot Product Attention
    |    |--- Output Projection
    |--- Residual Add
    |--- LayerNorm
    |--- Feed Forward (384 -> 1536 -> 384)
    |    |--- ReLU activation function
    |--- Residual Add
[Final LayerNorm]
    |
[Output Projection (untied)]
    |
Next token logits
```

### Tokenizer

- Byte-level BPE using the GPT-4 regex split pattern, so any UTF-8 text round-trips.
- Trained on the training split to a 1024 token vocabulary: 256 bytes, the learned merges, then the special tokens.
- `<|endoftext|>` separates stories. Chat tokens (`<|system_start|>`, `<|user_start|>`, `<|assistant_start|>` and their `_end` pairs) are reserved for future chat fine-tuning.
- Special token strings in dataset text are encoded as plain text, so data cannot inject control tokens.

### Model

A decoder-only transformer in the style of GPT-2 (`config.py` holds the defaults):

| Setting | Value |
| --- | --- |
| Layers | 8 |
| Heads | 6 |
| Embedding size | 384 |
| Context length | 512 tokens |
| Vocabulary | 1024 |
| Parameters | ~15M |

- GPT-2 weight init, with residual output projections scaled by `1/sqrt(2 * n_layer)`.

### Training

- Random 512 token windows sampled from the memory-mapped `train.bin`.
- AdamW with weight decay on 2D weights only, gradient clipping at norm 1.0.
- Linear warmup, then cosine decay to 10% of the peak learning rate.
- Every eval writes a full checkpoint (model, optimizer, scheduler, RNG state), so runs resume exactly with `--resume`. The best validation loss is also saved to `model_best.pt`.

### Generation

Loads a checkpoint, seeds the context with `<|endoftext|>` plus your prompt, and samples with temperature and top-k until the model emits `<|endoftext|>`.

## Layout

- `src/tiny_talk/config.py` - model and training hyperparameters
- `src/tiny_talk/tokenizer.py` - byte-level BPE tokenizer with special tokens
- `src/tiny_talk/prepare.py` - downloads the dataset, trains the tokenizer, tokenizes
- `src/tiny_talk/model.py` - GPT transformer
- `src/tiny_talk/data.py` - token loading and batching
- `src/tiny_talk/train.py` / `generate.py` - CLI entry points
- `tests/` - pytest suite for the tokenizer, data prep, model and training loop
