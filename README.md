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
[Token Embedding]  (no position embedding, RoPE handles position)
    |
[Transformer Block x8:]
    |--- LayerNorm
    |--- Multi-head Latent Attention (6 heads)
    |    |--- Compress x -> latent c_kv (128) + shared RoPE key (32)   <- cached
    |    |--- Expand latent -> per-head keys and values
    |    |--- Lightning indexer picks the top 128 past tokens per query
    |    |--- Causal attention over the picked tokens only
    |    |--- Output Projection
    |--- Residual Add
    |--- LayerNorm
    |--- SwiGLU Feed Forward (384 -> 2 x 1024 -> 384)
    |--- Residual Add
[Final LayerNorm]
    |
[Output Projection (weight-tied)]
    |
Next token logits
```

### Tokenizer

- Byte-level BPE using the GPT-4 regex split pattern, so any UTF-8 text round-trips.
- Trained on the training split to a 1024 token vocabulary: 256 bytes, the learned merges, then the special tokens.
- `<|endoftext|>` separates stories. Chat tokens (`<|system_start|>`, `<|user_start|>`, `<|assistant_start|>` and their `_end` pairs) are reserved for future chat fine-tuning.
- Special token strings in dataset text are encoded as plain text, so data cannot inject control tokens.

### Model

A decoder-only transformer that starts from GPT-2 and swaps in the attention and MLP designs used by DeepSeek V3.2 (`config.py` holds the defaults):

| Setting | Value |
| --- | --- |
| Layers | 8 |
| Heads | 6 |
| Embedding size | 384 |
| Context length | 512 tokens |
| Vocabulary | 1024 |
| KV latent rank | 128 |
| RoPE key dims | 32 |
| Indexer | 4 heads x 32 dims, top 128 |
| Parameters | ~15M |

- GPT-2 weight init, with residual output projections scaled by `1/sqrt(2 * n_layer)`.

#### RoPE (rotary position embeddings)

GPT-2 adds a learned vector per position to each token, which mixes position into everything the token carries and caps context at the table size.
RoPE instead rotates query and key vectors in 2D pairs, by an angle of `position * theta^(-2i/d)` for pair `i`.
Early pairs spin fast and capture local order, late pairs spin slowly and capture long range.
A dot product between two rotated vectors depends only on the angle between them, so attention scores depend on relative distance `m - n`, not absolute position.
Values are never rotated, so the information a token passes along is free of position noise.
A cached key is rotated once at its own position and stays valid as generation continues.

#### SwiGLU feed forward

The MLP computes `w2(silu(w1 x) * w3 x)`.
`silu(w1 x)` acts as a smooth, learned gate that scales each hidden unit of `w3 x`, so the layer can multiply features together instead of only thresholding them like GELU.
The hidden size is `8/3 * 384 = 1024`, so the three matrices use the same parameters as the old `384 -> 1536 -> 384` MLP.

#### Multi-head Latent Attention (MLA)

Standard attention caches a full key and value per head for every past token, which is what limits context length and batch size at inference.
MLA compresses each token into a 128-dim latent `c_kv`, and a shared up-projection rebuilds each head's 64-dim key and value from it when they are needed.
Only the latent is cached, so each head still gets its own keys, which grouped-query attention gives up.
RoPE cannot be applied inside the latent: a position-dependent rotation would sit between the two projections and prevent folding them together at inference.
So position goes in a separate 32-dim RoPE key, shared by all heads, matched by a 32-dim RoPE part of each query ("decoupled RoPE").
Per token per layer, the cache holds 128 + 32 latent/RoPE values plus a 32-dim indexer key, which is 192 floats instead of 768 for standard attention (4x smaller).
`generate()` uses this cache, so each new token runs the model on one position instead of the whole window.

#### Sparse top-k attention (DeepSeek Sparse Attention)

A small "lightning indexer" scores every past token for each query: `I[t, s] = sum_j w[t, j] * relu(q_j[t] . k[s])`, with 4 small query heads, one shared key and per-head weights.
Each query then attends only to its 128 highest-scoring past tokens; with 128 or fewer tokens of context this is the same as dense attention.
Picking the top 128 cannot be differentiated, so the language model loss cannot train the indexer.
Instead the indexer reads a detached copy of the input and gets its own KL loss toward the dense attention distribution (averaged over heads), so it learns which tokens dense attention would have looked at.
The aux loss is added only in training mode and touches only indexer weights, so the reported validation loss is pure language modelling loss.
At 512 tokens this is a masked dense computation, so it shows the mechanism rather than saving compute; real savings need a kernel that gathers only the picked keys.

### Training

- Random 512 token windows sampled from the memory-mapped `train.bin`.
- AdamW with weight decay on 2D weights only, gradient clipping at norm 1.0.
- Linear warmup, then cosine decay to 10% of the peak learning rate.
- Every eval writes a full checkpoint (model, optimizer, scheduler, RNG state), so runs resume exactly with `--resume`. The best validation loss is also saved to `model_best.pt`.

### Generation

Loads a checkpoint, seeds the context with `<|endoftext|>` plus your prompt, and samples with temperature and top-k until the model emits `<|endoftext|>`.
Sampling keeps the MLA latent cache, so each step only processes the newest token.
When the cache reaches 512 tokens, it is rebuilt from the last 256.

## Layout

- `src/tiny_talk/config.py` - model and training hyperparameters
- `src/tiny_talk/tokenizer.py` - byte-level BPE tokenizer with special tokens
- `src/tiny_talk/prepare.py` - downloads the dataset, trains the tokenizer, tokenizes
- `src/tiny_talk/model.py` - transformer with RoPE, MLA, sparse attention and SwiGLU
- `src/tiny_talk/data.py` - token loading and batching
- `src/tiny_talk/train.py` / `generate.py` - CLI entry points
- `tests/` - pytest suite for the tokenizer, data prep, model and training loop
