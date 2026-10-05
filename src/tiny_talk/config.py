from dataclasses import dataclass

import torch


@dataclass
class GPTConfig:
    vocab_size: int = 1024
    block_size: int = 256
    n_embed: int = 384
    n_head: int = 6
    n_layer: int = 8
    dropout: float = 0
    rope_theta: float = 10000.0
    # MLA: keys/values are rebuilt from a kv_lora_rank latent, plus one shared
    # qk_rope_dim key per token that carries position.
    kv_lora_rank: int = 128
    qk_rope_dim: int = 32
    # Sparse attention: a small indexer picks the index_topk keys each query
    # attends to.
    index_n_heads: int = 4
    index_head_dim: int = 32
    index_topk: int = 128

    def __post_init__(self):
        if self.n_embed % self.n_head:
            raise ValueError(
                f"n_embed ({self.n_embed}) must be divisible by n_head ({self.n_head})"
            )
        for name in ("qk_rope_dim", "index_head_dim"):
            if getattr(self, name) % 2:
                raise ValueError(f"{name} must be even for RoPE")


@dataclass
class TrainConfig:
    batch_size: int = 20
    max_iters: int = 5000
    eval_interval: int = 500
    eval_iters: int = 200
    learning_rate: float = 6e-4
    warmup_iters: int = 200  # linear warmup from ~0 up to learning_rate
    min_lr_ratio: float = 0.1  # cosine decay floor, as a fraction of learning_rate
    weight_decay: float = 0.1  # applied to 2D weights only
    grad_clip: float = 1.0  # max global grad norm; 0 disables
    seed: int = 1337


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"
