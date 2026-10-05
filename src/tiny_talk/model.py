import math

import torch
from torch import nn
from torch.nn import functional as F

from tiny_talk.config import GPTConfig


def rope_cache(dim, length, theta):
    """cos/sin tables for rotating dim-sized vectors at positions 0..length-1."""
    # Pair i spins at theta^(-2i/dim): early pairs fast (local), late slow (far).
    inv_freq = theta ** (-torch.arange(0, dim, 2).float() / dim)
    angles = torch.outer(torch.arange(length).float(), inv_freq)
    angles = torch.cat((angles, angles), dim=-1)  # (length, dim)
    return angles.cos(), angles.sin()


def apply_rope(x, cos, sin):
    """Rotate each (x_i, x_{i+dim/2}) pair of the last dim by its position angle."""
    x1, x2 = x.chunk(2, dim=-1)
    return x * cos + torch.cat((-x2, x1), dim=-1) * sin


class MLAttention(nn.Module):
    """Multi-head Latent Attention (DeepSeek-V2) with DSA sparse top-k (V3.2).

    Keys and values are rebuilt per head from a small shared latent, so the
    KV cache only stores the latent, one shared RoPE key and the indexer key.
    A lightweight indexer scores past tokens and each query only attends to
    its index_topk best ones.
    """

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embed // cfg.n_head
        self.rope_dim = cfg.qk_rope_dim
        self.kv_rank = cfg.kv_lora_rank
        self.index_n_heads = cfg.index_n_heads
        self.index_topk = cfg.index_topk
        self.dropout_p = cfg.dropout

        # Per head: a content query (matched against the latent's keys) and a
        # RoPE query (matched against the shared positional key).
        self.wq = nn.Linear(
            cfg.n_embed, cfg.n_head * (self.head_dim + self.rope_dim), bias=False
        )
        # Down-projection: what gets cached, latent + shared RoPE key.
        self.wdkv = nn.Linear(cfg.n_embed, self.kv_rank + self.rope_dim, bias=False)
        self.kv_norm = nn.RMSNorm(self.kv_rank)
        # Up-projection: latent -> per-head content keys and values.
        self.wukv = nn.Linear(self.kv_rank, cfg.n_head * 2 * self.head_dim, bias=False)
        self.proj = nn.Linear(cfg.n_embed, cfg.n_embed)
        self.dropout = nn.Dropout(cfg.dropout)

        # Lightning indexer: several small query heads against one shared key.
        self.index_q = nn.Linear(
            cfg.n_embed, cfg.index_n_heads * cfg.index_head_dim, bias=False
        )
        self.index_k = nn.Linear(cfg.n_embed, cfg.index_head_dim, bias=False)
        self.index_k_norm = nn.RMSNorm(cfg.index_head_dim)
        self.index_w = nn.Linear(cfg.n_embed, cfg.index_n_heads, bias=False)

        for name, dim in (("rope", self.rope_dim), ("index", cfg.index_head_dim)):
            cos, sin = rope_cache(dim, cfg.block_size, cfg.rope_theta)
            self.register_buffer(f"{name}_cos", cos, persistent=False)
            self.register_buffer(f"{name}_sin", sin, persistent=False)
        self.aux_loss = None

    def forward(self, x, cache=None):
        """x: (B, T, C). cache: dict filled in place across calls, or None."""
        B, T, C = x.shape
        H, D, R = self.n_head, self.head_dim, self.rope_dim
        pos = cache["c_kv"].size(1) if cache else 0
        rope = self.rope_cos[pos : pos + T], self.rope_sin[pos : pos + T]
        index_rope = self.index_cos[pos : pos + T], self.index_sin[pos : pos + T]

        q = self.wq(x).view(B, T, H, D + R).transpose(1, 2)
        q_nope, q_rope = q.split([D, R], dim=-1)
        q = torch.cat((q_nope, apply_rope(q_rope, *rope)), dim=-1)

        c_kv, k_rope = self.wdkv(x).split([self.kv_rank, R], dim=-1)
        c_kv = self.kv_norm(c_kv)
        k_rope = apply_rope(k_rope, *rope)
        # The indexer trains only on its own aux loss, never through the LM.
        xd = x.detach()
        k_index = apply_rope(self.index_k_norm(self.index_k(xd)), *index_rope)

        if cache is not None:
            for name, t in (("c_kv", c_kv), ("k_rope", k_rope), ("k_index", k_index)):
                cache[name] = torch.cat((cache[name], t), dim=1) if name in cache else t
            c_kv, k_rope, k_index = cache["c_kv"], cache["k_rope"], cache["k_index"]
        S = c_kv.size(1)

        # ponytail: up-projects the whole cached latent every step; absorbing
        # wukv into wq / proj avoids that if decode speed matters.
        k_nope, v = (
            self.wukv(c_kv).view(B, S, H, 2 * D).transpose(1, 2).split(D, dim=-1)
        )
        k = torch.cat((k_nope, k_rope[:, None].expand(B, H, S, R)), dim=-1)

        # Query i sits at absolute position pos + i and sees keys 0..pos + i.
        causal = torch.arange(S, device=x.device) <= torch.arange(
            pos, pos + T, device=x.device
        ).unsqueeze(1)  # (T, S)

        # Index score I[t, s] = sum_j w[t, j] * relu(q_j[t] . k[s]).
        q_index = self.index_q(xd).view(B, T, self.index_n_heads, -1)
        q_index = apply_rope(q_index.transpose(1, 2), *index_rope)  # (B, h, T, d)
        w = self.index_w(xd) * self.index_n_heads**-0.5  # (B, T, h)
        scores = (q_index @ k_index.unsqueeze(1).transpose(-2, -1)).relu()
        scores = scores * q_index.size(-1) ** -0.5
        index = torch.einsum("bht,bhts->bts", w.transpose(1, 2), scores)
        index = index.masked_fill(~causal, float("-inf"))

        mask = causal.expand(B, T, S)
        if S > self.index_topk:
            # ponytail: masks full scores, so no FLOP savings at this size;
            # a gather/sparse kernel is what makes DSA cheap at long context.
            top = index.topk(self.index_topk, dim=-1).indices
            picked = torch.zeros_like(mask).scatter_(-1, top, True)
            mask = mask & picked
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask.unsqueeze(1),
            dropout_p=self.dropout_p if self.training else 0.0,
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        self.aux_loss = None
        if self.training:
            # Teach the indexer to rank keys the way dense attention would:
            # KL(dense attention summed over heads || softmax(index)).
            with torch.no_grad():
                att = (q @ k.transpose(-2, -1)) * q.size(-1) ** -0.5
                att = att.masked_fill(~causal, float("-inf")).softmax(dim=-1)
                target = att.mean(dim=1)  # (B, T, S)
            log_p = index.log_softmax(dim=-1).masked_fill(~causal, 0.0)
            self.aux_loss = (
                (torch.xlogy(target, target) - target * log_p).sum(-1).mean()
            )

        return self.dropout(self.proj(out))


class FeedForward(nn.Module):
    """SwiGLU: silu(w1 x) gates w3 x elementwise before projecting back down."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # 8/3 * n_embed keeps the three matrices at the old 4x MLP's param count.
        hidden = 8 * cfg.n_embed // 3
        self.w1 = nn.Linear(cfg.n_embed, hidden, bias=False)
        self.w3 = nn.Linear(cfg.n_embed, hidden, bias=False)
        self.w2 = nn.Linear(hidden, cfg.n_embed, bias=False)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.sa = MLAttention(cfg)
        self.ffwd = FeedForward(cfg)
        self.ln1 = nn.LayerNorm(cfg.n_embed)
        self.ln2 = nn.LayerNorm(cfg.n_embed)

    def forward(self, x, cache=None):
        x = x + self.sa(self.ln1(x), cache)
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(cfg.vocab_size, cfg.n_embed)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embed)
        self.lm_head = nn.Linear(cfg.n_embed, cfg.vocab_size, bias=False)
        # Weight tying: the input embedding doubles as the output projection.
        self.lm_head.weight = self.token_embedding_table.weight

        # GPT-2 init. Each block adds two residual branches (attention and MLP)
        # into the stream, so their output projections are scaled down by
        # sqrt(2 * n_layer) to keep the stream's variance steady with depth.
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith(("sa.proj.weight", "ffwd.w2.weight")):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * cfg.n_layer))

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None, cache=None):
        """cache: one dict per block, filled in place, for incremental decoding."""
        # No position embedding: RoPE inside attention carries position.
        x = self.token_embedding_table(idx)  # (B, T, C)
        for i, block in enumerate(self.blocks):
            x = block(x, None if cache is None else cache[i])
        x = self.ln_f(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        if targets is None:
            return logits, None

        B, T, C = logits.shape
        loss = F.cross_entropy(logits.view(B * T, C), targets.view(B * T))
        if self.training:
            # Indexer aux losses only reach indexer weights, so the LM is unaffected.
            loss = loss + sum(block.sa.aux_loss for block in self.blocks)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """Yield each newly sampled (B, 1) batch of token ids as it is produced.

        Args:
            idx: (B, T) context token ids
            max_new_tokens: Number of tokens to sample
            temperature: Below 1 sharpens the distribution, above 1 flattens it
            top_k: If set, sample only from the k most likely tokens
        """
        bs = self.cfg.block_size
        cache = None
        for _ in range(max_new_tokens):
            if cache is None or cache[0]["c_kv"].size(1) == bs:
                # ponytail: when the cache fills, re-prefill the last half of the
                # context; a sliding cache avoids the recompute.
                keep = idx[:, -bs:] if cache is None else idx[:, -(bs // 2) :]
                cache = [{} for _ in self.blocks]
                logits, _ = self(keep, cache=cache)
            else:
                logits, _ = self(idx[:, -1:], cache=cache)
            logits = logits[:, -1, :] / temperature
            if top_k is not None:
                kth = torch.topk(logits, min(top_k, logits.size(-1))).values[:, [-1]]
                logits = logits.masked_fill(logits < kth, float("-inf"))
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
            yield idx_next
