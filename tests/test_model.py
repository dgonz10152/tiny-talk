import torch
from torch.nn import functional as F

from tiny_talk.config import GPTConfig
from tiny_talk.model import GPT

# index_topk below block_size so sparse top-k attention is exercised.
CFG = GPTConfig(
    vocab_size=11,
    block_size=8,
    n_embed=16,
    n_head=2,
    n_layer=2,
    dropout=0.0,
    kv_lora_rank=8,
    qk_rope_dim=4,
    index_n_heads=2,
    index_head_dim=4,
    index_topk=4,
)


def test_forward_shapes_and_loss():
    model = GPT(CFG)
    idx = torch.randint(CFG.vocab_size, (3, CFG.block_size))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, CFG.block_size, CFG.vocab_size)
    assert loss.ndim == 0


def test_attention_is_causal():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    idx = torch.randint(CFG.vocab_size, (1, CFG.block_size))
    changed = idx.clone()
    changed[0, -1] = (changed[0, -1] + 1) % CFG.vocab_size
    a, _ = model(idx)
    b, _ = model(changed)
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-6)


def test_generate_length():
    model = GPT(CFG).eval()
    out = list(model.generate(torch.zeros((1, 1), dtype=torch.long), 20))
    assert len(out) == 20
    assert all(tok.shape == (1, 1) for tok in out)


def test_generate_top_k_one_is_greedy():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    idx = torch.randint(CFG.vocab_size, (1, 4))
    logits, _ = model(idx)
    tok = next(model.generate(idx, 1, top_k=1))
    assert tok.item() == logits[0, -1].argmax().item()


def test_generate_top_k_restricts_choices():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    idx = torch.randint(CFG.vocab_size, (1, 4))
    logits, _ = model(idx)
    allowed = set(logits[0, -1].topk(3).indices.tolist())
    for _ in range(50):
        tok = next(model.generate(idx, 1, temperature=5.0, top_k=3))
        assert tok.item() in allowed


def test_gpt2_init():
    cfg = GPTConfig(vocab_size=512, block_size=64, n_embed=256, n_head=4, n_layer=4)
    model = GPT(cfg)
    block = model.blocks[0]
    assert abs(block.ffwd.w1.weight.std().item() - 0.02) < 1e-3
    assert abs(model.token_embedding_table.weight.std().item() - 0.02) < 1e-3
    resid_std = 0.02 / (2 * cfg.n_layer) ** 0.5
    for w in (block.sa.proj.weight, block.ffwd.w2.weight):
        assert abs(w.std().item() - resid_std) < 1e-3
    assert torch.all(block.sa.proj.bias == 0)


def test_output_projection_tied_to_embedding():
    model = GPT(CFG)
    assert model.lm_head.weight is model.token_embedding_table.weight


def test_cached_decoding_matches_full_forward():
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    idx = torch.randint(CFG.vocab_size, (2, CFG.block_size))
    full, _ = model(idx)
    cache = [{} for _ in model.blocks]
    steps = [model(idx[:, :3], cache=cache)[0]]
    steps += [
        model(idx[:, t : t + 1], cache=cache)[0] for t in range(3, CFG.block_size)
    ]
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-5)
    # Cache holds only the latent, the shared RoPE key and the indexer key.
    assert cache[0]["c_kv"].shape == (2, CFG.block_size, CFG.kv_lora_rank)
    assert cache[0]["k_rope"].shape == (2, CFG.block_size, CFG.qk_rope_dim)


def test_sparse_attention_ignores_unpicked_keys(monkeypatch):
    torch.manual_seed(0)
    model = GPT(CFG).eval()
    attn = model.blocks[0].sa
    x = torch.randn(1, CFG.block_size, CFG.n_embed)
    seen = {}
    real_sdpa = F.scaled_dot_product_attention

    def spy(*args, attn_mask, **kwargs):
        seen["mask"] = attn_mask
        return real_sdpa(*args, attn_mask=attn_mask, **kwargs)

    monkeypatch.setattr(F, "scaled_dot_product_attention", spy)
    attn(x)
    mask = seen["mask"][0, 0]
    assert not mask.triu(1).any()  # causal
    assert (mask.sum(-1) <= CFG.index_topk).all()
    assert mask[-1].sum() == CFG.index_topk


def test_indexer_aux_loss_only_trains_indexer():
    torch.manual_seed(0)
    model = GPT(CFG)
    idx = torch.randint(CFG.vocab_size, (2, CFG.block_size))
    model(idx, idx)
    aux = sum(b.sa.aux_loss for b in model.blocks)
    assert aux > 0
    aux.backward()
    for name, p in model.named_parameters():
        has_grad = p.grad is not None and p.grad.abs().sum() > 0
        assert has_grad == (".sa.index_" in name), name

    model.eval()
    model(idx, idx)
    assert all(b.sa.aux_loss is None for b in model.blocks)


def test_generate_past_block_size():
    model = GPT(CFG).eval()
    out = list(
        model.generate(torch.zeros((1, 1), dtype=torch.long), 3 * CFG.block_size)
    )
    assert len(out) == 3 * CFG.block_size
