import torch

from tiny_talk.config import GPTConfig
from tiny_talk.model import GPT

CFG = GPTConfig(
    vocab_size=11, block_size=8, n_embed=16, n_head=2, n_layer=2, dropout=0.0
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
    assert abs(block.ffwd.net[0].weight.std().item() - 0.02) < 1e-3
    assert abs(model.token_embedding_table.weight.std().item() - 0.02) < 1e-3
    resid_std = 0.02 / (2 * cfg.n_layer) ** 0.5
    for w in (block.sa.proj.weight, block.ffwd.net[2].weight):
        assert abs(w.std().item() - resid_std) < 1e-3
    assert torch.all(block.ffwd.net[0].bias == 0)


def test_output_projection_tied_to_embedding():
    model = GPT(CFG)
    assert model.lm_head.weight is model.token_embedding_table.weight
