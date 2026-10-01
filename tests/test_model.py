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
    out = model.generate(torch.zeros((1, 1), dtype=torch.long), 20)
    assert out.shape == (1, 21)
