import pytest

from tiny_talk.config import GPTConfig, TrainConfig
from tiny_talk.model import GPT
from tiny_talk.train import configure_optimizer, lr_factor


def test_lr_factor_warms_up_then_decays_to_floor():
    tcfg = TrainConfig(max_iters=1000, warmup_iters=100, min_lr_ratio=0.1)
    assert lr_factor(0, tcfg) == pytest.approx(0.01)
    assert lr_factor(99, tcfg) == pytest.approx(1.0)
    assert lr_factor(100, tcfg) == pytest.approx(1.0)
    assert lr_factor(550, tcfg) == pytest.approx(0.55)  # cosine midpoint
    assert lr_factor(1000, tcfg) == pytest.approx(0.1)


def test_lr_factor_without_warmup():
    tcfg = TrainConfig(max_iters=10, warmup_iters=0, min_lr_ratio=0.1)
    assert lr_factor(0, tcfg) == pytest.approx(1.0)
    assert lr_factor(10, tcfg) == pytest.approx(0.1)


CFG = GPTConfig(
    vocab_size=11, block_size=8, n_embed=16, n_head=2, n_layer=2, dropout=0.0
)
TCFG = TrainConfig(
    batch_size=2, max_iters=4, eval_interval=2, eval_iters=1, warmup_iters=1
)


def test_configure_optimizer_decays_only_matrices():
    model = GPT(CFG)
    decay, no_decay = configure_optimizer(model, TCFG).param_groups
    assert decay["weight_decay"] == TCFG.weight_decay
    assert no_decay["weight_decay"] == 0.0
    assert all(p.dim() >= 2 for p in decay["params"])
    assert all(p.dim() < 2 for p in no_decay["params"])
    assert len(decay["params"]) + len(no_decay["params"]) == len(
        list(model.parameters())
    )
