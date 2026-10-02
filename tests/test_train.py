import pytest

from tiny_talk.config import TrainConfig
from tiny_talk.train import lr_factor


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
