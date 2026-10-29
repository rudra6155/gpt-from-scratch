"""Checkpoint/resume is exact, and the LR schedule has the intended shape."""

import math

import numpy as np
import pytest
import torch

from gpt.config import ModelConfig, TrainConfig
from gpt.train import get_lr, train


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("data")
    rng = np.random.default_rng(0)
    # a learnable stream: repeating pattern + noise, vocab 32
    base = np.tile(np.arange(32, dtype=np.uint16), 400)
    noise = rng.integers(0, 32, size=base.size, dtype=np.uint16)
    stream = np.where(rng.random(base.size) < 0.1, noise, base).astype(np.uint16)
    stream[:11000].tofile(d / "train.bin")
    stream[11000:].tofile(d / "val.bin")
    return d


def cfgs(data_dir, out_dir, **kw):
    m = ModelConfig(vocab_size=32, block_size=16, n_layer=1, n_head=2, n_embd=16, dropout=0.1)
    t = TrainConfig(data_dir=str(data_dir), out_dir=str(out_dir), log_csv=str(out_dir / "log.csv"),
                    batch_size=4, max_steps=12, warmup_steps=3, eval_interval=4, eval_iters=2,
                    device="cpu", dtype="fp32", seed=7, **kw)
    return m, t


def test_resume_is_bit_exact(data_dir, tmp_path):
    full = train(*cfgs(data_dir, tmp_path / "full"))["model"]

    m, t = cfgs(data_dir, tmp_path / "split", stop_step=8)
    assert train(m, t) == {"stopped_at": 8}
    m, t = cfgs(data_dir, tmp_path / "split", resume=True)
    resumed = train(m, t)["model"]

    for (k, a), (_, b) in zip(full.state_dict().items(), resumed.state_dict().items()):
        assert torch.equal(a, b), f"{k} differs after resume"
    rows_full = (tmp_path / "full" / "log.csv").read_text().splitlines()
    rows_split = (tmp_path / "split" / "log.csv").read_text().splitlines()
    strip = lambda rows: [",".join(r.split(",")[:5]) for r in rows]  # ignore timing columns
    assert strip(rows_full) == strip(rows_split)
    assert [r.split(",")[0] for r in rows_full[1:]] == ["0", "4", "8", "12"]


def test_training_reduces_loss(data_dir, tmp_path):
    m, t = cfgs(data_dir, tmp_path / "run")
    t.max_steps, t.lr = 60, 3e-3
    r = train(m, t)
    rows = (tmp_path / "run" / "log.csv").read_text().splitlines()[1:]
    first, last = float(rows[0].split(",")[2]), float(rows[-1].split(",")[2])
    assert last < first - 0.5


def test_lr_schedule_shape():
    t = TrainConfig(lr=1.0, min_lr=0.1, warmup_steps=10, max_steps=110)
    assert get_lr(0, t) == pytest.approx(0.1)
    assert get_lr(9, t) == pytest.approx(1.0)
    assert get_lr(10, t) == pytest.approx(1.0)
    assert get_lr(60, t) == pytest.approx(0.55)          # cosine midpoint
    assert get_lr(110, t) == pytest.approx(0.1)
    lrs = [get_lr(s, t) for s in range(10, 111)]
    assert all(a >= b for a, b in zip(lrs, lrs[1:]))      # monotone decay after warmup
    assert not any(math.isnan(x) for x in lrs)
