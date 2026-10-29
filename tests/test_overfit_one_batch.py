"""The same check `make smoke` runs: memorise one batch to loss < 0.1 on CPU in < 60 s."""

from gpt.train import overfit_one_batch


def test_overfit_one_batch():
    r = overfit_one_batch(verbose=False)  # asserts internally; re-checked here for clarity
    assert r["loss"] < 0.1
    assert r["elapsed_s"] < 60
