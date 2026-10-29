"""Causality: the prediction at position t must depend only on tokens 0..t.

Two independent checks — a perturbation test and a Jacobian (gradient) test — on every
positional-encoding variant, plus a negative control proving the tests can fail.
"""

import pytest
import torch

from gpt.config import ModelConfig
from gpt.model import GPT


def make(pos="learned", causal=True, seed=0):
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=64, block_size=16, n_layer=2, n_head=2, n_embd=32, dropout=0.0,
                      pos_encoding=pos, causal=causal)
    return GPT(cfg).eval()


@pytest.mark.parametrize("pos", ["learned", "sinusoidal", "none"])
def test_future_tokens_do_not_change_past_logits(pos):
    model = make(pos)
    torch.manual_seed(1)
    idx = torch.randint(0, 64, (2, 12))
    base, _ = model(idx)
    for t in range(12):
        changed = idx.clone()
        changed[:, t] = (changed[:, t] + 7) % 64
        out, _ = model(changed)
        torch.testing.assert_close(out[:, :t], base[:, :t], rtol=0, atol=1e-6)
        assert not torch.allclose(out[:, t], base[:, t]), "position t must see its own token"


@pytest.mark.parametrize("pos", ["learned", "sinusoidal", "none"])
def test_jacobian_is_lower_triangular(pos):
    """d logits[t] / d embedding[s] == 0 for every s > t."""
    model = make(pos)
    T = 8
    idx = torch.randint(0, 64, (1, T))
    emb = model.tok_emb(idx).detach().requires_grad_(True)

    def forward_from_embeddings(e):
        x = e
        if model.cfg.pos_encoding == "learned":
            x = x + model.pos_emb(torch.arange(T))
        elif model.cfg.pos_encoding == "sinusoidal":
            x = x + model.pos_table[:T]
        for blk in model.blocks:
            x = blk(x)
        return model.lm_head(model.ln_f(x))

    logits = forward_from_embeddings(emb)
    for t in range(T):
        (g,) = torch.autograd.grad(logits[0, t].sum(), emb, retain_graph=True)
        influence = g[0].abs().sum(-1)                     # (T,) influence of each input position
        assert torch.all(influence[t + 1:] == 0), f"position {t} sees the future: {influence}"
        assert torch.all(influence[: t + 1] > 0)


def test_negative_control_without_mask_leaks_future():
    """With the causal mask removed the perturbation test MUST fail — the test has teeth."""
    model = make(causal=False)
    idx = torch.randint(0, 64, (1, 10))
    base, _ = model(idx)
    changed = idx.clone()
    changed[:, -1] = (changed[:, -1] + 1) % 64
    out, _ = model(changed)
    assert not torch.allclose(out[:, 0], base[:, 0], atol=1e-6)


def test_generation_prefix_consistency():
    """Logits for a prefix are identical whether or not more tokens follow it."""
    model = make()
    idx = torch.randint(0, 64, (1, 16))
    full, _ = model(idx)
    for t in (1, 5, 11):
        part, _ = model(idx[:, :t])
        torch.testing.assert_close(part, full[:, :t], rtol=0, atol=1e-5)
