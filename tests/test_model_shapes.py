import math

import pytest
import torch

from gpt.config import ModelConfig, build_configs
from gpt.eval import evaluate_perplexity, generate, sample_next
from gpt.model import GPT, sinusoidal_table

CFG = dict(vocab_size=100, block_size=16, n_layer=2, n_head=4, n_embd=32, dropout=0.0)


def make(**kw):
    torch.manual_seed(0)
    return GPT(ModelConfig(**{**CFG, **kw}))


@pytest.mark.parametrize("pos", ["learned", "sinusoidal", "none"])
@pytest.mark.parametrize("T", [1, 7, 16])
def test_logits_and_loss_shapes(pos, T):
    model = make(pos_encoding=pos)
    idx = torch.randint(0, 100, (3, T))
    logits, loss = model(idx, idx)
    assert logits.shape == (3, T, 100)
    assert loss.dim() == 0 and torch.isfinite(loss)
    logits, loss = model(idx)
    assert loss is None


def test_sequence_longer_than_block_size_raises():
    with pytest.raises(ValueError):
        make()(torch.zeros(1, 17, dtype=torch.long))


def test_initial_loss_is_near_uniform():
    """With 0.02-std init the logits are ~0, so loss ~= ln(V)."""
    model = make()
    idx = torch.randint(0, 100, (8, 16))
    _, loss = model(idx, torch.randint(0, 100, (8, 16)))
    assert abs(loss.item() - math.log(100)) < 0.1


def test_weight_tying_toggle():
    tied, untied = make(tie_weights=True), make(tie_weights=False)
    assert tied.lm_head.weight is tied.tok_emb.weight
    assert untied.lm_head.weight is not untied.tok_emb.weight
    assert untied.num_params() - tied.num_params() == 100 * 32
    # gradient from the output layer flows into the shared embedding
    idx = torch.randint(0, 100, (2, 8))
    _, loss = tied(idx, idx)
    loss.backward()
    assert tied.tok_emb.weight.grad is not None


def test_param_count_formula():
    """Per block: 4 C^2 + 4C (attn) + 8 C^2 + 5C (mlp) + 4C (2 LNs) = 12 C^2 + 13 C."""
    m = make(pos_encoding="sinusoidal", tie_weights=True)
    C, V, L = 32, 100, 2
    expected = V * C + L * (12 * C * C + 13 * C) + 2 * C
    assert m.num_params() == expected
    assert make(pos_encoding="learned").num_params(non_embedding=False) == expected + 16 * C


def test_sinusoidal_table_properties():
    pe = sinusoidal_table(16, 32)
    assert pe.shape == (16, 32)
    torch.testing.assert_close(pe[0, 0::2], torch.zeros(16))
    torch.testing.assert_close(pe[0, 1::2], torch.ones(16))
    m = make(pos_encoding="sinusoidal")
    assert abs(m.pos_table.std().item() - 0.02) < 1e-6  # matched to the learned-table init
    assert "pos_table" not in dict(m.named_parameters())
    assert "pos_table" not in m.state_dict()  # rebuilt, not stored


def test_residual_projection_init_is_scaled():
    m = make(n_layer=8)
    std = m.blocks[0].attn.out_proj.weight.std().item()
    assert abs(std - 0.02 / math.sqrt(16)) < 0.002


def test_optimizer_groups():
    m = make()
    opt = m.configure_optimizer(0.1, 1e-3, (0.9, 0.95), "cpu")
    decay, no_decay = opt.param_groups
    assert all(p.dim() >= 2 for p in decay["params"]) and decay["weight_decay"] == 0.1
    assert all(p.dim() < 2 for p in no_decay["params"]) and no_decay["weight_decay"] == 0.0
    n_opt = sum(p.numel() for g in opt.param_groups for p in g["params"])
    assert n_opt == m.num_params(non_embedding=False)  # every param exactly once (tied counted once)


def test_sampling_filters():
    logits = torch.tensor([[1.0, 5.0, 3.0, 0.0]])
    assert sample_next(logits, temperature=0).item() == 1
    g = torch.Generator().manual_seed(0)
    draws = {sample_next(logits, 1.0, top_k=2, generator=g).item() for _ in range(200)}
    assert draws <= {1, 2}
    # probs ~ [0.015, 0.866, 0.117, 0.002]; top_p=0.5 keeps only the argmax
    draws = {sample_next(logits, 1.0, top_p=0.5, generator=g).item() for _ in range(100)}
    assert draws == {1}
    draws = {sample_next(logits, 1.0, top_p=0.9, generator=g).item() for _ in range(300)}
    assert draws == {1, 2}


def test_generate_length_and_context_crop():
    m = make().eval()
    idx = torch.zeros(1, 3, dtype=torch.long)
    out = generate(m, idx, max_new_tokens=30, temperature=1.0, top_k=10)  # 33 > block_size
    assert out.shape == (1, 33)
    assert torch.equal(out[:, :3], idx)


def test_perplexity_eval_on_uniform_model():
    m = make().eval()
    tokens = torch.randint(0, 100, (16 * 10 + 1,))
    r = evaluate_perplexity(m, tokens, batch_size=4)
    assert r["tokens"] == 160
    assert abs(r["ppl"] - 100) < 15  # near-uniform init -> ppl ~ V


def test_build_configs_overrides():
    m, t = build_configs("tiny", {"pos_encoding": "none", "seed": 3, "lr": None})
    assert m.pos_encoding == "none" and t.seed == 3 and t.lr == 1e-3
    with pytest.raises(ValueError):
        ModelConfig(n_embd=30, n_head=4)
