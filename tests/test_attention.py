import ast
import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from gpt.attention import CausalSelfAttention, attention, causal_mask

SRC = Path(__file__).resolve().parents[1] / "src" / "gpt"


def test_no_forbidden_apis_or_libraries():
    """Model code (not tests) must not call built-in attention or import HF/einops/tiktoken.
    Uses the AST, so mentions in docstrings/comments don't count."""
    banned_attrs = {"MultiheadAttention", "scaled_dot_product_attention"}
    banned_modules = {"transformers", "einops", "tiktoken", "tokenizers", "sentencepiece"}
    for f in SRC.glob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            if isinstance(node, ast.Attribute):
                assert node.attr not in banned_attrs, f"{f.name}:{node.lineno} uses {node.attr}"
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    assert alias.name.split(".")[0] not in banned_modules, f"{f.name} imports {alias.name}"
            elif isinstance(node, ast.ImportFrom) and node.module:
                assert node.module.split(".")[0] not in banned_modules, f"{f.name} imports {node.module}"


def test_causal_mask_shape_and_values():
    m = causal_mask(4)
    assert m.dtype == torch.bool
    assert torch.equal(m, torch.tensor([[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1]]).bool())


def test_attention_matches_torch_oracle():
    torch.manual_seed(0)
    B, H, T, D = 2, 3, 7, 8
    q, k, v = (torch.randn(B, H, T, D, dtype=torch.float64) for _ in range(3))
    ours, _ = attention(q, k, v, causal_mask(T))
    ref = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # oracle only
    torch.testing.assert_close(ours, ref)


def test_attention_matches_explicit_loop():
    """Independent reference: per-head, per-query python loop over allowed keys."""
    torch.manual_seed(1)
    B, H, T, D = 1, 2, 5, 4
    q, k, v = (torch.randn(B, H, T, D, dtype=torch.float64) for _ in range(3))
    ours, _ = attention(q, k, v, causal_mask(T))
    ref = torch.zeros_like(ours)
    for h in range(H):
        for i in range(T):
            scores = torch.stack([q[0, h, i] @ k[0, h, j] / math.sqrt(D) for j in range(i + 1)])
            w = torch.softmax(scores, 0)
            ref[0, h, i] = sum(w[j] * v[0, h, j] for j in range(i + 1))
    torch.testing.assert_close(ours, ref)


def test_attention_weights_are_causal_distributions():
    torch.manual_seed(2)
    layer = CausalSelfAttention(n_embd=32, n_head=4, block_size=16).eval()
    x = torch.randn(2, 10, 32)
    layer(x, keep_attn=True)
    w = layer.last_attn                                          # (B, H, T, T)
    assert w.shape == (2, 4, 10, 10)
    torch.testing.assert_close(w.sum(-1), torch.ones(2, 4, 10))
    upper = torch.triu(torch.ones(10, 10, dtype=torch.bool), diagonal=1)
    assert torch.all(w[..., upper] == 0)
    assert torch.all(w[..., 0, 0] == 1)                          # first token attends only to itself


def test_multihead_equals_concat_of_single_heads():
    """Splitting into heads then concatenating == running each head's slice separately."""
    torch.manual_seed(3)
    C, H, T = 16, 4, 6
    layer = CausalSelfAttention(C, H, block_size=8).eval().double()
    x = torch.randn(1, T, C, dtype=torch.float64)
    out = layer(x)
    q, k, v = layer.q_proj(x), layer.k_proj(x), layer.v_proj(x)
    D = C // H
    heads = [attention(q[..., h * D:(h + 1) * D], k[..., h * D:(h + 1) * D],
                       v[..., h * D:(h + 1) * D], causal_mask(T))[0] for h in range(H)]
    torch.testing.assert_close(out, layer.out_proj(torch.cat(heads, dim=-1)))


@pytest.mark.parametrize("T", [1, 5, 16])
def test_output_shape(T):
    layer = CausalSelfAttention(n_embd=24, n_head=3, block_size=16)
    assert layer(torch.randn(3, T, 24)).shape == (3, T, 24)


def test_scaling_by_sqrt_head_dim():
    """Constant keys/queries: unscaled scores would be D * c^2; we must see c^2 * sqrt(D)."""
    D = 16
    q = torch.full((1, 1, 2, D), 0.5)
    k = torch.full((1, 1, 2, D), 0.5)
    v = torch.tensor([[[[1.0] * D, [3.0] * D]]])
    _, w = attention(q, k, v, mask=None)
    torch.testing.assert_close(w, torch.full((1, 1, 2, 2), 0.5))  # equal scores -> uniform
    k2 = k.clone()
    k2[..., 1, :] = 0.0
    _, w2 = attention(q, k2, v, mask=None)
    expected = torch.softmax(torch.tensor([0.25 * D / math.sqrt(D), 0.0]), 0)
    torch.testing.assert_close(w2[0, 0, 0], expected)
