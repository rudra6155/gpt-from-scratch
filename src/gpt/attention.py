"""Multi-head causal self-attention, written out explicitly.

Deliberately NOT using torch.nn.MultiheadAttention or F.scaled_dot_product_attention:
every step of  softmax(Q K^T / sqrt(d_head) + causal_mask) V  is visible below.
(The tests use F.scaled_dot_product_attention only as a reference oracle.)

Shapes: B batch, T sequence length, C = n_embd, H = n_head, D = C // H.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def causal_mask(T: int, device: torch.device | None = None) -> torch.Tensor:
    """(T, T) bool, True where query i may attend to key j, i.e. j <= i."""
    return torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))


def attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
              mask: torch.Tensor | None = None,
              dropout: nn.Module | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Scaled dot-product attention on (..., T, D) tensors.

    Returns (output, attention_weights). `mask` is broadcastable to (..., T, T), True = keep.
    """
    d = q.size(-1)
    scores = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(d))       # (..., T, T)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    # Softmax in at least fp32: under bf16 autocast the exp/sum loses too much precision.
    # (Upcast only half types: a blanket .float() silently DOWNcasts fp64 — see README.)
    if scores.dtype in (torch.float16, torch.bfloat16):
        weights = F.softmax(scores.float(), dim=-1).to(q.dtype)
    else:
        weights = F.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return weights @ v, weights


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0,
                 bias: bool = True, causal: bool = True):
        super().__init__()
        assert n_embd % n_head == 0
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.causal = causal
        # Separate projections (rather than one fused 3C matmul) to keep Q/K/V explicit.
        self.q_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.k_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.v_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.out_proj = nn.Linear(n_embd, n_embd, bias=bias)
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)
        # Non-persistent: rebuilt from block_size, never stored in checkpoints.
        self.register_buffer("mask", causal_mask(block_size).view(1, 1, block_size, block_size),
                             persistent=False)
        self.last_attn: torch.Tensor | None = None  # filled when forward(..., keep_attn=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        return x.view(B, T, self.n_head, self.head_dim).transpose(1, 2)  # (B, H, T, D)

    def forward(self, x: torch.Tensor, keep_attn: bool = False) -> torch.Tensor:
        B, T, C = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        mask = self.mask[:, :, :T, :T] if self.causal else None
        y, w = attention(q, k, v, mask, self.attn_dropout)       # (B, H, T, D)
        if keep_attn:
            self.last_attn = w.detach()
        y = y.transpose(1, 2).contiguous().view(B, T, C)           # concat heads
        return self.resid_dropout(self.out_proj(y))
