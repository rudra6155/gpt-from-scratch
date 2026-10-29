"""Pre-LN transformer block:  x + Attn(LN(x)),  then  x + MLP(LN(x)).

Pre-LN (Xiong et al. 2020; used by GPT-2/GPT-3) keeps the residual stream an identity
path from input to output, so gradients reach early layers without passing through
a LayerNorm at every block. Post-LN (original Transformer) needs careful warmup to
avoid divergence at depth.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CausalSelfAttention


class MLP(nn.Module):
    """Position-wise feed-forward: C -> 4C -> GELU -> C, dropout on the output (residual branch)."""

    def __init__(self, n_embd: int, dropout: float = 0.0, bias: bool = True):
        super().__init__()
        self.fc = nn.Linear(n_embd, 4 * n_embd, bias=bias)
        self.act = nn.GELU(approximate="tanh")  # the GPT-2 tanh approximation
        self.proj = nn.Linear(4 * n_embd, n_embd, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.proj(self.act(self.fc(x))))


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, block_size: int, dropout: float = 0.0,
                 bias: bool = True, causal: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd, bias=bias)
        self.attn = CausalSelfAttention(n_embd, n_head, block_size, dropout, bias, causal)
        self.ln2 = nn.LayerNorm(n_embd, bias=bias)
        self.mlp = MLP(n_embd, dropout, bias)

    def forward(self, x: torch.Tensor, keep_attn: bool = False) -> torch.Tensor:
        x = x + self.attn(self.ln1(x), keep_attn=keep_attn)  # dropout lives inside each branch
        x = x + self.mlp(self.ln2(x))
        return x
