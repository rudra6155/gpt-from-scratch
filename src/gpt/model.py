"""Decoder-only GPT.

    idx (B,T) -> token embedding (+ positional encoding) -> dropout
              -> N x pre-LN Block -> final LayerNorm -> lm_head -> logits (B,T,V)

Positional encoding is selected by ModelConfig.pos_encoding:
  learned     trainable (block_size, C) table, as in GPT-2 (default)
  sinusoidal  fixed sin/cos table from Vaswani et al. 2017, no parameters, rescaled to
              std INIT_STD so it starts at the same magnitude as the learned table
              (unscaled it is ~35x larger than the token embeddings: see README)
  none        no positional signal at all ("NoPE"); the causal mask still leaks
              position implicitly (Haviv et al. 2022), which the ablation measures.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Block
from .config import ModelConfig

INIT_STD = 0.02  # GPT-2 init std for Linear/Embedding weights


def sinusoidal_table(n_pos: int, dim: int) -> torch.Tensor:
    """PE[p, 2i] = sin(p / 10000^(2i/dim)),  PE[p, 2i+1] = cos(same)."""
    pos = torch.arange(n_pos, dtype=torch.float32).unsqueeze(1)                 # (P, 1)
    inv_freq = torch.exp(-math.log(10000.0) * torch.arange(0, dim, 2, dtype=torch.float32) / dim)
    pe = torch.zeros(n_pos, dim)
    pe[:, 0::2] = torch.sin(pos * inv_freq)
    pe[:, 1::2] = torch.cos(pos * inv_freq[: dim // 2])
    return pe


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        if cfg.pos_encoding == "learned":
            self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        elif cfg.pos_encoding == "sinusoidal":
            table = sinusoidal_table(cfg.block_size, cfg.n_embd)
            self.register_buffer("pos_table", table * (INIT_STD / table.std()), persistent=False)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList(
            Block(cfg.n_embd, cfg.n_head, cfg.block_size, cfg.dropout, cfg.bias, cfg.causal)
            for _ in range(cfg.n_layer)
        )
        self.ln_f = nn.LayerNorm(cfg.n_embd, bias=cfg.bias)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        self.apply(self._init_weights)
        # GPT-2: scale the init of projections that write into the residual stream by
        # 1/sqrt(2 * n_layer) so the stream's variance does not grow with depth.
        for name, p in self.named_parameters():
            if name.endswith("attn.out_proj.weight") or name.endswith("mlp.proj.weight"):
                nn.init.normal_(p, mean=0.0, std=INIT_STD / math.sqrt(2 * cfg.n_layer))
        # Tie AFTER init so both names point at one tensor with the embedding's init.
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=INIT_STD)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=INIT_STD)

    def num_params(self, non_embedding: bool = True) -> int:
        """Unique parameters (a tied weight counts once); non_embedding drops the learned
        position table, following nanoGPT's reporting convention."""
        n = sum(p.numel() for p in self.parameters())  # parameters() de-duplicates tied tensors
        if non_embedding and self.cfg.pos_encoding == "learned":
            n -= self.pos_emb.weight.numel()
        return n

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None,
                keep_attn: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        B, T = idx.shape
        if T > self.cfg.block_size:
            raise ValueError(f"sequence length {T} exceeds block_size {self.cfg.block_size}")
        x = self.tok_emb(idx)                                    # (B, T, C)
        if self.cfg.pos_encoding == "learned":
            x = x + self.pos_emb(torch.arange(T, device=idx.device))
        elif self.cfg.pos_encoding == "sinusoidal":
            x = x + self.pos_table[:T]
        x = self.drop(x)
        for block in self.blocks:
            x = block(x, keep_attn=keep_attn)
        logits = self.lm_head(self.ln_f(x))                      # (B, T, V)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

    def configure_optimizer(self, weight_decay: float, lr: float, betas: tuple[float, float],
                            device_type: str) -> torch.optim.AdamW:
        """AdamW with decoupled weight decay on matrices only (Linear/Embedding weights);
        biases and LayerNorm gains are not decayed, since shrinking them toward 0 is not a
        sensible prior. Grouping by ndim is the same rule nanoGPT uses."""
        params = [p for p in self.parameters() if p.requires_grad]  # tied weight appears once
        decay = [p for p in params if p.dim() >= 2]
        no_decay = [p for p in params if p.dim() < 2]
        groups = [{"params": decay, "weight_decay": weight_decay},
                  {"params": no_decay, "weight_decay": 0.0}]
        return torch.optim.AdamW(groups, lr=lr, betas=betas, fused=(device_type == "cuda"))
