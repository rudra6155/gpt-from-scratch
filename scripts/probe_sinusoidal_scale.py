"""Why the sinusoidal table is rescaled (README "Things that broke").

Runs the ablation preset (seed 0, sinusoidal) three ways and writes one CSV per variant to
results/sinusoidal_scale/:
  matched   the shipped model: table rescaled to std 0.02 (= learned-embedding init)
  unscaled  raw sin/cos table (per-element RMS ~0.71, ~35x the token embeddings)
  vaswani   raw table + token embeddings multiplied by sqrt(d_model) (Vaswani et al. 2017)
The two non-shipped variants are produced by monkeypatching, so the library stays clean.

    python scripts/probe_sinusoidal_scale.py
"""

import math
import sys

import torch

import gpt.model as M
from gpt.config import build_configs
from gpt.train import train

RAW_TABLE = M.sinusoidal_table


def run(variant: str) -> None:
    init, forward = M.GPT.__init__, M.GPT.forward
    if variant in ("unscaled", "vaswani"):
        def raw_init(self, cfg):
            init(self, cfg)
            self.pos_table = RAW_TABLE(cfg.block_size, cfg.n_embd)  # replace the rescaled buffer
        M.GPT.__init__ = raw_init
    if variant == "vaswani":
        emb_forward = torch.nn.Embedding.forward

        def patched(self, idx, targets=None, keep_attn=False):
            self.tok_emb.forward = lambda i: emb_forward(self.tok_emb, i) * math.sqrt(self.cfg.n_embd)
            return forward(self, idx, targets, keep_attn)
        M.GPT.forward = patched
    try:
        m, t = build_configs("ablation", {"pos_encoding": "sinusoidal", "seed": 0,
                                          "out_dir": f"artifacts/runs/sinusoidal_scale/{variant}",
                                          "log_csv": f"results/sinusoidal_scale/{variant}.csv"})
        r = train(m, t)
        print(f"{variant}: final val loss {r['final_val_loss']:.4f}")
    finally:
        M.GPT.__init__, M.GPT.forward = init, forward


if __name__ == "__main__":
    for v in sys.argv[1:] or ["matched", "unscaled", "vaswani"]:
        run(v)
