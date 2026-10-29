"""Configuration dataclasses, named presets, and a tiny CLI-override helper.

Two dataclasses keep model shape and optimisation settings separate, so a checkpoint
can carry the ModelConfig alone and be rebuilt without any training settings.
Field names are unique across both classes so each one maps to exactly one CLI flag.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass, fields
from typing import Any, get_type_hints

POS_ENCODINGS = ("learned", "sinusoidal", "none")


@dataclass
class ModelConfig:
    vocab_size: int = 512
    block_size: int = 128          # maximum context length T
    n_layer: int = 4
    n_head: int = 4
    n_embd: int = 128              # d_model; must be divisible by n_head
    dropout: float = 0.1
    bias: bool = True              # biases in Linear + LayerNorm layers
    pos_encoding: str = "learned"  # learned | sinusoidal | none
    tie_weights: bool = True       # share token embedding and lm_head weight
    causal: bool = True            # False ONLY for the deliberate-breakage experiment

    def __post_init__(self) -> None:
        if self.n_embd % self.n_head != 0:
            raise ValueError(f"n_embd={self.n_embd} not divisible by n_head={self.n_head}")
        if self.pos_encoding not in POS_ENCODINGS:
            raise ValueError(f"pos_encoding must be one of {POS_ENCODINGS}, got {self.pos_encoding!r}")


@dataclass
class TrainConfig:
    data_dir: str = "data/tinyshakespeare"
    out_dir: str = "artifacts/runs/tiny"     # resumable checkpoints (with optimizer state)
    log_csv: str = "results/tiny_loss.csv"
    export_path: str = ""                    # weights-only checkpoint written at the end
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_steps: int = 3000
    lr: float = 1e-3
    min_lr: float = 1e-4
    warmup_steps: int = 100
    weight_decay: float = 0.1
    beta1: float = 0.9
    beta2: float = 0.95
    grad_clip: float = 1.0                   # <= 0 disables clipping
    eval_interval: int = 250
    eval_iters: int = 20
    seed: int = 1337
    device: str = "auto"                     # auto | cpu | cuda
    dtype: str = "auto"                      # auto | bf16 | fp32
    resume: bool = False
    stop_step: int = -1                      # >=0: stop after checkpointing at this step (simulated preemption)


# Presets. `tiny` is CPU-runnable in minutes and its exported weights stay < 5 MB.
PRESETS: dict[str, dict[str, dict[str, Any]]] = {
    "tiny": {
        "model": dict(vocab_size=512, block_size=128, n_layer=4, n_head=4, n_embd=128, dropout=0.1),
        "train": dict(batch_size=32, max_steps=3000, lr=1e-3, min_lr=1e-4, warmup_steps=100,
                      eval_interval=50, eval_iters=20,
                      out_dir="artifacts/runs/tiny", log_csv="results/tiny_loss.csv",
                      export_path="artifacts/checkpoints/tiny.pt"),
    },
    # Same model as tiny, shorter schedule: 9 runs (3 encodings x 3 seeds) must be cheap.
    "ablation": {
        "model": dict(vocab_size=512, block_size=128, n_layer=4, n_head=4, n_embd=128, dropout=0.1),
        "train": dict(batch_size=32, max_steps=1500, lr=1e-3, min_lr=1e-4, warmup_steps=100,
                      eval_interval=50, eval_iters=20),
    },
    # GPU config (~10.7M params). Not part of the DoD; provided for real-hardware runs.
    "small": {
        "model": dict(vocab_size=512, block_size=256, n_layer=6, n_head=6, n_embd=384, dropout=0.2),
        "train": dict(batch_size=64, max_steps=5000, lr=6e-4, min_lr=6e-5, warmup_steps=200,
                      eval_interval=250, eval_iters=40,
                      out_dir="artifacts/runs/small", log_csv="results/small_loss.csv",
                      export_path="artifacts/checkpoints/small.pt"),
    },
}


def str2bool(v: str) -> bool:
    if v.lower() in ("1", "true", "yes", "y", "on"):
        return True
    if v.lower() in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"expected a boolean, got {v!r}")


def add_dataclass_args(parser: argparse.ArgumentParser, cls: type) -> None:
    """Expose every dataclass field as `--name VALUE` (default None = keep preset value)."""
    hints = get_type_hints(cls)
    for f in fields(cls):
        typ = hints[f.name]
        parser.add_argument(f"--{f.name}", type=str2bool if typ is bool else typ, default=None,
                            help=f"{cls.__name__}.{f.name} (default {f.default!r})")


def build_configs(preset: str | None, overrides: dict[str, Any]) -> tuple[ModelConfig, TrainConfig]:
    """preset values < explicit CLI overrides (entries whose value is None are ignored)."""
    p = PRESETS.get(preset, {"model": {}, "train": {}}) if preset else {"model": {}, "train": {}}
    model_kw, train_kw = dict(p["model"]), dict(p["train"])
    model_names = {f.name for f in fields(ModelConfig)}
    train_names = {f.name for f in fields(TrainConfig)}
    for k, v in overrides.items():
        if v is None:
            continue
        if k in model_names:
            model_kw[k] = v
        elif k in train_names:
            train_kw[k] = v
    return ModelConfig(**model_kw), TrainConfig(**train_kw)


def asdict(cfg: Any) -> dict[str, Any]:
    return dataclasses.asdict(cfg)
