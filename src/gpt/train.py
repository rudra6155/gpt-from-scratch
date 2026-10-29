"""Training loop.

    python -m gpt.train --preset tiny                    # Tiny Shakespeare, CPU or GPU
    python -m gpt.train --preset tiny --resume true      # continue from out_dir/ckpt_last.pt
    python -m gpt.train --preset ablation --pos_encoding none --seed 2 ...
    python -m gpt.train --smoke                          # overfit one batch (asserts loss < 0.1, < 60 s)

Every ModelConfig / TrainConfig field is also a CLI flag (see gpt.config).

Loop structure (step s = one optimizer update, s = 0 .. max_steps-1):
    if s % eval_interval == 0: evaluate -> CSV row -> checkpoint (stores "next step = s")
    lr(s) -> grad_accum_steps micro-batches (bf16 autocast) -> clip -> AdamW step
then one final evaluate/CSV/checkpoint at s = max_steps.

Resume is bit-exact on CPU: the checkpoint holds model + optimizer + python/numpy/torch/cuda
RNG states + the data-sampling generator, and evaluation uses its own fixed-seed generator so
it never perturbs the training RNG stream (tests/test_resume.py checks this).
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .config import ModelConfig, TrainConfig, add_dataclass_args, asdict, build_configs
from .model import GPT

CSV_FIELDS = ["step", "train_loss", "val_loss", "val_ppl", "lr", "tokens_per_sec", "elapsed_s"]


# ----------------------------------------------------------------------------- utilities

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def resolve_device(name: str) -> str:
    if name == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return name


def autocast_ctx(device: str, dtype: str):
    """bf16 autocast on GPUs that support it; fp32 otherwise (CPU bf16 autocast is slow)."""
    use_bf16 = dtype == "bf16" or (dtype == "auto" and device == "cuda"
                                   and torch.cuda.is_bf16_supported())
    if use_bf16:
        return torch.autocast(device_type=device, dtype=torch.bfloat16), "bf16"
    return contextlib.nullcontext(), "fp32"


def get_lr(step: int, cfg: TrainConfig) -> float:
    """Linear warmup to cfg.lr, then cosine decay to cfg.min_lr at cfg.max_steps."""
    if step < cfg.warmup_steps:
        return cfg.lr * (step + 1) / cfg.warmup_steps
    if step >= cfg.max_steps:
        return cfg.min_lr
    progress = (step - cfg.warmup_steps) / max(1, cfg.max_steps - cfg.warmup_steps)
    return cfg.min_lr + 0.5 * (1.0 + math.cos(math.pi * progress)) * (cfg.lr - cfg.min_lr)


def load_tokens(data_dir: str | Path, split: str) -> torch.Tensor:
    arr = np.fromfile(Path(data_dir) / f"{split}.bin", dtype=np.uint16)
    return torch.from_numpy(arr.astype(np.int64))


def get_batch(data: torch.Tensor, batch_size: int, block_size: int,
              gen: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    """Random contiguous windows; y is x shifted left by one token."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=gen)
    x = torch.stack([data[i: i + block_size] for i in ix.tolist()])
    y = torch.stack([data[i + 1: i + 1 + block_size] for i in ix.tolist()])
    return x, y


@torch.no_grad()
def estimate_loss(model: GPT, splits: dict[str, torch.Tensor], cfg: TrainConfig,
                  block_size: int, device: str, ctx) -> dict[str, float]:
    """Mean loss over eval_iters batches per split. A fresh fixed-seed generator means
    every evaluation sees the same batches (lower-variance curves) and the training
    RNG stream is left untouched (exact resume)."""
    model.eval()
    out = {}
    for name, data in splits.items():
        gen = torch.Generator().manual_seed(10_000 + cfg.seed)
        losses = []
        for _ in range(cfg.eval_iters):
            x, y = get_batch(data, cfg.batch_size, block_size, gen)
            with ctx:
                _, loss = model(x.to(device), y.to(device))
            losses.append(loss.item())
        out[name] = float(np.mean(losses))
    model.train()
    return out


def rng_state(batch_gen: torch.Generator) -> dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "batch_gen": batch_gen.get_state(),
    }


def set_rng_state(state: dict[str, Any], batch_gen: torch.Generator) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state["cuda"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])
    batch_gen.set_state(state["batch_gen"])


def export_weights(model: GPT, path: str | Path, tokenizer_path: str | Path | None,
                   extra: dict[str, Any] | None = None) -> None:
    """Weights-only checkpoint (no optimizer state), loadable with weights_only=True.
    The tokenizer merges are embedded so sampling needs nothing but this file."""
    payload: dict[str, Any] = {"model_config": asdict(model.cfg), "model": model.state_dict()}
    if tokenizer_path and Path(tokenizer_path).exists():
        payload["tokenizer"] = json.loads(Path(tokenizer_path).read_text(encoding="utf-8"))
    payload.update(extra or {})
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


# ----------------------------------------------------------------------------- training

def train(mcfg: ModelConfig, tcfg: TrainConfig) -> dict[str, Any]:
    device = resolve_device(tcfg.device)
    ctx, precision = autocast_ctx(device, tcfg.dtype)
    out_dir = Path(tcfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / "ckpt_last.pt"

    meta_path = Path(tcfg.data_dir) / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta["vocab_size"] != mcfg.vocab_size:
            print(f"[train] vocab_size {mcfg.vocab_size} -> {meta['vocab_size']} (from {meta_path})")
            mcfg.vocab_size = meta["vocab_size"]
    splits = {"train": load_tokens(tcfg.data_dir, "train"), "val": load_tokens(tcfg.data_dir, "val")}
    assert int(splits["train"].max()) < mcfg.vocab_size, "token id >= vocab_size"

    seed_everything(tcfg.seed)
    batch_gen = torch.Generator().manual_seed(tcfg.seed)
    model = GPT(mcfg).to(device)
    opt = model.configure_optimizer(tcfg.weight_decay, tcfg.lr, (tcfg.beta1, tcfg.beta2), device)

    start_step, best_val, elapsed0 = 0, float("inf"), 0.0
    if tcfg.resume and ckpt_path.exists():
        # weights_only=False: the checkpoint holds numpy RNG state (trusted local file).
        ck = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        set_rng_state(ck["rng"], batch_gen)
        start_step, best_val, elapsed0 = ck["step"], ck["best_val"], ck["elapsed_s"]
        print(f"[train] resumed from {ckpt_path} at step {start_step}")
    elif tcfg.resume:
        print(f"[train] --resume given but {ckpt_path} not found; starting fresh")

    # CSV: fresh run overwrites; resume keeps rows up to the checkpoint step only.
    csv_path = Path(tcfg.log_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []
    if start_step > 0 and csv_path.exists():
        with open(csv_path, newline="") as f:
            rows = [r for r in csv.DictReader(f) if int(r["step"]) <= start_step]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        w.writerows(rows)

    n_params = model.num_params()
    tokens_per_step = tcfg.batch_size * mcfg.block_size * tcfg.grad_accum_steps
    print(f"[train] device={device} precision={precision} params={n_params / 1e6:.3f}M "
          f"tokens/step={tokens_per_step} steps={start_step}->{tcfg.max_steps}")

    def evaluate_and_log(step: int, lr: float, tps: float, elapsed: float) -> float:
        losses = estimate_loss(model, splits, tcfg, mcfg.block_size, device, ctx)
        row = {"step": step, "train_loss": f"{losses['train']:.5f}", "val_loss": f"{losses['val']:.5f}",
               "val_ppl": f"{math.exp(losses['val']):.4f}", "lr": f"{lr:.6e}",
               "tokens_per_sec": "" if math.isnan(tps) else f"{tps:.1f}", "elapsed_s": f"{elapsed:.1f}"}
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_FIELDS).writerow(row)
        print(f"step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
              f"| val ppl {math.exp(losses['val']):8.3f} | lr {lr:.2e} | tok/s {tps:9.0f}", flush=True)
        return losses["val"]

    def save_ckpt(next_step: int, elapsed: float) -> None:
        torch.save({"model": model.state_dict(), "optimizer": opt.state_dict(),
                    "model_config": asdict(mcfg), "train_config": asdict(tcfg),
                    "step": next_step, "best_val": best_val, "elapsed_s": elapsed,
                    "rng": rng_state(batch_gen)}, ckpt_path)

    def sync() -> None:
        if device == "cuda":
            torch.cuda.synchronize()

    model.train()
    elapsed = elapsed0
    t_last, tokens_since = time.perf_counter(), 0
    skip_first_eval = start_step > 0  # that row + checkpoint already exist
    for step in range(start_step, tcfg.max_steps):
        lr = get_lr(step, tcfg)
        if step % tcfg.eval_interval == 0 and not skip_first_eval:
            sync()
            dt = time.perf_counter() - t_last
            elapsed += dt
            tps = tokens_since / dt if tokens_since else float("nan")
            val = evaluate_and_log(step, lr, tps, elapsed)
            best_val = min(best_val, val)
            save_ckpt(step, elapsed)
            if step == tcfg.stop_step:
                print(f"[train] stop_step={step} reached (simulated preemption)")
                return {"stopped_at": step}
            sync()
            t_last, tokens_since = time.perf_counter(), 0
        skip_first_eval = False

        for g in opt.param_groups:
            g["lr"] = lr
        for _ in range(tcfg.grad_accum_steps):
            x, y = get_batch(splits["train"], tcfg.batch_size, mcfg.block_size, batch_gen)
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with ctx:
                _, loss = model(x, y)
            (loss / tcfg.grad_accum_steps).backward()
        if tcfg.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        tokens_since += tokens_per_step
        if not math.isfinite(loss.item()):
            raise RuntimeError(f"non-finite loss at step {step}")

    sync()
    dt = time.perf_counter() - t_last
    elapsed += dt
    final_val = evaluate_and_log(tcfg.max_steps, get_lr(tcfg.max_steps, tcfg),
                                 tokens_since / dt if tokens_since else float("nan"), elapsed)
    best_val = min(best_val, final_val)
    save_ckpt(tcfg.max_steps, elapsed)
    if tcfg.export_path:
        export_weights(model, tcfg.export_path, Path(tcfg.data_dir) / "tokenizer.json",
                       {"final_val_loss": final_val, "steps": tcfg.max_steps, "seed": tcfg.seed})
        size_mb = os.path.getsize(tcfg.export_path) / 1e6
        print(f"[train] exported weights to {tcfg.export_path} ({size_mb:.2f} MB)")
    return {"final_val_loss": final_val, "best_val_loss": best_val, "elapsed_s": elapsed,
            "params": n_params, "model": model}


# ----------------------------------------------------------------------------- smoke test

SMOKE_TEXT = (
    "First Citizen: Before we proceed any further, hear me speak. All: Speak, speak. "
    "First Citizen: You are all resolved rather to die than to famish? All: Resolved. resolved. "
    "First Citizen: First, you know Caius Marcius is chief enemy to the people."
)


def overfit_one_batch(max_steps: int = 400, target_loss: float = 0.1, time_limit_s: float = 60.0,
                      seed: int = 0, verbose: bool = True) -> dict[str, float]:
    """Memorise one fixed batch on CPU. A correct model + optimizer drives the loss to ~0;
    failing to do so means something in forward/backward/optimizer is broken.
    Raw bytes (vocab 256) so the check does not depend on tokenizer training."""
    seed_everything(seed)
    cfg = ModelConfig(vocab_size=256, block_size=32, n_layer=2, n_head=4, n_embd=64, dropout=0.0)
    model = GPT(cfg)  # CPU on purpose: the claim is "< 60 s on CPU"
    data = torch.tensor(list(SMOKE_TEXT.encode("utf-8")), dtype=torch.long)
    x, y = get_batch(data, 8, cfg.block_size, torch.Generator().manual_seed(seed))
    opt = model.configure_optimizer(weight_decay=0.0, lr=3e-3, betas=(0.9, 0.95), device_type="cpu")

    t0 = time.perf_counter()
    loss_val, step = float("inf"), 0
    for step in range(1, max_steps + 1):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        loss_val = loss.item()
        if verbose and (step == 1 or step % 25 == 0):
            print(f"smoke step {step:4d} loss {loss_val:.4f}")
        if loss_val < target_loss:
            break
    elapsed = time.perf_counter() - t0
    if verbose:
        print(f"smoke: loss {loss_val:.4f} after {step} steps in {elapsed:.1f}s")
    assert loss_val < target_loss, f"failed to overfit one batch: loss {loss_val:.4f} >= {target_loss}"
    assert elapsed < time_limit_s, f"overfit took {elapsed:.1f}s >= {time_limit_s}s"
    return {"loss": loss_val, "steps": step, "elapsed_s": elapsed}


# ----------------------------------------------------------------------------- CLI

def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--preset", default="tiny", help="tiny | ablation | small")
    p.add_argument("--smoke", action="store_true", help="overfit one batch and exit")
    add_dataclass_args(p, ModelConfig)
    add_dataclass_args(p, TrainConfig)
    args = vars(p.parse_args(argv))
    if args.pop("smoke"):
        overfit_one_batch()
        print("SMOKE PASSED")
        return
    mcfg, tcfg = build_configs(args.pop("preset"), args)
    summary = train(mcfg, tcfg)
    summary.pop("model", None)
    print(json.dumps(summary))


if __name__ == "__main__":
    sys.exit(main())
