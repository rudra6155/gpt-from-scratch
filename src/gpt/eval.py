"""Evaluation: held-out perplexity and sampling (temperature / top-k / top-p).

    python -m gpt.eval perplexity --ckpt artifacts/checkpoints/tiny.pt --data_dir data/tinyshakespeare
    python -m gpt.eval sample --ckpt artifacts/checkpoints/tiny.pt --prompt "ROMEO:" --out results/sample.txt
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

from .config import ModelConfig
from .model import GPT
from .tokenizer import ByteBPETokenizer
from .train import autocast_ctx, load_tokens, resolve_device


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[GPT, ByteBPETokenizer | None, dict]:
    """Load an exported (weights-only) checkpoint or a resumable training checkpoint."""
    try:
        ck = torch.load(path, map_location=device, weights_only=True)
    except Exception:  # resumable checkpoints also contain numpy RNG state
        ck = torch.load(path, map_location=device, weights_only=False)
    model = GPT(ModelConfig(**ck["model_config"])).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    tok = None
    if "tokenizer" in ck:
        t = ck["tokenizer"]
        tok = ByteBPETokenizer([tuple(m) for m in t["merges"]], t["pattern"])
    return model, tok, ck


@torch.no_grad()
def evaluate_perplexity(model: GPT, tokens: torch.Tensor, batch_size: int = 32,
                        device: str = "cpu", ctx=None) -> dict[str, float]:
    """Deterministic pass over the whole split in non-overlapping block_size windows.

    Each window is scored from a fresh context, so early tokens in a window see little
    history: this is slightly pessimistic vs. sliding-window evaluation, but cheap and exact.
    Returns token-weighted mean NLL (nats/token) and perplexity = exp(NLL).
    """
    ctx = ctx or contextlib.nullcontext()
    model.eval()
    T = model.cfg.block_size
    n_windows = (len(tokens) - 1) // T
    xs = tokens[: n_windows * T].view(n_windows, T)
    ys = tokens[1: n_windows * T + 1].view(n_windows, T)
    total_nll, total_tok = 0.0, 0
    for i in range(0, n_windows, batch_size):
        x, y = xs[i: i + batch_size].to(device), ys[i: i + batch_size].to(device)
        with ctx:
            logits, _ = model(x)
        nll = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum")
        total_nll += nll.item()
        total_tok += y.numel()
    mean = total_nll / total_tok
    return {"nll": mean, "ppl": math.exp(mean), "tokens": total_tok}


def sample_next(logits: torch.Tensor, temperature: float = 1.0, top_k: int | None = None,
                top_p: float | None = None, generator: torch.Generator | None = None) -> torch.Tensor:
    """logits (B, V) -> next token ids (B, 1).

    temperature 0 -> greedy argmax. Filters apply in order: temperature, top-k, top-p
    (nucleus, Holtzman et al. 2020: keep the smallest set of tokens whose probability mass
    reaches top_p; the most likely token is always kept).
    """
    if temperature == 0:
        return logits.argmax(dim=-1, keepdim=True)
    logits = logits.float() / temperature
    if top_k is not None and 0 < top_k < logits.size(-1):
        kth = torch.topk(logits, top_k, dim=-1).values[..., -1:]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and 0 < top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, dim=-1, descending=True)
        probs = F.softmax(sorted_logits, dim=-1)
        mass_before = torch.cumsum(probs, dim=-1) - probs   # mass of strictly-more-likely tokens
        remove_sorted = mass_before >= top_p
        remove = torch.zeros_like(remove_sorted).scatter(-1, sorted_idx, remove_sorted)
        logits = logits.masked_fill(remove, float("-inf"))
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1, generator=generator)


@torch.no_grad()
def generate(model: GPT, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0,
             top_k: int | None = None, top_p: float | None = None,
             generator: torch.Generator | None = None) -> torch.Tensor:
    """Autoregressive sampling; the context is cropped to the last block_size tokens.
    (No KV cache: each step recomputes the window. Fine at this scale; see README.)"""
    model.eval()
    for _ in range(max_new_tokens):
        cond = idx[:, -model.cfg.block_size:]
        logits, _ = model(cond)
        nxt = sample_next(logits[:, -1, :], temperature, top_k, top_p, generator)
        idx = torch.cat([idx, nxt], dim=1)
    return idx


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    pp = sub.add_parser("perplexity")
    pp.add_argument("--ckpt", required=True)
    pp.add_argument("--data_dir", default="data/tinyshakespeare")
    pp.add_argument("--split", default="val")
    pp.add_argument("--device", default="auto")
    pp.add_argument("--out", default="", help="optional JSON file for the result")
    ps = sub.add_parser("sample")
    ps.add_argument("--ckpt", required=True)
    ps.add_argument("--prompt", default="ROMEO:")
    ps.add_argument("--max_new_tokens", type=int, default=400)
    ps.add_argument("--temperature", type=float, default=0.8)
    ps.add_argument("--top_k", type=int, default=50)
    ps.add_argument("--top_p", type=float, default=0.95)
    ps.add_argument("--seed", type=int, default=1337)
    ps.add_argument("--device", default="cpu")
    ps.add_argument("--out", default="")
    ps.add_argument("--tokenizer", default="data/tinyshakespeare/tokenizer.json",
                    help="used only if the checkpoint has no embedded tokenizer")
    a = p.parse_args(argv)

    device = resolve_device(a.device)
    model, tok, ck = load_checkpoint(a.ckpt, device)
    if a.cmd == "perplexity":
        ctx, precision = autocast_ctx(device, "auto")
        r = evaluate_perplexity(model, load_tokens(a.data_dir, a.split), device=device, ctx=ctx)
        print(f"{a.split}: nll {r['nll']:.4f} nats/token | perplexity {r['ppl']:.3f} "
              f"| {r['tokens']} tokens | {precision}")
        if a.out:
            r.update(ckpt=a.ckpt, split=a.split, precision=precision, windows="non-overlapping",
                     block_size=model.cfg.block_size)
            Path(a.out).write_text(json.dumps(r, indent=2) + "\n", encoding="utf-8")
            print(f"[eval] wrote {a.out}")
        return

    if tok is None:
        if not Path(a.tokenizer).exists():
            raise SystemExit("checkpoint has no embedded tokenizer and --tokenizer not found")
        tok = ByteBPETokenizer.load(a.tokenizer)
    gen = torch.Generator(device=device).manual_seed(a.seed)
    idx = torch.tensor([tok.encode(a.prompt)], dtype=torch.long, device=device)
    out = generate(model, idx, a.max_new_tokens, a.temperature, a.top_k, a.top_p, gen)
    text = tok.decode(out[0].tolist())
    header = (f"# checkpoint: {a.ckpt}\n# prompt: {a.prompt!r} | max_new_tokens={a.max_new_tokens} "
              f"temperature={a.temperature} top_k={a.top_k} top_p={a.top_p} seed={a.seed}\n"
              f"# model: {model.num_params() / 1e6:.3f}M params, val loss "
              f"{ck.get('final_val_loss', ck.get('best_val', float('nan'))):.4f}\n\n")
    print(text)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(header + text + "\n", encoding="utf-8")
        print(f"\n[eval] wrote {a.out}")


if __name__ == "__main__":
    main()
