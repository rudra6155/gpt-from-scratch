"""Download Tiny Shakespeare, train the byte-level BPE on the TRAIN split only, and write
uint16 token files.

    python data/prepare_tinyshakespeare.py [--vocab_size 512] [--out_dir data/tinyshakespeare]

Outputs (in out_dir): input.txt, tokenizer.json, train.bin, val.bin, meta.json.
Split: first 90% of characters -> train, last 10% -> val (contiguous, so no passage leaks
across the split). The tokenizer never sees val text. Re-running is a no-op unless
--force is given or the vocab size changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from gpt.tokenizer import ByteBPETokenizer  # noqa: E402

URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="data/tinyshakespeare")
    ap.add_argument("--vocab_size", type=int, default=512)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    out = Path(a.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    meta_path = out / "meta.json"
    if meta_path.exists() and not a.force:
        meta = json.loads(meta_path.read_text())
        if meta.get("vocab_size") == a.vocab_size:
            print(f"[data] {out} already prepared (vocab {a.vocab_size}); use --force to rebuild")
            return

    raw = out / "input.txt"
    if not raw.exists():
        print(f"[data] downloading {URL}")
        urllib.request.urlretrieve(URL, raw)
    # newline="" keeps the file's exact bytes (no \r\n translation on Windows)
    with open(raw, encoding="utf-8", newline="") as f:
        text = f.read()
    n = len(text)
    split = int(n * (1 - a.val_frac))
    train_text, val_text = text[:split], text[split:]
    print(f"[data] {n:,} chars -> train {len(train_text):,} / val {len(val_text):,}")

    t0 = time.perf_counter()
    tok = ByteBPETokenizer.train(train_text, a.vocab_size)
    print(f"[data] trained BPE: {tok.vocab_size} tokens in {time.perf_counter() - t0:.1f}s")
    tok.save(out / "tokenizer.json")

    stats = {}
    for name, part in (("train", train_text), ("val", val_text)):
        ids = tok.encode(part)
        assert tok.decode(ids) == part, f"{name}: BPE round-trip failed"
        np.array(ids, dtype=np.uint16).tofile(out / f"{name}.bin")
        stats[name] = len(ids)
        print(f"[data] {name}: {len(ids):,} tokens ({len(part.encode('utf-8')) / len(ids):.2f} bytes/token)")

    meta = {"vocab_size": tok.vocab_size, "train_tokens": stats["train"], "val_tokens": stats["val"],
            "source": URL, "val_frac": a.val_frac}
    meta_path.write_text(json.dumps(meta, indent=2))
    longest = sorted(tok.vocab[256:], key=len)[-8:]
    print("[data] longest learned tokens:", [b.decode("utf-8", "replace") for b in longest])


if __name__ == "__main__":
    main()
