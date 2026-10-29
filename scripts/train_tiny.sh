#!/usr/bin/env bash
# Train the tiny preset on Tiny Shakespeare (CPU-runnable; uses the GPU if present).
# Writes results/tiny_loss.{csv,png}, results/tiny_val_perplexity.json,
# artifacts/checkpoints/tiny.pt and results/sample.txt.
# Extra args are forwarded to gpt.train, e.g.  bash scripts/train_tiny.sh --max_steps 500
set -euo pipefail
PY="${PY:-python}"

"$PY" -m gpt.train --preset tiny "$@"
"$PY" -m gpt.plots loss --csv results/tiny_loss.csv --out results/tiny_loss.png \
    --title "Tiny Shakespeare: tiny preset (4L, 128d, BPE-512)"
"$PY" -m gpt.eval perplexity --ckpt artifacts/checkpoints/tiny.pt --data_dir data/tinyshakespeare \
    --out results/tiny_val_perplexity.json
"$PY" -m gpt.eval sample --ckpt artifacts/checkpoints/tiny.pt --prompt "ROMEO:" --out results/sample.txt
