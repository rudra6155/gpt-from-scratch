#!/usr/bin/env bash
# GPU config (~10.7M params). Not required for the acceptance checks.
# Writes results/small_loss.{csv,png} and artifacts/checkpoints/small.pt (not committed: > 5 MB).
set -euo pipefail
PY="${PY:-python}"

"$PY" -m gpt.train --preset small "$@"
"$PY" -m gpt.plots loss --csv results/small_loss.csv --out results/small_loss.png \
    --title "Tiny Shakespeare: small preset (6L, 384d, BPE-512)"
"$PY" -m gpt.eval perplexity --ckpt artifacts/checkpoints/small.pt --data_dir data/tinyshakespeare
"$PY" -m gpt.eval sample --ckpt artifacts/checkpoints/small.pt --prompt "ROMEO:" \
    --device auto --out results/sample_small.txt
