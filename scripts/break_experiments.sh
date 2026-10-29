#!/usr/bin/env bash
# Deliberately break things and log what happens (README: "Things that broke").
# Each run is the ablation preset (1500 steps, seed 0) plus the listed overrides.
set -euo pipefail
PY="${PY:-python}"
OUT=results/breaks
mkdir -p "$OUT"
run() {  # name, extra args...
  local name=$1; shift
  echo "=== $name ==="
  "$PY" -m gpt.train --preset ablation --seed 0 --out_dir "artifacts/runs/breaks/$name" \
      --log_csv "$OUT/$name.csv" "$@" || echo "!!! $name crashed (exit $?)" | tee "$OUT/$name.crash.txt"
}
run baseline
run no_causal_mask --causal false
run no_warmup_lr10x_no_clip --warmup_steps 0 --lr 1e-2 --grad_clip 0

# Low training loss is not success: sample from the mask-less model and from the baseline.
for name in baseline no_causal_mask; do
  "$PY" -m gpt.eval sample --ckpt "artifacts/runs/breaks/$name/ckpt_last.pt" --prompt "ROMEO:" \
      --max_new_tokens 120 --out "$OUT/${name}_sample.txt"
done
"$PY" -m gpt.plots breaks --runs_dir "$OUT" --out results/breaks.png
