#!/usr/bin/env bash
# Positional-encoding ablation: {learned, sinusoidal, none} x seeds {0,1,2}, identical
# model/data/schedule otherwise. Per-run CSVs -> results/ablation_runs/,
# summary (mean ± std over seeds) -> results/ablation_pos_enc.{csv,png}.
set -euo pipefail
PY="${PY:-python}"
SEEDS="${SEEDS:-0 1 2}"
RUNS=results/ablation_runs
mkdir -p "$RUNS"

for pos in learned sinusoidal none; do
  for seed in $SEEDS; do
    echo "=== pos_encoding=$pos seed=$seed ==="
    "$PY" -m gpt.train --preset ablation --pos_encoding "$pos" --seed "$seed" \
        --out_dir "artifacts/runs/ablation/${pos}_seed${seed}" \
        --log_csv "$RUNS/${pos}_seed${seed}.csv" "$@"
  done
done
"$PY" -m gpt.plots ablation --runs_dir "$RUNS" --out_prefix results/ablation_pos_enc
