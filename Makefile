# gpt-scratch — every claim in the README maps to one of these targets.
#
# Works on Linux/macOS (CI, Docker) and on Windows with GNU make + Git Bash.
# GPU users:  make install TORCH_INDEX=https://download.pytorch.org/whl/cu126

SHELL := bash

ifeq ($(OS),Windows_NT)
  SYS_PY  ?= python
  VENV_PY := .venv/Scripts/python.exe
else
  SYS_PY  ?= python3
  VENV_PY := .venv/bin/python
endif

TORCH_INDEX ?= https://download.pytorch.org/whl/cpu
PY := $(VENV_PY)
IMAGE ?= gpt-scratch

export PY

.PHONY: help install test smoke data train-tiny train-small ablation sample breaks docker docker-test clean

help:
	@echo "make install | test | smoke | data | train-tiny | train-small | ablation | sample | breaks | docker | docker-test"

install:
	$(SYS_PY) -m venv .venv
	$(PY) -m pip install --upgrade pip
	$(PY) -m pip install torch --index-url $(TORCH_INDEX)
	$(PY) -m pip install -e ".[dev]"

test:
	$(PY) -m pytest

smoke:
	set -o pipefail; $(PY) -m gpt.train --smoke | tee results/smoke.log

data:
	$(PY) data/prepare_tinyshakespeare.py

train-tiny: data
	bash scripts/train_tiny.sh

train-small: data
	bash scripts/train_small.sh

ablation: data
	bash scripts/ablation_pos_enc.sh

sample:
	$(PY) -m gpt.eval sample --ckpt artifacts/checkpoints/tiny.pt --out results/sample.txt

breaks: data
	bash scripts/break_experiments.sh

docker:
	docker build -t $(IMAGE) .

docker-test:
	docker run --rm $(IMAGE) make test

clean:
	rm -rf .pytest_cache artifacts/runs
