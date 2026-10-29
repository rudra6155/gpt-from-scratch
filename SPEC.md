# SPEC — gpt-scratch

This file restates the acceptance criteria for the project *before* any code was written.
It is the contract the repo is checked against. Every box at the bottom must be ticked
by running the listed command on a fresh clone.

## Mission

A from-scratch, decoder-only, GPT-style transformer in pure PyTorch.

- No HuggingFace / `transformers`, no `einops`, no `tiktoken`, no nanoGPT code copied verbatim
  (nanoGPT may be *read*; conceptual borrowings are credited in the README).
- Educational depth **and** real correctness: every design choice documented, every claim
  reproducible by a command, every README claim backed by a committed artifact file.

## Technical requirements

| # | Component | Requirement |
|---|-----------|-------------|
| 1 | Tokenizer | Byte-level BPE from scratch. `train(corpus, vocab_size) -> merges`; `encode` / `decode`; save/load JSON; round-trip tests on unicode, emoji, whitespace. |
| 2 | Attention | Explicit Q/K/V projections, scaled dot-product, causal mask, multi-head, output projection. **Not** `torch.nn.MultiheadAttention`, **not** `F.scaled_dot_product_attention`. |
| 3 | Block | Pre-LN (LayerNorm before attention and before MLP), residual connections, GELU MLP, dropout on residual branches. |
| 4 | Model | Learned positional embeddings by default; sinusoidal path selectable by config (plus `none` for the ablation). Weight tying embedding <-> lm_head toggleable. |
| 5 | Training | AdamW; linear warmup + cosine LR; grad clip 1.0; bf16 autocast when available; grad accumulation; seeds for python / numpy / torch / cudnn; checkpoint + resume; val perplexity every N steps; CSV log with `step, train_loss, val_loss, lr, tokens_per_sec`. CPU-runnable small config; GPU for real runs. |
| 6 | Eval | Perplexity on the held-out split; sampling with temperature, top-k, top-p. |

## Acceptance commands (must pass on a fresh clone)

| Command | Must do |
|---------|---------|
| `make install` | create a venv and install dependencies |
| `make test` | run pytest, total < 90 s on CPU |
| `make smoke` | overfit one batch to loss < 0.1 in < 60 s on CPU — **asserted in code** |
| `make train-tiny` | train on Tiny Shakespeare; write `results/tiny_loss.csv` + `results/tiny_loss.png` |
| `make ablation` | positional encoding {learned, sinusoidal, none} x 3 seeds; write `results/ablation_pos_enc.csv` + `.png` (mean ± std over seeds) |
| `make docker` | build the Docker image |
| `make docker-test` | run the test suite inside the image |

CI: GitHub Actions runs `make test` and `make smoke` on every push.

## Artifacts to commit

- `results/tiny_loss.csv`, `results/tiny_loss.png` — from a real run on this machine
- `results/ablation_pos_enc.csv`, `results/ablation_pos_enc.png` — mean ± std over 3 seeds
- `results/sample.txt` — a real generation from the trained checkpoint
- `artifacts/checkpoints/tiny.pt` — < 5 MB

## README requirements

- "What this is / What this is NOT" (small educational GPT; NOT GPT-2 scale, NOT instruction-tuned, NOT production)
- ASCII architecture diagram
- Command table mapping every claim to a command
- "Design decisions": pre-LN, warmup + cosine, byte-level BPE, weight tying
- "Things that broke": >= 5 real entries, each with its fix
- "Limitations and next steps"
- Citations: GPT-2, GPT-3, Chinchilla
- No claim without an artifact file

## Repo layout

```
gpt-scratch/
  README.md  SPEC.md  pyproject.toml  Makefile  Dockerfile
  .github/workflows/ci.yml
  src/gpt/{__init__,tokenizer,attention,block,model,train,eval,config}.py
  tests/{test_tokenizer,test_attention,test_causality,test_model_shapes,test_overfit_one_batch}.py
  scripts/{train_tiny.sh,train_small.sh,ablation_pos_enc.sh}
  data/prepare_tinyshakespeare.py
  results/    artifacts/
```

Extra files are allowed where they make a requirement testable (e.g. `tests/test_resume.py`,
`src/gpt/plots.py`).

## Definition of done

- [x] fresh clone: `make install && make test && make smoke` pass (clean checkout of the index, Windows; Docker/Linux too)
- [x] real training CSV + PNG committed
- [x] ablation (3 configs x 3 seeds) committed
- [x] README has every section, including "Things that broke"
- [ ] CI green (pending first push); [x] Docker builds and passes tests (`make docker && make docker-test`)

Process rule: after finishing, run every DoD command, report "these failed: ...", and iterate
until all are green. Do not declare done before that.
