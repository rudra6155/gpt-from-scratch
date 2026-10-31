# SPEC — gpt-scratch

Technical specification, architectural requirements, and verification benchmarks for `gpt-scratch`.
Every target is grounded by reproducible automated targets and committed empirical artifacts.

## Architecture & Scope

A ground-up, decoder-only transformer implemented strictly in pure PyTorch without high-level library abstractions.

- No external model frameworks (`transformers`, `einops`, `tiktoken`).
- Educational clarity with mathematical and numerical rigor: full transparency into tensor shapes, residual scaling, attention masks, and gradient flow.
- Every experimental claim backed by reproducible scripts and committed benchmark artifacts.

## Technical Specifications

| # | Subsystem | Engineering Specification |
|---|-----------|---------------------------|
| 1 | Tokenizer | Byte-level BPE from scratch. `train(corpus, vocab_size) -> merges`; `encode` / `decode`; save/load JSON; full round-trip guarantees on UTF-8, multi-byte emoji, and whitespace. |
| 2 | Attention | Explicit Q, K, V projections, scaled dot-product, causal upper-triangular mask, multi-head projection, output projection. Built from basic tensor ops (no `torch.nn.MultiheadAttention` or fused kernels). |
| 3 | Transformer Block | Pre-LN architecture (LayerNorm before attention and MLP), residual skip connections, GELU activation, dropout on residual branches. Residual projections scaled by $1 / \sqrt{2 \cdot N_{\text{layer}}}$. |
| 4 | Embeddings & Positional | Learned positional embeddings by default; pluggable sinusoidal table (matched-std scaled) and ablation mode (`none`). Toggleable weight tying between input embedding and unembedding head. |
| 5 | Training Pipeline | AdamW optimizer ($\beta_1=0.9, \beta_2=0.95$, weight decay $0.1$ on 2D matrices only); linear warmup + cosine decay to $0.1 \times \text{lr}$; grad clip at $1.0$; mixed precision (`bfloat16`/`float16`) with fp32 softmax fallback; bit-exact checkpoint resume across all RNG streams. |
| 6 | Evaluation & Sampling | Windowed perplexity calculation; autoregressive sampling supporting temperature scaling, top-k filtering, and nucleus (top-p) truncation. |

## Verification & Acceptance Targets

| Target Command | Acceptance Assertion |
|----------------|----------------------|
| `make install` | Virtualenv creation and editable package installation |
| `make test` | 65-test verification suite (Jacobian causality, attention oracle, round-trip BPE) |
| `make smoke` | Overfit one batch to loss < 0.1 in < 60s on CPU |
| `make train-tiny` | Train tiny preset on Tiny Shakespeare; log loss curves and generate sample |
| `make ablation` | Positional encoding ablation ({learned, sinusoidal, none} $\times$ 3 seeds) |
| `make docker` | Containerized image build |
| `make docker-test` | Execute verification suite within isolated Linux container |

CI: GitHub Actions executes `make test` and `make smoke` on every commit.

## Tracked Artifacts

- `results/tiny_loss.csv`, `results/tiny_loss.png` — Real training telemetry on local GPU/CPU
- `results/ablation_pos_enc.csv`, `results/ablation_pos_enc.png` — Mean ± std over 3 seeds
- `results/sample.txt` — Autoregressive sample generated from trained tiny weights
- `artifacts/checkpoints/tiny.pt` — Self-contained model checkpoint (< 5 MB)

## Repository Layout

```
gpt-scratch/
  README.md  SPEC.md  DESIGN.md  pyproject.toml  Makefile  Dockerfile
  .github/workflows/ci.yml
  src/gpt/{__init__,tokenizer,attention,block,model,train,eval,config,plots}.py
  tests/{test_tokenizer,test_attention,test_causality,test_model_shapes,test_overfit_one_batch,test_resume}.py
  scripts/{train_tiny.sh,train_small.sh,ablation_pos_enc.sh,break_experiments.sh,probe_sinusoidal_scale.py}
  data/prepare_tinyshakespeare.py
  results/    artifacts/
```

## Acceptance Status

- [x] Clean checkout validation: `make install && make test && make smoke` passing.
- [x] Verified training CSV and loss curves generated.
- [x] Positional encoding ablation matrix (3 variants $\times$ 3 seeds) executed.
- [x] Complete architecture and failure-mode documentation.
- [x] Docker container build and headless test suite verified.
