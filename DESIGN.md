# DESIGN — gpt-scratch Inspection Workbench

## Identity & Intent
- **System:** Local model inspector and inference workbench for a from-scratch PyTorch transformer trained on Tiny Shakespeare (0.875M parameters).
- **Target Audience:** ML engineers, researchers, and systems developers evaluating tokenizer behavior, attention dynamics, loss curves, and empirical ablations.
- **Character:** Precision instrument. Utilitarian, focused, unadorned. Modeled after internal research tools and profilers (e.g., PyTorch Profiler, Karpathy's minimal tooling), rather than marketing frontends.

## UI & Design Specifications
1. **High Data-to-Ink Ratio:**
   - Zero decorative noise: no marketing gradients, glowing halos, floating particles, or decorative blur meshes.
   - Clean solid matte surfaces with structured borders.
   - Strict avoidance of decorative status pills, emojis in telemetry labels, or mock window chrome.
2. **Empirical Grounding & Telemetry:**
   - Every metric maps directly to committed artifacts (`results/` and `artifacts/checkpoints/tiny.pt`).
   - Architecture constraints stated explicitly: 0.875M parameters, 1MB corpus, autoregressive character/token continuation.
   - Live stream telemetry with explicit per-token throughput (`tok/s`) and elapsed latency.
3. **Typography & Spatial Hierarchy:**
   - Interface sans-serif: Modern high-density sans-serif (`Inter`, `-apple-system`, `system-ui`).
   - Tensors, tokens, code, and metrics: Fixed-width monospace (`JetBrains Mono`, `ui-monospace`, `monospace`).
   - Palette: Low-contrast matte slate/graphite base (`#111317` foundation, `#181b22` panels, `#232732` dividers).
   - Functional Accent: Focused amber (`#d97706` / `#f59e0b`) reserved strictly for active states, interactive controls, and primary execution.
4. **Interactive Inspection:**
   - Real-time token streaming with precise latency and throughput measurement.
   - Interactive token decomposition showing raw byte rank and exact token IDs ($0 \dots 511$).
   - Explicit random seed controls for reproducible sampling.
   - Tabular, side-by-side inspection of ablation experiments and failure modes.
