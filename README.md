# gpt-from-scratch
A GPT-style transformer written from scratch in pure PyTorch: byte-level BPE tokenizer, explicit causal attention, pre-LN blocks, bit-exact checkpoint resume, and an eval/sampling CLI. Trains on Tiny Shakespeare in about 2 minutes on a laptop GPU (0.875M params, ppl 17.9). Includes a positional-encoding ablation and a log of what broke.
