# HELIX Architecture

## Core learning engine (baseline v0.1)

An independently implemented decoder-only transformer trained from random
initialization. No pretrained weights are used as a foundation.

| Component | Choice | Rationale |
|-----------|--------|-----------|
| Backbone | Decoder-only transformer | Simple, well-understood, strong language-modeling baseline |
| Position | RoPE (rotary embeddings) | Extrapolates better than learned/absolute positions |
| Normalization | RMSNorm (pre-norm) | Stable training, fewer parameters than LayerNorm |
| Feed-forward | SwiGLU | Better parameter efficiency than ReLU/GeLU MLPs |
| Attention | Causal self-attention + GQA | Fewer K/V heads reduce memory with near-MQA quality |
| Output | Weight-tied projection | Reduces params, ties input/output representations |
| Vocab | Byte-level BPE (from scratch) | Every byte representable; lossless round-trip |
| Optimizer | AdamW + cosine LR + warmup | Proven, stable baseline |

## CPU-first design

- Ordinary CPU is the primary target.
- `torch.set_num_threads` auto-detected from CPU count.
- float32 default (float16/bfloat16 optional where CPU supports them).
- Conservative fallbacks; no assumed SIMD/AVX/NEON in the core path.

## Project layout

```
helix/         tokenizer, model, data, train, infer, CLI
configs/       JSON experiment configs (e.g. helix-tiny.json)
data/          corpus + tokenizer artifacts + train/val bins
out/           checkpoints + logs (gitignored)
tests/         pytest suite
docs/          architecture, research ledger, model card, data pipeline
```

See `docs/RESEARCH_LEDGER.md` for adopted techniques, sources, and measured results.
