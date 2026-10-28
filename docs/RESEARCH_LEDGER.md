# HELIX Research Ledger

Each adopted technique records: hypothesis, source/reference, implementation
status, baseline, hardware, configuration, measured results, regressions,
limitations, and decision.

## v0.1 — Minimal CPU baseline (random init)

| Entry | Detail |
|-------|--------|
| Hypothesis | A tiny decoder-only transformer can be trained end-to-end on CPU from random init, with checkpoint/resume and eval, using only numpy+torch. |
| Reference | nanoGPT (karpathy/nanoGPT) — inspected as a reference; implementation written independently. |
| Techniques | RoPE, RMSNorm (pre-norm), SwiGLU, GQA, weight tying, AdamW + cosine LR + warmup, gradient accumulation, checkpoint save/resume. |
| Status | Implemented (helix/model.py, helix/train.py, helix/data.py, helix/tokenizer.py). |
| Hardware | 2 vCPU, ~3.8 GB RAM, CPU only. |
| Baseline | random initialization; seed corpus in data/seed_corpus.txt. |
| Config | configs/helix-tiny.json (2 layers, 4 heads, 128 embd, block 128, ~300 iters). |
| Results | Pending first training run (see out/tiny-run/log.jsonl). |
| Limitations | Tiny model/corpus — quality is a smoke test, not a capability claim. Byte-level BPE without regex pre-tokenization is slower to encode than a production tokenizer. No KV cache during generation (full recompute). |
| Decision | Ship v0.1 baseline, then iterate: eval harness, ingestion pipeline, architecture ablations. |

## Adoption policy

Every technique is a hypothesis to benchmark. Adopt only when measured
evaluation justifies it: quality per token, quality per second, memory
efficiency, stability, reproducibility, or inference speed. Distinguish
established methods from experimental/unverified ideas.
