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

## v0.2 — Evaluation framework + corpus discovery/ingestion

| Entry | Detail |
|-------|--------|
| Hypothesis | A CPU-runnable, contamination-resistant evaluation suite plus a rights-aware corpus manifest can be built with executable code and tests, without network access. |
| Reference | nanoGPT eval patterns; OLMo/DataTrove-style manifests inspected as references; implementation written independently. |
| Techniques | Original eval tasks generated at eval time (arithmetic, code completion, cloze, needle-in-haystack, calibration); per-document provenance manifest (source id, URL, format, language, license, rights, content hash, retrieval timestamp, quality score, quarantine flag). |
| Status | Implemented (helix/eval.py, helix/corpus.py, tests/test_eval.py, tests/test_corpus.py); eval wired into `helix eval` CLI. |
| Hardware | 2 vCPU, ~3.8 GB RAM, CPU only. |
| Baseline | v0.1 tiny model on out/tiny-run/ckpt-best.pt. |
| Results | out/eval-report.json: 2/5 tasks passed on the tiny baseline (long-context recall + calibration pass; arithmetic/code/cloze fail — expected, model memorized seed corpus but did not generalize). |
| Limitations | Eval tasks are heuristics, not validated benchmarks. Corpus scan is local-only; network ingestion (Internet Archive, Wikipedia, Gutenberg) recorded as candidate sources but not fetched by default. No semantic dedup, no OCR, no malware scanning yet. |
| Decision | Ship v0.2 eval+corpus; next: benchmark harness (throughput/latency/RAM), then architecture ablations and scaling. |

## Adoption policy

Every technique is a hypothesis to benchmark. Adopt only when measured
evaluation justifies it: quality per token, quality per second, memory
efficiency, stability, reproducibility, or inference speed. Distinguish
established methods from experimental/unverified ideas.
