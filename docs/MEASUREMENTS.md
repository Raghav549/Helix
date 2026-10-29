# HELIX measurements

Every number on this page was produced by this checkout on the hardware below. The
raw JSON is committed next to this file in [`docs/measurements/`](measurements/) so the
tables can be re-derived instead of trusted.

Nothing here is an aggregate of published benchmarks, and nothing here is a claim of
general assistant capability. The models are **randomly initialised, then trained for
minutes on a small documented corpus**. They are measured honestly and they are weak.

## Run environment

| Item | Value |
| --- | --- |
| CPU | 2 vCPU (logical), AVX-512F/AVX2/AVX/SSE4.2 |
| RAM | 3.85 GB |
| Accelerator | none (`has_cuda: false`) |
| Threads | `torch.set_num_threads(2)` |
| Python / PyTorch / NumPy | 3.11.2 / 2.5.1+cu124 (CPU) / 1.26.4 |
| HELIX version | 0.13.0 |
| Training commit | `14c20bccd26a7c316a56048e9600705de2131aba` |
| Seed | 1337 (all stages) |

## Model under test

| Property | Value |
| --- | --- |
| Parameters | 803,712 (≈0.8 M) |
| Architecture | 3 transformer layers, 4 query heads / 2 KV heads (GQA), n_embd 128, RoPE, weight tying |
| Context (block size) | 128 tokens |
| Vocabulary | 512 tokens, byte-level BPE trained on the run corpus |
| Precision | float32 (CPU); int8 tried for generation, see below |
| Checkpoint format | `.pt` state dict + JSON config, `ckpt-best.pt` / `ckpt-final.pt` |

## Data

| Property | Value |
| --- | --- |
| Sources | 9 documented files (HELIX seed corpus, nanoGPT README, and other MIT-permitted text) |
| Corpus bytes | 102,879 |
| Corpus sha256 | `97a20dc524825f488f9c5df6c203f6bad4e3af827039491e643fc0b5764ead69` |
| Tokenised train / val | `train.npy` `2f5028…`, `val.npy` `1ec65d…` (sha256 recorded in provenance) |
| Split | Held-out validation slice carved by offset, never trained on |
| Provenance | `manifest.json` per source: licence, retrieval time, content hash, attribution |

The training corpus is small and fully documentable. No scraped web text, no private
data, nothing whose licence is unknown.

## Stage 1 — pretraining (200 steps)

| Metric | Value |
| --- | --- |
| Iterations | 200 (batch 8 × block 128) |
| Tokens processed | 204,800 |
| Wall clock | 12.53 s |
| Throughput | 16,340 tokens/s |
| Peak RSS | 697.9 MB |
| Best validation loss | 4.78095 |
| Final grad norm | recorded in `pretrain-metrics.json` |
| Skipped steps (non-finite) | 0 |

## Stage 2 — supervised fine-tuning

| Metric | Value |
| --- | --- |
| Pairs generated | 516 (math, code, logic, conversation) |
| Train / held-out | 465 / 51, split by pair hash so a task family cannot appear on both sides |
| Iterations | 60 |
| Initial held-out loss | recorded in `sft-metrics.json` |
| Final held-out loss | 3.5044 |
| Wall clock | 6.90 s |
| Pairs sha256 | `14fc8038…` (pairs file is hashed into the metrics) |

## Stage 2b — instruction-tuning scaling (and why the default stops at 60 steps)

Same base checkpoint, same 516 pairs (465 train / 51 held-out), same seed, only the number
of SFT steps changed. Every evaluation uses 50 sampled held-out batches so the columns are
comparable.

| SFT steps | train loss | SFT held-out loss | validation loss | perplexity | probes passed |
| --- | --- | --- | --- | --- | --- |
| 60 (default) | 3.3516 | 3.5044 | 4.977 | 145.0 | 1/5 |
| 300 | 2.9000 | 3.1111 | 5.172 | 176.2 | 1/5 |
| 1000 | 2.2669 | 2.5241 | 5.857 | 349.7 | 1/5 |

More instruction tuning keeps improving the instruction-format held-out loss while the
general validation loss gets steadily worse and no capability probe changes. At this model
size that is overfitting to a narrow synthetic instruction distribution, so the default
checkpoint stays at 60 steps and the longer runs are recorded as a negative result rather
than shipped. Raw data: [`docs/measurements/sft-scaling.json`](measurements/sft-scaling.json).

## Stage 3 — evaluation on the SFT checkpoint

Three of the five tasks are deliberately hard for a 0.8 M-parameter model, and the
results say so:

| Task | Category | Metric | Value | Threshold | Passed |
| --- | --- | --- | --- | --- | --- |
| arithmetic_addition | reasoning | accuracy | 0.00 | 0.50 | ✗ |
| code_completion_is_even | code execution | isolated test pass | 0.00 | 1.00 | ✗ |
| sentence_completion_capital | memorisation | accuracy | 0.00 | 1.00 | ✗ |
| long_context_recall | long context | accuracy | 0.00 | 1.00 | ✗ |
| calibration_true_vs_false | calibration | true loss < false loss | 1.00 | 1.00 | ✓ |

Validation on the same checkpoint: **loss 4.9701, perplexity 144.04**, over 30 sampled
held-out windows (batch 8, block 128, seed 1337).

Calibration is the only passing signal, and it is the weakest possible one: the model
assigns slightly lower loss to true statements than to their negations
(5.2286 vs 5.2590). Arithmetic completions are repetitions of the end-of-text token, so
free-form answers are not usable. That is why user-facing answers for math, forecasting
and retrieval come from tools, which are evaluated separately and label themselves.

## Stage 4 — throughput and memory

Training step (forward + backward), batch 4:

| Block size | Tokens/s | ms/iter | Peak RSS |
| --- | --- | --- | --- |
| 64 | 12,194 | 20.99 | 525 MB |
| 128 | 15,271 | 33.53 | 586 MB |

Generation (32 new tokens, greedy):

| Mode | ms/token | Tokens/s | Output identical to full recompute |
| --- | --- | --- | --- |
| Full recompute | 2.015 | 496 | baseline |
| KV cache | 1.718 | 582 | yes |
| int8 quantised | 2.059 | 486 | no speedup on this CPU |

KV caching gives a **1.17×** speed-up and byte-identical output. int8 gives **0.98×** —
i.e. no win — and is therefore not enabled by default. This is exactly the kind of
change that must be measured rather than assumed.

Serving latency observed through the live local server (same machine, model already
loaded): first SSE byte after model load, 24 tokens in ≈1.2 s at the default sampling
settings, and `/api/agent` symbolic math answered in ≈0.6 s including the tool call.

## Stage 5 — retrieval quality

The retriever (`helix/retrieval.py`) is deliberately plain: stopword-filtered token
overlap scored with BM25, blended 65/35 with a hashed bag-of-words cosine signal. No
embedding service, no network, so it can be measured and reviewed.

Measured with `scripts/eval_retrieval.py` on a hand-labelled set of 12 questions over the
repository's own documentation (11 documents, the `docs/measurements/` reports excluded, `k = 4`):

| Metric | Value |
| --- | --- |
| recall@1 | 0.75 (9/12) |
| recall@4 | 0.83 (10/12) |
| MRR | 0.771 |
| Citations resolving to a real file and a valid character range | 12/12 |
| Misses | *"how do I reproduce the training runs on a CPU"* (not in top 4) and *"where should runtime artifacts be stored"* (not in top 4) |

The caveat is part of the result: the questions were written after the documents, so they
share vocabulary with them; three cases accept more than one document because the fact
genuinely appears in several places. This is indicative of whether citation plumbing and
ranking work, not a semantic-search benchmark. Regenerate with:

```sh
python scripts/eval_retrieval.py --out docs/measurements/retrieval-eval.json
```

`tests/test_measurements.py` recomputes these numbers and fails if the committed report
drifts from the current documentation.

## Fortran kernels

`helix/native/kernels.f90` is compiled with `f2py` when a Fortran toolchain is present,
and `helix/numerics.py` falls back to the pure-Python implementation otherwise. In this
environment **gfortran is not installed**, so the pure-Python backend is what ran; the
benchmark reports `backend: "python"` rather than pretending otherwise. The kernels are
limited to small numeric routines (trend fitting, distance accumulation); the language
model is not written in Fortran.

## How to reproduce

All artifacts stay outside the repository (`HELIX_DATA_ROOT` must not resolve inside it),
so only these small JSON reports are committed.

```bash
export HELIX_DATA_ROOT=/tmp/helix-run          # outside the repo
mkdir -p "$HELIX_DATA_ROOT"

# 1. corpus from the repository's pinned, licence-checked sources
python scripts/prepare_corpus.py \
  --manifest "$HELIX_DATA_ROOT/source-manifest.json" \
  --output "$HELIX_DATA_ROOT/corpus.txt"

# 2. pretraining (tokeniser and .npy splits are derived from corpus.txt on first run)
cp configs/reproduce-cpu.json "$HELIX_DATA_ROOT/llm-config.json"
python -m helix train --config "$HELIX_DATA_ROOT/llm-config.json"

# 3. supervised fine-tuning on generated, held-out-split pairs
python scripts/generate_sft_pairs.py \
  --base data/sft_pairs.json \
  --output "$HELIX_DATA_ROOT/sft-pairs.json" \
  --manifest "$HELIX_DATA_ROOT/sft-pairs-manifest.json" \
  --seed 1337 --math-count 300 --code-count 48 --logic-count 80 --conversation-count 40
python -m helix sft \
  --config configs/reproduce-cpu-sft.json \
  --base-checkpoint "$HELIX_DATA_ROOT/llm/ckpt-final.pt" \
  --sft-pairs "$HELIX_DATA_ROOT/sft-pairs.json" \
  --out-dir "$HELIX_DATA_ROOT/sft" --max-iters 60

# 4. evaluation and benchmarking
python -m helix eval \
  --config "$HELIX_DATA_ROOT/llm-config.json" \
  --checkpoint "$HELIX_DATA_ROOT/sft/ckpt-best.pt" \
  --out-path "$HELIX_DATA_ROOT/sft-eval.json"
python -m helix bench \
  --config "$HELIX_DATA_ROOT/llm-config.json" \
  --checkpoint "$HELIX_DATA_ROOT/sft/ckpt-best.pt" \
  --out-path /tmp/helix-bench-report.json
```

`configs/reproduce-cpu.json` is the exact configuration behind the tables above
(model geometry plus training hyper-parameters, extracted from the run's own
`metrics.json`), with paths made relative to the data root. A four-step smoke run of that
config was executed during this work and reproduced the same parameter count (803,712)
and the same data split.

`metrics.json` records the git commit, corpus/train/val/vocab/merges hashes, environment
versions, skipped steps and peak RSS, so a run can be tied back to exact inputs.
`gfortran` is absent here, so the numeric benchmark reports `backend: "python"`; the
Fortran path is exercised only in an environment that provides `f2py`.

## What these measurements do not show

* They do not show fluent text generation. Loss ≈ 4.97 with a 512-token vocabulary is
  far from coherent English, and the evaluation tasks confirm it.
* They are single-run numbers on a 2-vCPU box; run-to-run throughput varies by a few
  percent and nothing here is a hardware benchmark.
* They say nothing about the quality of the tool outputs beyond the correctness tests in
  `tests/test_numerics.py` and `tests/test_orchestrator.py` (SymPy results are checked
  against exact closed forms).
