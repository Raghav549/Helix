# Research ledger — measured CPU runs

This ledger supersedes older, unverified baseline notes. It records only the
included runs; small diagnostic tasks are not broad or contamination-proof
benchmarks. Raw configs, checkpoints, logs, and per-example results are under
`artifacts/`. Reproduction and dependencies: [REPRODUCIBILITY.md](REPRODUCIBILITY.md).

## Environment and corpus

On 2026-10-28: 2 available CPU threads, 3.85 GiB RAM, Python 3.11, PyTorch
`2.5.1+cu124`, CUDA unavailable. PyPI's wheel pulled CUDA-associated packages,
but all recorded experiments used CPU. `artifacts/environment-freeze.txt`
records installed versions. The 102,879-byte reviewed corpus has nine sources,
462/52 training/validation chunks and a train-only 512-token BPE vocabulary.
The ordered paragraph holdout is not source-independent.

## Language and instruction tuning

| Hypothesis / comparison | Actual experiment | Result / decision |
| --- | --- | --- |
| A small independently initialized transformer can train end to end on CPU | `configs/helix-research.json`; 803,712 parameters; 1,800 steps × 8 × 128 = 1,843,200 tokens | 90.10 s; best sampled validation loss **3.381563**. Keep as a reproducible baseline, not a useful general assistant. `artifacts/llm/metrics.json` |
| Re-evaluation on a fixed sample of the same held-out split | 30 held-out batches, batch 8, block 128, seed 1337 | Base **3.3845 loss / 29.50 perplexity**; SFT **3.4640 / 31.95**. `artifacts/{llm,sft}/eval.json` |
| Separate response-masked SFT | 48 authored pairs, 44 train/4 held out, 600 steps, batch 6 | Holdout loss **4.6907 initial**, **4.3319 best at step 25**, **5.4646 final**; best selected, clear overfitting. `artifacts/sft/sft-metrics.json` |
| Basic model behavior | Five tiny original probes per checkpoint | Both pass only the true-vs-false likelihood comparison (1/5). Arithmetic, isolated code completion, capital cloze and long-context recall fail. The 900-token recall case exceeds the 128-token window. This is not statistical calibration or an external benchmark. |

The token budget is the number processed, not unique dataset tokens. The split,
small source mixture, and four-example SFT holdout are strong limitations; the
model frequently gives nonsensical output. No modern pretrained-model baseline,
safety, multilingual, fairness, or human-quality evaluation was run.

## Architecture and inference comparisons

`configs/helix-compare.json`: random-init transformer, linear attention and
recurrent variants trained for 120 steps on identical batches, **30,720 tokens
per variant**. Fixed validation windows: transformer loss **5.429923**, linear
**5.441763**, recurrent **5.460378**. Transformer/linear had 155,968 parameters,
recurrent 180,416; wall compute differed (1.022/1.741/4.590 s). This is a smoke
comparison, not a compute-matched or universal architecture ranking.
`artifacts/ablation/report.json` contains all curves and parameters.

The serial trained-checkpoint benchmark (`artifacts/llm/bench.json`) measured
forward/backward throughput of **15,951.65 tokens/s** at block 64 and
**20,039.16 tokens/s** at block 128 (batch 4); process peak RSS was **594.23 MB**
at block 128. On one 32-token greedy prompt, cached output matched full
recomputation and was **1.16×** as fast; int8 output also matched that prompt
but was **0.92×** full-recompute speed. These are host- and prompt-specific;
match on one prompt does not prove lossless quantization.

## Synthetic image generation and understanding

- Attribute CNN: 93,560 randomly initialized parameters, 240 training / 90
  held-out renderer scenes, 300 steps; **90/90** all color/shape/position
  labels correct on those scenes. Confidence is uncalibrated; no natural-photo
  evaluation. `artifacts/vision/metrics.json`.
- ViT/projector + HELIX LM captioner: 875,584 total parameters; random vision
  initialization plus the HELIX base checkpoint; 240/90 synthetic scenes, 800
  steps, batch 16. Best held-out loss **0.06094** at step 700 vs **4.78524**
  before training; greedy captions exact on **64/90 (71.1%)**, with 90 color,
  83 shape, 65 position labels correct. The held-out set is still synthetic and
  used for selection. `artifacts/caption/{metrics,eval}.json`.
- VAE + text-conditioned diffusion: 134,815 randomly initialized parameters,
  240/60 synthetic scenes, 600 VAE + 3,000 denoiser steps. Fixed-noise held-out
  denoising loss **1.108509 → 0.115961**, reconstruction MSE **0.151162 →
  0.011181**; not perceptual quality or text adherence. The saved samples are
  blurry 32×32 images. `artifacts/image-gen/metrics.json`, `sample-*.png`.

### Physics-informed synthesis comparison

For 18 fixed synthetic scenes, pixel MSE to the renderer in [0,1] RGB was
**0.027668 learned-only** and **0.003088 renderer-guided**. The guided model
receives the reference image at strength 0.25 and therefore has extra
information. The attribute classifier is out of its validated domain on
synthetic model outputs. Neither MSE nor classifier scores establish realism,
quality on natural images, or a fair text-only advantage. Full results and
limitations: `artifacts/image-gen/physics-comparison.json`.

## Final verification (2026-10-28)

- The complete Python suite passed **130 tests** (14 warnings); `pip check`,
  wheel construction/asset inclusion and release-file checksum verification
  succeeded on the recorded environment.
- A clean corpus/tokenizer rebuild reproduced the retained vocabulary, merges,
  train and validation arrays byte-for-byte. Separate short CLI smoke runs
  performed 3 language steps and 2 steps each of SFT, vision, image and
  caption training; these checks are not the reported full experiments.
- The live UI passed jsdom and headless Chromium 153 smoke tests covering
  chat, rendering, image analysis/editing, approval-gated execution, math,
  plotting and mobile layout. The test browser used `--no-sandbox` in this
  isolated environment; HELIX's separate code sandbox remained enabled.

## Security and evaluation decisions

The web UI defaults to loopback and has no authentication; origin checks do not
make network exposure safe. Python execution requires approval and fail-closed
Linux namespace/chroot/seccomp isolation with resource bounds. It shares the
host kernel and is not a dedicated VM. Math uses a separate AST-validated,
resource-limited SymPy worker. Tests cover denied sockets, host filesystem
access, forks, timeouts, payload-specific approvals, memory consent, origin
checks and real checkpoints; they cannot prove absence of vulnerabilities.

References informing independent implementations include RoPE/RMSNorm/GQA,
DDPM and latent diffusion, ViT/vision-to-language projection, and the MIT
[nanoGPT](https://github.com/karpathy/nanoGPT) and BSD-style
[SymPy](https://github.com/sympy/sympy) projects. Third-party source material
used as data is pinned and separately attributed in [PROVENANCE.md](PROVENANCE.md).
