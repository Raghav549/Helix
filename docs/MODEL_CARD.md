# HELIX model cards — 2026-10-28 CPU experiments

All weights below originate in HELIX training runs; no external pretrained
weights were used. Trained checkpoints are research artifacts, not safety
certified. Source and dependency terms: [provenance](PROVENANCE.md).

## Language and instruction checkpoints

- `artifacts/llm/ckpt-best.pt`: independently initialized 803,712-parameter
  decoder-only transformer, 512-token byte BPE vocabulary, 128-token context.
  Trained for 1,800 steps / 1,843,200 tokens on the reviewed small corpus.
- `artifacts/sft/sckpt-best.pt`: same architecture initialized from the HELIX
  base checkpoint, then fine-tuned on 44 of 48 authored instruction pairs.
  Four pairs were held out for checkpoint selection. Best held-out response loss
  was 4.3319; it worsened to 5.4646 at step 600 (initial 4.6907).
- In the same seeded 30-batch base-language validation, base loss/perplexity was
  3.3845/29.50 and SFT was 3.4640/31.95. Each passed only the true-versus-false
  likelihood probe out of five small tasks. This is **not** a calibration,
  safety or broad capability benchmark. See `artifacts/{llm,sft}/eval.json`.
- Intended use: experimentation with training/inference infrastructure. Outputs
  may be nonsensical, incorrect, biased, unsafe or memorized. No medical, legal,
  financial, security-critical or unsupervised use. No safety/fairness audit,
  preference alignment or privacy guarantee.

## Image generation

`artifacts/image-gen/ckpt-final.pt` is a 134,815-parameter text-conditioned
latent diffusion system with a VAE, trained from random initialization on 240
HELIX-rendered scenes. It outputs 32×32 RGB images. On 60 held-out synthetic
scenes, fixed-noise denoising loss fell 1.1085 → 0.1160 and reconstruction loss
0.1512 → 0.0112; these are not human/perceptual scores. Samples are blurry and
prompt fidelity is unreliable. The physics-guided mode receives a rendered
reference; the renderer is a separate deterministic method, not a learned
photorealistic generator. No natural-image, fairness or content-safety study.

## Synthetic visual models

- `artifacts/vision/ckpt-final.pt`: 93,560-parameter CNN trained from random
  initialization for three attribute labels. It got all three labels right on
  90/90 held-out renderer scenes. Confidence values are uncalibrated.
- `artifacts/caption/ckpt-best.pt`: 875,584-parameter ViT/projector + HELIX
  language model. Vision and projection start randomly; the language part starts
  from the HELIX base checkpoint. After 800 training steps, the checkpoint
  selected at step 700 produced 64/90 exact synthetic captions (71.1%): color
  90/90, shape 83/90, position 65/90. This shared synthetic held-out set was
  used for development and is not a natural-photo benchmark.
- On arbitrary images, pixel statistics are real measurements but model labels
  and captions can be wrong or meaningless. Do not treat them as factual.

## Licensing and weights

HELIX-authored weights are distributed under the MIT terms in
`artifacts/WEIGHTS_LICENSE.txt`. The training corpus contains separately
licensed third-party text/code, and outputs can reproduce source content; retain
applicable notices and review source licenses before redistribution. Model
weights and source data are not an assertion of freedom from all downstream
rights obligations. Use only trusted checkpoints.
