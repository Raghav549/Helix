# Recorded HELIX artifacts

These are the 2026-10-28 CPU experiments described in
`../docs/RESEARCH_LEDGER.md` and `../docs/MODEL_CARD.md`.

- `data/`: assembled mixed-source corpus, tokenizer and token arrays.
- `llm/`: random-init base model, training/evaluation/benchmark records.
- `sft/`: HELIX-base instruction tuning and held-out diagnostics.
- `vision/`: synthetic-domain attribute classifier.
- `caption/`: synthetic-domain ViT-to-HELIX captioner and per-example evaluation.
- `image-gen/`: diffusion/VAE checkpoint, samples and physics comparison.
- `ablation/`: three small architecture-comparison checkpoints and report.
- `environment-freeze.txt`: recorded dependency versions.
- `SHA256SUMS`: checksums for retained artifacts, inputs, and configurations;
  verify from the repository root with `sha256sum -c artifacts/SHA256SUMS`.

`WEIGHTS_LICENSE.txt` applies to HELIX-authored weights. The mixed corpus and
underlying sources retain their separate licenses and notices documented in
`../docs/PROVENANCE.md`; it is not a blanket MIT-only dataset. Checkpoints should
only be loaded from trusted sources. These models are experimental and are not
validated as general-purpose, safe, or photorealistic.
