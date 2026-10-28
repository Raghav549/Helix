# HELIX instruction-tuning data

The CPU training workflow creates its SFT dataset under `HELIX_DATA_ROOT/out/reproduce/`; generated examples and model checkpoints are not committed to Git. It starts with the reviewed, authored `data/sft_pairs.json` set and adds deterministic, locally verifiable practice tasks for arithmetic, basic Python functions, simple logic, and conversation behaviour.

The generator records a seed, the source-pair SHA-256, requested counts, actual post-deduplication counts, and total examples in `sft-pairs-manifest.json`. The generated data is an auxiliary training signal, not evidence that HELIX has learned broad factual knowledge or advanced reasoning. It must not be described as human-authored or as an external corpus.

SFT uses held-out examples and early stopping. The best checkpoint is selected by held-out loss, not by training loss. Training metrics and the generated-data manifest are attached to the versioned model release so regressions and overfitting can be audited. The benchmark suite remains separate from the SFT training pairs.

To regenerate locally:

```bash
export HELIX_DATA_ROOT=/path/to/external/helix-data
python scripts/generate_sft_pairs.py \
  --base data/sft_pairs.json \
  --output "$HELIX_DATA_ROOT/sft-pairs.json" \
  --manifest "$HELIX_DATA_ROOT/sft-pairs-manifest.json" \
  --seed 1337
```

Use only approved source data in the base file. Do not add private conversations, unlicensed text, or benchmark answers to training data.
