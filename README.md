# HELIX

An open-source, research-grade AI system engineered from the ground up, trained
from random initialization, CPU-first, with zero-budget-first development.

## Status: minimal end-to-end baseline

This is the first reproducible baseline of HELIX. It contains:

- A from-scratch byte-level BPE tokenizer (`helix/tokenizer.py`)
- A decoder-only transformer (RoPE, RMSNorm, SwiGLU, grouped-query attention, weight tying) (`helix/model.py`)
- A data pipeline that builds train/val token bins from a seed corpus (`helix/data.py`)
- A CPU training loop with gradient accumulation, checkpoint save/resume, deterministic seeds, and JSON logging (`helix/train.py`)
- Greedy/temperature sampling with a CLI chat interface (`helix/infer.py`, `helix/cli.py`)
- Tests and documentation (`tests/`, `docs/`)

Everything is implemented independently; open-source projects were used only as
inspectable references. No pretrained model weights are used as a foundation.

## Quick start (CPU)

```bash
# install dependencies
pip install -r requirements.txt

# train the tiny baseline from random init
python3 -m helix train --config configs/helix-tiny.json

# generate text
python3 -m helix generate --prompt "The helix is" --max-new-tokens 60

# interactive chat
python3 -m helix chat

# run tests
python3 -m pytest tests/
```

## Design principles

- **Zero-budget-first:** every core workflow runs on local CPU with free software.
- **From random initialization:** HELIX learns from its own training; pretrained
  weights are never silently substituted.
- **Documented & reproducible:** configs, seeds, checkpoints, dataset manifests,
  model cards, and a research ledger are kept alongside the code.
- **CPU-capable, GPU-ready:** ordinary CPUs first; GPU/distributed paths are
  optional and do not break the CPU baseline.

See `docs/ARCHITECTURE.md`, `docs/RESEARCH_LEDGER.md`, and `docs/MODEL_CARD.md`.
