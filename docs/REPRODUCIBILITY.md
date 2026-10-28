# Reproducibility

All commands below run from the repository root. `artifacts/` contains the
recorded outputs; use `out/` for new runs so that the included checkpoints are
not overwritten. The recorded machine had 2 CPU threads, 3.85 GiB RAM, Python
3.11 and PyTorch 2.5.1; no accelerator was available. Runtime, exact losses and
bitwise results can differ across OS, PyTorch versions, thread counts and CPUs.

## Environment and tests

```sh
./scripts/setup.sh                     # official CPU PyTorch wheel first
# Or HELIX_TORCH_SOURCE=pypi ./scripts/setup.sh when the CPU index is unreachable.
. .venv/bin/activate
export HELIX_THREADS=2 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1
python -m helix doctor
python -m pytest -q
```

`requirements.txt` pins direct packages; `artifacts/environment-freeze.txt`
records the actual installed transitive packages but includes PyPI
CUDA-associated packages and is not a universal lockfile. The setup also pins
optional `pypdf`. Linux namespace, mount, seccomp and system-Python availability
vary by host. The `doctor` command returns nonzero when the isolated executor
cannot be established; execution fails closed. The rest of the CPU platform
can still be used.

## Rebuild corpus and tokenizer; train a fresh language model

The source files are already checked in with notices. The following rebuild
writes a separate manifest/corpus; the recorded corpus SHA-256 is
`97a20dc524825f488f9c5df6c203f6bad4e3af827039491e643fc0b5764ead69`.

```sh
mkdir -p out/reproduce
python scripts/prepare_corpus.py --manifest out/reproduce/manifest.json --output out/reproduce/corpus.txt
sha256sum out/reproduce/corpus.txt
python - <<'PY'
import json
from pathlib import Path
c = json.loads(Path('configs/helix-research.json').read_text())
c['train']['out_dir'] = 'out/reproduce/llm'
for key, path in {'corpus_file': 'out/reproduce/corpus.txt',
                  'train_file': 'out/reproduce/data/train.npy',
                  'val_file': 'out/reproduce/data/val.npy',
                  'vocab_file': 'out/reproduce/data/vocab.json',
                  'merges_file': 'out/reproduce/data/merges.json'}.items():
    c['data'][key] = path
Path('out/reproduce/llm-config.json').write_text(json.dumps(c, indent=2))
PY
python -m helix train --config out/reproduce/llm-config.json
python -m helix eval --config out/reproduce/llm-config.json \
  --checkpoint out/reproduce/llm/ckpt-best.pt --eval-iters 30 --seed 1337 \
  --out-path out/reproduce/llm-eval.json
```

The config uses seed 1337, 1,800 steps, batch 8 and block 128. The first run
builds the tokenizer from training chunks. To resume, set `train.resume_from`
in a copy of the config and increase `max_iters`, retaining the intended
learning-rate schedule; checkpoints include optimizer and RNG states.
Provenance metadata rejects changed prepared files/corpus. Test coverage checks
that a fixed-schedule interrupted run matches an uninterrupted run.

## Instruction tuning and model diagnostics

```sh
python -m helix sft --config out/reproduce/llm-config.json \
  --base-checkpoint out/reproduce/llm/ckpt-best.pt --sft-pairs data/sft_pairs.json \
  --out-dir out/reproduce/sft --max-iters 600 --block-size 128 --batch-size 6 --lr 0.0002
python -m helix eval --config out/reproduce/llm-config.json \
  --checkpoint out/reproduce/sft/sckpt-best.pt --eval-iters 30 --seed 1337 \
  --out-path out/reproduce/sft-eval.json
python -m helix bench --config out/reproduce/llm-config.json \
  --checkpoint out/reproduce/llm/ckpt-best.pt --block-sizes 64,128 \
  --out-path out/reproduce/bench.json
```

The evaluation tasks are small probes, not established benchmarks. Timings must
be measured serially on a quiet machine and should not be compared as universal
performance claims. For the architecture comparison, copy
`configs/helix-compare.json`, change `train.out_dir` to an `out/` path, then run
`python -m helix ablate --config <copy> --out-path out/reproduce/ablation.json`.
Its variants use the same token batches but different parameter/compute counts.

## Image and visual experiments

```sh
python -m helix vision-train --out-dir out/reproduce/vision \
  --count 240 --val-count 90 --steps 300 --batch-size 24 --seed 1337
python -m helix image-train --synthetic --config configs/image-gen-small.json \
  --out-dir out/reproduce/image-gen --count 240 --val-count 60
python -m helix caption-train --base-checkpoint out/reproduce/llm/ckpt-best.pt \
  --out-dir out/reproduce/caption --steps 800 --batch-size 16
python -m helix visual-eval --config out/reproduce/image-gen/config.json \
  --image-checkpoint out/reproduce/image-gen/ckpt-final.pt \
  --vision-checkpoint out/reproduce/vision/ckpt-final.pt \
  --out-path out/reproduce/physics-comparison.json --seed 2026
```

Caption training regenerates the same renderer scenes and uses the included
vocabulary/merges (rebuilding the corpus with the same inputs yields the same
vocabulary). Vision and diffusion are randomly initialized; the captioner uses
an independently initialized ViT/projector and the specified HELIX language
checkpoint. Guided synthesis receives a renderer reference and must not be
presented as a fair text-only comparison. No natural-photo benchmark is included.

## Optional UI smoke tests

Start `python -m helix web` in another terminal, then install test-only Node
packages under the ignored `out/` directory. Use only a trusted server: the DOM
smoke test creates/deletes its own unique memory note and runs an approved code
snippet. The browser test needs a locally installed Chromium executable; it uses
real interactions and saves screenshots under `out/`.

```sh
npm install --prefix out/ui-test jsdom@24.1.3 puppeteer-core@25.16.0
NODE_PATH=out/ui-test/node_modules node tests/ui_smoke.cjs
NODE_PATH=out/ui-test/node_modules HELIX_BROWSER=/path/to/chromium node tests/browser_smoke.cjs
```

Both default to `http://127.0.0.1:8766/` and accept `HELIX_URL`. The optional
`HELIX_BROWSER_NO_SANDBOX=1` flag disables the **browser's** sandbox, and
should only be used in an isolated test environment, never for untrusted sites;
it does not change HELIX's separate fail-closed code sandbox.

## External storage and initial corpus

Set `HELIX_DATA_ROOT` to a directory outside the repository before running
training or the web server. Relative `artifacts/...` paths in loaded
configurations resolve below that root. Direct `DataConfig(...)` constructors
used by tests and custom scripts retain the path they receive; only
`load_config()` applies the external-root rule.

Run the verified release migration with:

```sh
HELIX_DATA_ROOT=/media/helix/data python scripts/migrate_artifacts.py
HELIX_DATA_ROOT=/media/helix/data python scripts/migrate_artifacts.py --check
```

The initial source pipeline reads the MIT-licensed HELIX seed corpus from the
repository, copies it to external `raw/` storage without a network request,
filters benchmark names, deduplicates and writes deterministic splits:

```sh
HELIX_DATA_ROOT=/media/helix/data python -m helix sources
```

This does not download third-party material. The catalog currently has one
deterministic six-topic seed, so a rerun writes the same `corpus/train.txt`,
`val.txt` and `test.txt` names and their SHA-256 files. Preserve a prior run
by copying the whole `corpus/` directory before rerunning; the external-root
layout intentionally keeps these generated files out of Git.
