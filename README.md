# HELIX

HELIX is an open-source, CPU-first AI research workspace: a trainable language
model, image synthesis and understanding experiments, a coding tool, and a
math/data lab in one local interface. The included checkpoints were trained by
HELIX on CPU; no external pretrained model weights or paid services are used.

**This is a small research prototype, not a general-purpose assistant.** The
language model frequently produces incorrect or incoherent answers. The learned
image generator produces blurry 32×32 images. Its visual models were evaluated
only on synthetic shapes, not natural photographs. Verify all outputs and do not
use them for safety-critical decisions. Robotics is outside the project scope.

## Run locally

From the repository root on Linux with Python 3.11, `unshare`, `mount`, and
libseccomp available:

```sh
./scripts/setup.sh                   # installs CPU PyTorch first, then dependencies
. .venv/bin/activate
python -m helix doctor               # checks hardware and whether isolation works
python -m helix web                  # http://127.0.0.1:8766
```

The setup uses the official PyTorch CPU wheel index. If that index is unreachable,
`HELIX_TORCH_SOURCE=pypi ./scripts/setup.sh` uses PyPI instead; on Linux this
may download several gigabytes of CUDA-related packages even when running on CPU.
Python dependencies and the optional PDF reader are installed locally. See
[reproducibility](docs/REPRODUCIBILITY.md) for exact versions and commands.

**The web server has no authentication.** Its default loopback binding is for a
single trusted user. Anyone who can reach it can read saved notes and invoke
its APIs, including approved-code execution; origin checks are not a substitute
for authentication. Do not expose it to an untrusted network or enter secrets
into a shared preview. `--host 0.0.0.0` is appropriate only for a trusted,
controlled preview. The Flask development server is not a production service.
When a trusted TLS-terminating proxy does not forward the original scheme,
pass its exact browser origin, for example `--external-origin https://preview.example`.
This only adjusts origin checks; it does not add authentication.

## What's working

- **Language:** byte-level BPE, a 3-layer/803,712-parameter decoder-only
  transformer, checkpoint/resume, response-masked instruction tuning, sampling,
  streaming chat, KV cache with exact window re-priming, and optional int8
  experiments. Chat history stays in the browser session. Memory is opt-in,
  editable and deletable; saved notes are stored **unencrypted** on the host.
- **Images:** a tiny text-conditioned VAE/diffusion model trained from random
  initialization, image-to-image edits, and a separate direct-light renderer
  for a single sphere or cube. The guided mode gives the model a rendered
  reference. Upload analysis includes pixel measurements, a synthetic-domain
  classifier, and a ViT-to-HELIX captioner; neither learned visual model is
  validated on photographs.
- **Coding:** Python AST analysis, unverified model explanations, test-skeleton
  generation, diff previews, and execution requiring explicit approval. Approved
  execution fails closed unless Linux namespace/chroot/seccomp isolation is
  available. Repository writes in the Python API require approval and a matching
  source hash; the web interface only previews diffs.
- **Research:** bounded symbolic calculations using SymPy, CSV summaries and
  paired-column plots, and explicitly saved local experiment records. SymPy and
  Matplotlib are external open-source tools, not HELIX model predictions.
- **Data:** source manifests with licenses, provenance, hashes, quarantine,
  duplicate and limited contamination checks, and train-only tokenizer fitting.
  Unknown rights are not treated as permission to train.

## Included measurements

Measured on a 2-thread CPU with 3.85 GiB RAM, without a GPU. Details and
limitations are in the [research ledger](docs/RESEARCH_LEDGER.md); raw reports
and checkpoints are in `artifacts/`.

| Experiment | Result | Scope |
| --- | --- | --- |
| Language pretraining | 1,843,200 tokens; best sampled validation loss 3.3816 | 102,879-byte mixed-source corpus, 128-token window |
| Seeded validation | loss 3.3845, perplexity 29.50 | 30 sampled held-out batches; not a broad benchmark |
| Language / SFT probes | 1 of 5 passed for each | Only a true-vs-false likelihood comparison passed |
| Visual classifier | 90/90 held-out scenes correct on all 3 attributes | Synthetic renderer domain only |
| ViT-to-language captioner | 64/90 exact captions (71.1%) | Same synthetic domain; no natural-photo evaluation |
| Diffusion | held-out denoising loss 1.1085 → 0.1160 | Synthetic scenes; not perceptual quality or prompt fidelity |

The four-pair SFT holdout worsened by the end of training; the earlier best
checkpoint was selected. The synthetic guided image comparison and its
confounding reference input are described in the ledger. No comparison with a
modern large pretrained model or claim of photorealism is made.

## Useful commands

```sh
python -m helix --help
python -m helix chat
python -m helix generate --prompt 'Explain recursion' --kv-cache
python -m helix research --operation differentiate --expression 'x**3 + 2*x'
python -m helix scene-render --prompt 'a blue sphere on the left' --output out/scene.png
python -m helix image-analyze --image out/scene.png --caption
python -m helix code-run --file example.py --approve
python -m pytest -q
```

Run from the repository root: configurations contain relative data paths. The
prebuilt artifacts are versioned separately from `out/`, which holds local runs,
memory, generated images and audits. Only load checkpoints from trusted sources.
The Python agent audit log may contain raw inputs; the web audit records metadata
and input hashes rather than raw prompts or code, but hashes can still leak
information through guessing.

## Documentation and licenses

- [Architecture](docs/ARCHITECTURE.md) · [Data pipeline](docs/DATA_PIPELINE.md)
- [Model cards](docs/MODEL_CARD.md) · [Research ledger](docs/RESEARCH_LEDGER.md)
- [Reproduce the experiments](docs/REPRODUCIBILITY.md) · [Provenance and licenses](docs/PROVENANCE.md)

HELIX-authored code and weights are MIT-licensed; mixed-source training data
retain their original notices and terms. See `LICENSE`,
`artifacts/WEIGHTS_LICENSE.txt`, and [the source inventory](docs/PROVENANCE.md).


## Render test deployment

A Render Free CPU web-service test deployment was created from branch
`fix/ci-render-readiness` at:
https://helix-chat.onrender.com
Dashboard: https://dashboard.render.com/web/srv-dbh43gvlk1mc73evf1ig

This is an experimental preview, not a production-safe public assistant:
- The service process is live, but the bundled model checkpoints are stored under
  `artifacts/` and `.gitignore` excludes most generated artifact files. The chat
  endpoint must be tested before considering this a working deployed chatbot.
- The web app has no authentication; do not submit secrets, sensitive personal
  information, or untrusted code.
- Render Free services have ephemeral local storage and can sleep while idle.
- Full-corpus training and long-running model training are not run by the web
  service. Training needs a separate, resumable compute job.
- The latest passing public end-to-end deployment check has not yet been
  recorded. Service liveness is not proof that the HELIX checkpoint loads or
  generates a reply.
