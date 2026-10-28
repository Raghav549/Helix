# Architecture and boundaries

## Language

`tokenizer.py` learns a byte-level BPE vocabulary from training documents only.
`model.py` implements a causal decoder transformer with rotary positions,
RMSNorm, SwiGLU, grouped-query attention and tied embeddings. The released
configuration has three layers, four query heads, two KV heads, embedding width
128 and a 128-token window. `train.py` uses AdamW, warmup/cosine scheduling,
gradient accumulation, validation checkpoints, RNG/optimizer/scaler state, and
CPU quota-aware threading. Resume equivalence is tested for a fixed CPU setup.

`infer.py` supports full recomputation, KV-cached incremental inference and
streaming UTF-8 decoding. When the window fills, it recomputes the sliding window
rather than retaining stale keys/values. Other experimental block types use
full recomputation. INT8 is an experiment, not a guaranteed speedup. `sft.py`
performs a separate response-only supervised stage on HELIX-authored pairs.

## Images and visual understanding

`physics.py` is a deterministic pinhole/direct-light renderer for one sphere or
cube, simple material and hard shadow. It is useful for controlled data and
experiments, not a general scene renderer or physical simulation.

`image_gen.py` implements a small VAE, latent denoiser, noise schedule, trainable
byte/GRU prompt encoder, approximate reverse sampler and image-to-image path.
The prompt encoder uses the first 128 UTF-8 bytes; the included image architecture
uses fixed block counts/downsampling and float32, rejecting unsupported settings.
`image_workflow.py` supplies synthetic data and a reviewed local image-manifest
path. `visual.py` trains a CNN for color, shape and position. Separately,
`image_understanding.py` provides a ViT and projector into the HELIX language
model; `caption_workflow.py` trained and evaluated this ViT-to-language model
using a randomly initialized vision encoder and HELIX's own language checkpoint.
The UI exposes both the classifier and this learned captioner. Neither is
validated on general photographs.

## Coding and safety

`coding.py` has bounded repository inspection, Python AST analysis, generated
callable/import test skeletons, diff previews and an approval/hash-guarded write
API. A hash check is not an atomic multi-user transaction. Generated tests are
not evidence of behavioral correctness unless independently implemented and run.

`agent.py` provides a small planning/tool loop. Consequential tool calls require
a single-use grant tied to the exact tool input; untrusted documents cannot
grant permissions. There is no guarantee that the tiny model follows task
instructions or resists all prompt injections. Step/time limits are checked
between calls, not a preemptive deadline for a running model forward pass.

`sandbox.py` and `_sandbox_runner.py` execute Python once with Linux user, mount,
PID and network namespaces, a minimal chroot, read-only system Python, dropped
capabilities, seccomp, rlimits and a wall-clock timeout. If setup fails, code is
not run. This shares the host kernel and is **not** a VM or a guarantee against
kernel vulnerabilities; use a dedicated VM for hostile workloads. The sandbox
requires compatible system Python and Linux facilities. The web execution route
and `CodingAgent` require explicit approval; `run_python` itself is a low-level
execution primitive and callers must gate it.

## Research and web

`research.py` implements statistics, CSV summaries/plots and experiment
records; `math_worker.py` runs validated SymPy operations in a resource-limited
child process. The math worker is not the Python execution sandbox. SymPy is an
independent computation tool, not a proof of correctness for every expression.

`web.py` serves `helix/static/` and relative JSON/SSE endpoints with lazy loading,
bounded requests, same-origin checks and metadata-only web audit events. An
explicit `--external-origin` supports trusted TLS-terminating proxies that do
not forward the browser scheme; it is an exact origin, not authentication. The
server has **no authentication**. Its default loopback binding is important;
network reachability grants access. Generated images, optional records, and
unencrypted SQLite memories are stored under `out/` on the server host.

## Layout

`helix/`: source; `configs/`: runnable configurations; `data/`: source data and
manifest; `artifacts/`: released checkpoints, reports and licenses; `out/`:
ignored local runtime data; `tests/`: automated checks. The tested environment
is Linux CPU/Python 3.11; accelerators and distributed training are not claimed
as tested. Details and numeric results: [research ledger](RESEARCH_LEDGER.md).
