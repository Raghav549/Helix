# HELIX

HELIX is an open-source, CPU-first AI research workspace and the code behind **Beta by
helix**: a trainable language model, a symbolic/numeric tool layer, retrieval over local
documents, image synthesis and analysis experiments, a coding sandbox, and web and
Android clients that expose them through one HTTP API. Everything runs from
configuration in this repository; the language and image checkpoints are trained by
HELIX itself on CPU, with no external pretrained weights and no paid inference service.

**What to expect.** HELIX is a small, honestly measured research system, not a
general-purpose assistant. Its 803,712-parameter model is trained from random
initialisation on a ~103 KB documented corpus, and it is evaluated rather than assumed:
validation loss 4.97, perplexity 144, and 1 of 5 capability probes passing (a
true-vs-false calibration check). Free-form answers are frequently incoherent. The parts
of the product that are reliable are the tools — SymPy mathematics, numeric forecasting,
sandboxed code execution and local-document retrieval — and every answer that comes from
a tool says which tool produced it. The learned image model produces blurry 32×32
samples and was validated only on synthetic shapes, never on photographs. Verify all
outputs; do not use them for safety-critical decisions.

## Architecture

```
                       ┌──────────────────────────────┐        ┌──────────────────────────┐
  browser  ──────────► │ Vercel: public/ static client│ ─────► │ api/[...path].js         │
                       │ (Beta by helix UI, no secrets)│        │ lib/helix-proxy.js       │
                       └──────────────────────────────┘        │ JSON/SSE, Basic auth,    │
                                                               │ timeouts, HTML-stripping  │
                                                               └───────────┬──────────────┘
                                                                           │ HTTPS + Basic
                                                                           ▼
                       ┌───────────────────────────────────────────────────────────────┐
  Android ───────────► │ Render: gunicorn (1 worker, 2 threads) → helix/wsgi.py →       │
  (mobile/, Expo)      │ helix/web.py (Flask factory)                                   │
                       │                                                                │
                       │  /api/chat, /api/chat/stream   ── helix/infer.py + helix/model  │
                       │  /api/agent                   ── helix/orchestrator.py          │
                       │        ├─ symbolic_math        ── helix/research.py (SymPy)     │
                       │        ├─ linear_forecast      ── helix/numerics.py (+Fortran)  │
                       │        ├─ document_search      ── helix/retrieval.py (hybrid)   │
                       │        └─ code_execution       ── helix/sandbox.py (namespaces) │
                       │  /api/image/*, /api/code/*, /api/research/*, /api/memory        │
                       │  /api/health, /api/status, /api/diagnostics, /healthz           │
                       └───────────────────────────────────────────────────────────────┘
                             │                                     │
                    out/ runtime data                   HELIX_DATA_ROOT (outside repo)
                    (audit, images, memory)             checkpoints, tokenised data, pairs
```

The Vercel layer is a dumb, defensive proxy: it never renders HTML as JSON, rejects
unknown `/api/*` routes itself, maps upstream timeouts to `504`, intercepts provider
error pages, forwards SSE without buffering, and adds a request id to every response.
The Flask layer owns the API contract and returns JSON for every `/api/*` path, including
404/405/413/429/500.

## Quick start (local)

Linux, Python 3.11, `unshare`, `mount` and libseccomp available for the sandbox:

```sh
./scripts/setup.sh                      # CPU PyTorch first, then the rest
. .venv/bin/activate
python -m helix doctor                  # hardware + isolation check
python -m helix web --config "$HELIX_DATA_ROOT/llm-config.json" \
                    --checkpoint "$HELIX_DATA_ROOT/llm/ckpt-best.pt"
# http://127.0.0.1:8766
```

`--config` and `--checkpoint` are the supported way to point the server at a trained
run; `HELIX_CHECKPOINT_PATH` is not read by `helix.web`. Without artifacts the server
still starts and reports `model_ready: false` instead of failing.

> **The local server has no authentication.** Its loopback binding is for one trusted
> user. Anyone who can reach it can read saved notes and call its APIs, including
> approved code execution. Do not expose it to an untrusted network. `--host 0.0.0.0`
> is only for a trusted, controlled preview; the Flask development server is not a
> production service.

## API

All routes are JSON; unknown `/api/*` paths return
`{"error": ..., "code": "route_not_found", "request_id": ...}` with status 404. Every
response carries `X-Request-ID` (an echoed client id when it is a safe token, otherwise
a generated one) and `X-API-Version`. `/api/v1/<route>` is an alias of `/api/<route>`.

| Method | Route | Purpose |
| --- | --- | --- |
| GET | `/healthz` | liveness only (no version, no paths, no secrets) |
| GET | `/api/health` | readiness: `model_ready`, `degraded`, `degraded_reason`, `checks` |
| GET | `/api/status` | capability matrix, loaded models, retrieval index, limitations |
| GET | `/api/diagnostics` | presence flags, counters, rate limits (auth required in production) |
| POST | `/api/chat` | one reply: `{message, history?, max_new_tokens?}` |
| POST | `/api/chat/stream` | same, as SSE: `meta` → `delta`* → `done`, `text/event-stream` |
| POST | `/api/agent` | classify → plan → permission → execute → validate; returns tool names, permissions, durations, input hashes |
| POST | `/api/research/math` | bounded SymPy operation (`differentiate`, `integrate`, `solve`, `simplify`, …) |
| POST | `/api/research/forecast` | least-squares trend with prediction interval, pure-Python/Fortran backend |
| POST | `/api/research/search` | hybrid retrieval over the configured local index with citations |
| GET | `/api/research/sources` | indexed documents, counts, hashes |
| GET/POST/DELETE | `/api/memory` | opt-in notes (stored unencrypted on the host) |
| POST | `/api/image/generate`, `/api/image/analyze`, `/api/image/edit` | physics renderer, vision tool, diffusion experiments |
| POST | `/api/code/analyze`, `/execute`, `/patch`, `/test-skeleton`, `/explain` | coding tools; execution needs approval and isolation |

Error bodies are stable: `code` is machine-readable
(`invalid_request`, `route_not_found`, `model_unavailable`, `compute_busy`,
`generation_failed`, `request_too_large`, `rate_limited`, `proxy_not_configured`,
`invalid_proxy_configuration`, `body_read_failed`, `render_cold_start`,
`vercel_not_found`, `upstream_error_page`, `unexpected_html`, `upstream_non_json`,
`upstream_stream_interrupted`, `upstream_timeout`, `upstream_too_large`), `request_id`
is present, and 429s include `retry_after_s`.

## Model and tokenizer

| Item | Value |
| --- | --- |
| Parameters | 803,712 |
| Architecture | decoder-only transformer, 3 layers, 4 heads / 2 KV heads (GQA), `n_embd` 128, RoPE, weight tying, dropout 0.1 |
| Context | 128 tokens |
| Tokenizer | byte-level BPE, 512-token vocabulary, fitted **only** on the training split |
| Precision | float32 on CPU; KV cache is exact (identical output, 1.17× faster); int8 measured at 0.98× and not enabled |
| Checkpoints | `ckpt-best.pt` / `ckpt-final.pt` + JSON config; atomic writes, resumable |

Full numbers, hardware and methodology: [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md);
raw JSON in [docs/measurements/](docs/measurements/).

## Training

```sh
export HELIX_DATA_ROOT=/tmp/helix-run          # must resolve outside the repository

python scripts/prepare_corpus.py --manifest "$HELIX_DATA_ROOT/source-manifest.json" \
                                 --output "$HELIX_DATA_ROOT/corpus.txt"
cp configs/reproduce-cpu.json "$HELIX_DATA_ROOT/llm-config.json"
python -m helix train --config "$HELIX_DATA_ROOT/llm-config.json"

python scripts/generate_sft_pairs.py --base data/sft_pairs.json \
  --output "$HELIX_DATA_ROOT/sft-pairs.json" \
  --manifest "$HELIX_DATA_ROOT/sft-pairs-manifest.json" --seed 1337
python -m helix sft --config configs/reproduce-cpu-sft.json \
  --base-checkpoint "$HELIX_DATA_ROOT/llm/ckpt-final.pt" \
  --sft-pairs "$HELIX_DATA_ROOT/sft-pairs.json" --out-dir "$HELIX_DATA_ROOT/sft"

python -m helix eval --config "$HELIX_DATA_ROOT/llm-config.json" \
  --checkpoint "$HELIX_DATA_ROOT/sft/ckpt-best.pt" --out-path "$HELIX_DATA_ROOT/sft-eval.json"
python -m helix bench --config "$HELIX_DATA_ROOT/llm-config.json" \
  --checkpoint "$HELIX_DATA_ROOT/sft/ckpt-best.pt" --out-path /tmp/bench-report.json
```

Training is seed-controlled, gradient-clipped, supports gradient accumulation and
resume, and refuses to write a checkpoint when the loss or gradient norm is non-finite
(`HELIX_ALLOW_NONFINITE_SKIP=1` opts into skipping and counting such steps instead).
`metrics.json` records the git commit, sha256 of corpus/train/val/vocab/merges, the
environment (Python, torch, device, threads), skipped steps, throughput and peak RSS.
Artifacts are written outside the repository and are never committed.

## Data provenance and licensing

`data/sources/` contains pinned, human-reviewed sources with a manifest recording
licence, retrieval time, sha256, attribution and usage rights. `prepare_corpus.py` only
materialises sources whose rights are confirmed; unknown rights are treated as *no
permission*, not as permission. Instruction data is generated by
`scripts/generate_sft_pairs.py` with a manifest and a hash, and its training/held-out
split is made per pair so a task family cannot leak across the boundary. Evaluation
examples are never trained on. See [docs/PROVENANCE.md](docs/PROVENANCE.md),
[docs/SFT_DATA.md](docs/SFT_DATA.md) and [docs/DATA_PIPELINE.md](docs/DATA_PIPELINE.md).

## Evaluation

Reported from this checkout, not from memory:

| Stage | Result |
| --- | --- |
| Pretraining (200 steps) | best validation loss 4.7809, 16,340 tok/s, 698 MB peak RSS, 0 skipped steps |
| Instruction tuning (60 steps) | held-out loss 3.5044, 465 train / 51 held-out pairs |
| Evaluation | validation loss 4.9701, perplexity 144.04, **1/5 probes passed** (calibration only) |
| Throughput | forward+backward 15,271 tok/s @ block 128; generation 566 tok/s (1.77 ms/token) |
| KV cache / int8 | 1.17× faster with identical output / 0.98× (no win) |

Method: held-out token windows for loss/perplexity, five fixed probes for capability,
and a benchmark script for throughput and memory. The full table, including what these
numbers do **not** show, is in [docs/MEASUREMENTS.md](docs/MEASUREMENTS.md); older
experiments are in the [research ledger](docs/RESEARCH_LEDGER.md).

## Security and privacy

- Secrets live in environment variables only. Nothing in `public/` or `mobile/` reads a
  credential at build time; the browser never sees the Render credentials — Vercel's
  proxy attaches Basic auth server-side.
- `/healthz` exposes no version, path or secret. `/api/diagnostics` requires the
  deployment's auth and reports presence flags and byte counts, never values.
- Logs are sanitised: no passwords, `Authorization` headers, raw prompts or code, and no
  filesystem paths in error responses. Audit entries store metadata and input hashes.
- Uploads are size-limited (4 MiB at the proxy, validated again at the app), and
  sandboxed code execution fails closed when namespace isolation is unavailable.
- Memory notes are opt-in and stored unencrypted on the host; `/api/memory` can delete
  them.
- Mobile stores chat history in app-local storage and Basic credentials in Android
  Keystore via `expo-secure-store`.

## Deployment

**Render (backend).** `render.yaml` defines `helix-chat`: Python 3.11, CPU torch,
gunicorn `--workers 1 --threads 2 --timeout 120`, health check `/healthz`,
`HELIX_REQUIRE_AUTH=1`, and secrets supplied by the dashboard:

| Variable | Purpose |
| --- | --- |
| `HELIX_AUTH_USERNAME`, `HELIX_AUTH_PASSWORD` | HTTP Basic auth required when `HELIX_REQUIRE_AUTH=1` |
| `HELIX_MODEL_BUNDLE_URL`, `HELIX_MODEL_BUNDLE_SHA256` | checksum-verified model download at build time |
| `HELIX_DATA_ROOT` | runtime data root (default `/tmp/helix-data`, must not be inside the repo) |
| `HELIX_RETRIEVAL_ROOT` | optional directory of local `.md`/`.txt`/`.html` documents to ground answers |
| `HELIX_RATE_LIMIT_PER_MINUTE`, `HELIX_CHAT_RATE_LIMIT_PER_MINUTE` | per-process token buckets |
| `HELIX_BUILD_SHA` | build identifier surfaced by `/api/version` |

**Vercel (frontend + proxy).** Static `public/` plus `api/**` serverless functions.
Required environment variables: `HELIX_API_BASE_URL` (the Render origin),
`HELIX_AUTH_USERNAME`, `HELIX_AUTH_PASSWORD`. If any is missing the proxy answers
`503 proxy_not_configured` instead of failing opaquely.

**Verify a deployment** (works against any origin, honours Basic auth):

```sh
HELIX_API_BASE_URL=https://helix-chat.onrender.com \
HELIX_AUTH_USERNAME=... HELIX_AUTH_PASSWORD=... \
  node scripts/smoke_deploy.mjs --base https://helix-web-lemon.vercel.app
```

It checks HTML shell, `/api/health`, `/api/status`, JSON 404s, SSE framing (not HTML),
`/api/chat`, and input validation. The same script against a local server passes
10/10 checks.

**Android.** See [mobile/README.md](mobile/README.md). CI
(`.github/workflows/mobile-android.yml`) type-checks, tests, validates the Expo config
and SDK dependency versions, then builds an APK and uploads it with a sha256. A debug
build has completed in CI (run 40837220492); a signed release requires the four signing
secrets and is attached to a GitHub release only for tags. Signing material comes from
repository secrets, and no APK is committed to this repository.

## Troubleshooting

| Symptom | Cause and fix |
| --- | --- |
| Browser: `Unexpected token 'T', "The page c"... is not valid JSON` | The upstream returned an HTML page (typically Render's "Application loading" interstitial) and something parsed it as JSON. The proxy now sniffs non-JSON 2xx/5xx bodies and answers `503 render_cold_start` (retryable, after bounded cold-start retries) or `503 upstream_error_page`; check the Render service log and `/healthz` directly. |
| `/api/chat` returns 503 `model_unavailable` | The checkpoint/config are missing or unreadable. `GET /api/diagnostics` lists artifact presence, not paths. |
| 401 on every `/api/*` call | The deployment sets `HELIX_REQUIRE_AUTH=1`; supply the Basic credentials (Vercel env vars, or Settings on Android). |
| 429 `compute_busy` | One inference at a time by design; the client should retry after `retry_after_s`. |
| Streaming works locally but not through a proxy | Check `Content-Type: text/event-stream`, no buffering (`X-Accel-Buffering: no`), and the function's `maxDuration`. |
| Learned image modes fail, physics rendering and image analysis still work | The diffusion checkpoint is absent; `/api/status` reports which model files exist and the response says which mode ran. |
| `HELIX data root must resolve outside the repository` | Point `HELIX_DATA_ROOT` somewhere else; runtime data is never stored in the checkout. |

## Testing

```sh
.venv/bin/python -m pytest -q                            # 239 passed, 6 skipped
node --test --test-timeout=20000 tests/node/proxy.test.cjs  # 13/13
cd mobile && npm run typecheck && npm run test:logic        # tsc clean, 29/29
```

Coverage includes API schemas, Basic auth, missing env vars, bundle verification,
empty/oversized prompts, streaming and mid-stream disconnect (the compute lock must be
released), proxy routing and HTML-error handling, per-user isolation, upload limits,
sandbox failure, math correctness, retrieval citations, token drift between web and
Android, and the committed measurement reports.

## Documentation

- [Architecture](docs/ARCHITECTURE.md) · [Usage](docs/USAGE.md) · [API contract](docs/ARCHITECTURE.md)
- [Measurements](docs/MEASUREMENTS.md) · [Research ledger](docs/RESEARCH_LEDGER.md) · [Model card](docs/MODEL_CARD.md)
- [Reproducibility](docs/REPRODUCIBILITY.md) · [Data pipeline](docs/DATA_PIPELINE.md) · [Provenance](docs/PROVENANCE.md)
- [SFT data](docs/SFT_DATA.md) · [Changelog](CHANGELOG.md) · [Android client](mobile/README.md)

HELIX-authored code and weights are MIT-licensed; mixed-source training data retain
their original notices and terms. See `LICENSE`, `artifacts/WEIGHTS_LICENSE.txt` and the
[source inventory](docs/PROVENANCE.md).
