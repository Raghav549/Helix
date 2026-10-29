# Changelog

All entries below describe changes that are in this repository and, where stated,
verified by tests in the same tree. Nothing here claims a deployment state that was not
observed; deployment status is tracked in [docs/STATUS.md](docs/STATUS.md).

## Unreleased

### Product

- New web client ("Beta by helix") in `public/`: bright paper surfaces, bold ink type,
  pink/coral/orange accents, sidebar with history and settings, greeting + quick-action
  cards, a composer with a decorative wave glow, and truthful activity rows
  (thinking / generating / reading local documents / running a tool). No percentage
  progress and no invented capabilities.
- Tool mode and a symbolic-math quick action call the real `/api/agent` endpoint; a
  "Forecast (least squares)" card calls `/api/research/forecast`.
- Design tokens live in CSS custom properties; the Android client mirrors them and a test
  fails when the palettes drift apart.
- New Expo/React Native Android client in `mobile/` sharing the same API contract and
  tokens, with streaming over `XMLHttpRequest`, a single non-streaming fallback,
  device-local history, Keystore-backed Basic credentials, image analysis, settings and
  an `expo prebuild`-verified manifest without camera/microphone permissions.

### Reliability

- Fixed the production browser error `Unexpected token 'T', "The page c"... is not valid
  JSON`: the Vercel proxy no longer relays upstream HTML, sniffs and retries cold-start
  interstitials, classifies provider error pages, and guarantees a JSON body with a
  request id for every `/api/*` failure.
- The proxy now forwards SSE without buffering, maps upstream timeouts to `504`,
  rejects oversized bodies with `413`, refuses unknown `/api/*` routes itself, and adds
  security headers in `vercel.json`.
- Flask returns JSON for 404/405/413/429/500 on `/api/*`, echoes or generates
  `X-Request-ID`, exposes `/api/v1/*` aliases, and reports honest degraded readiness.
- Sanitised access logging (request id, route, status, latency, hashed or omitted user
  input) and rate limiting on both chat and general routes.
- A client disconnect mid-stream releases the single compute slot (regression test).

### Intelligence

- Tool orchestration (`helix/orchestrator.py`): classify → plan → permission check →
  execute with timeouts → validate → synthesise, with per-call duration, permission
  level and input hash recorded; retrieved documents are treated as untrusted context.
- Symbolic mathematics via SymPy, least-squares forecasting with prediction intervals,
  hybrid local-document retrieval with citations, and an opt-in Fortran kernel with a
  pure-Python fallback and benchmark.
- Training robustness: abort (or explicitly skip) on non-finite loss or gradient norm,
  gradient clipping and accumulation, atomic checkpoints, resume, and metrics that
  record provenance (git commit, dataset/tokenizer hashes) and environment.
- Instruction-tuning scaling measured as a negative result: 1000 steps lower the SFT
  held-out loss (3.50 → 2.52) while general validation loss rises (4.98 → 5.86) and no
  probe improves, so the 60-step checkpoint remains the default.
- Retrieval quality measured on a labelled set (recall@1 0.75, recall@4 0.83, MRR 0.77,
  zero invalid citations) with the lexical retriever stated honestly.

### Documentation

- README rewritten: architecture diagram, API and environment reference, model and
  tokenizer facts, training commands, data provenance, evaluation methodology with
  actual results, hardware, security/privacy, Vercel + Render deployment, Android build,
  troubleshooting, testing and limitations.
- `docs/MEASUREMENTS.md` and `docs/measurements/*.json` publish the real run.
- `docs/STATUS.md` records deployed state, verified results and open blockers.

## 0.13.0

- Research workspace with language modelling, image experiments, coding tools, a
  research lab, source manifests and provenance, a research ledger, model cards and a
  local web UI. Baseline measured losses and synthetic-domain image results are recorded
  in `docs/RESEARCH_LEDGER.md`.
