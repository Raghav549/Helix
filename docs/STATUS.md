# Release status

Last verified: 2026-10-29. Branch `arena/70d7f428-helix` of
[`Raghav549/Helix`](https://github.com/Raghav549/Helix).

This page separates what was **verified here** from what **could not be verified**, and
lists the exact blockers. Nothing below is inferred from a dashboard status.

## Verified in this workspace

| Area | Evidence |
| --- | --- |
| Python test suite | `239 passed, 6 skipped` (`python -m pytest -q`, Python 3.11.2, torch 2.5.1 CPU) |
| Proxy tests | `15/15` (`node --test --test-timeout=20000 tests/node/proxy.test.cjs`) |
| Android logic tests | `30/30` (`cd mobile && npm run test:logic`) |
| Android typecheck | `npx tsc --noEmit` clean (strict, `noUncheckedIndexedAccess`) |
| Android bundle | `npx expo export --platform android` → 770 modules, 1.9 MB Hermes bytecode |
| Android native config | `npx expo prebuild --platform android` → manifest without `CAMERA`/`RECORD_AUDIO` |
| End-to-end local API | server on `127.0.0.1:8767` with real checkpoints: `/api/health model_ready:true`, SSE `meta/delta/done`, `/api/agent` symbolic answer |
| Deployment smoke script | `scripts/smoke_deploy.mjs` → 10/10 against the local server |
| Training pipeline | prepare corpus (102,879 B, 9 sources) → 200-step pretrain → 516 SFT pairs → 60-step SFT → eval → bench, all reproducible from `configs/reproduce-cpu*.json` |
| Retrieval quality | 12 labelled questions over the docs: recall@1 0.75, recall@4 0.83, MRR 0.77, all citations valid (`scripts/eval_retrieval.py`) |
| Measurements published | `docs/MEASUREMENTS.md` + `docs/measurements/*.json`, guarded by `tests/test_measurements.py` |
| CI on this branch | `HELIX tests` workflow: success |

## Not verified / blocked

| Item | State |
| --- | --- |
| Render backend `helix-chat.onrender.com` | Still serving Render's "Application loading" HTML to `/healthz` and to `/api/status` via Vercel. The latest recorded Render deploy says success, but the service does not serve responses. |
| Vercel **preview** of this branch | Verified live: `helix-web-git-arena-70d7f428-helix-raghav549s-projects.vercel.app` serves the new "Beta by helix" client and `/api/status` now returns JSON `503 {"code":"render_cold_start", ... "upstream_content_type":"text/html; charset=utf-8"}` instead of HTML. The browser no longer sees a JSON parse error. |
| Vercel production `helix-web-lemon.vercel.app` | **Deployed after merging PR #15** and verified: `/api/status` returns backend JSON, and missing routes (`/api/health`, `/api/version`) return proxy JSON `502 {"code":"upstream_non_json","upstream_status":404}` instead of an HTML page. The browser no longer shows a JSON parse error. |
| Render backend revision | Production Render is still on an older build: `/api/version` and `/api/health` answer `404 text/html`, and `/api/status` returns the old sparse schema. `render.yaml` sets `autoDeploy: no`, so merging did not redeploy it. |
| Render backend (all environments) | Returns HTTP 503 with `text/html` for `/healthz`, `/api/status` and every API route; the proxy now reports that as JSON. |
| Production browser chat | Cannot pass until the Render service serves. The HTML-as-JSON failure mode is fixed and tested in this branch (proxy returns `503 render_cold_start` with JSON), but that code is not deployed from this branch. |
| Android APK artifact | **Produced by CI**: run [40837220492](https://github.com/Raghav549/Helix/actions/runs/40837220492) built `assembleDebug` and uploaded the artifact `beta-by-helix-debug-unsigned-release` (55,995,560 bytes, expires 2026-11-12). It is a debug APK, not a signed release; no APK is committed to the repository. |
| Model quality | Measured, not blocked: 1/5 probes pass, answers are frequently incoherent. A longer instruction-tuning run (1000 steps) lowered SFT held-out loss 3.50 → 2.52 but raised general validation loss 4.98 → 5.86 with no probe change, so the 60-step checkpoint stays the default. Improvements beyond this scale need more compute than a free CPU tier provides. |
| Fortran kernels | `gfortran`/`f2py` unavailable in this environment, so only the pure-Python fallback ran; the benchmark reports `backend: "python"`. |

## Blockers (precise)

1. **Render is serving an older build.** It responds now, but without `/api/health`,
   `/api/version` or the current `/api/status` schema. Redeploying needs dashboard access
   or a `RENDER_API_KEY` (`autoDeploy: no` in `render.yaml`). The web and Android clients
   report this as "Backend outdated" instead of a parse error.
2. **No production deploy credential for this branch.** Vercel and Render deploy from the
   default branch; this work sits on `arena/70d7f428-helix`. Merging the branch makes the
   fixed proxy and client deployable.
3. **Signing secrets for a releasable APK.** `ANDROID_KEYSTORE_BASE64`,
   `ANDROID_KEYSTORE_PASSWORD`, `ANDROID_KEY_ALIAS`, `ANDROID_KEY_PASSWORD`. Without
   them, CI produces a debug APK only.
4. **Compute budget.** Larger-scale training (bigger corpus, longer schedules) is not
   attempted on free CPU tiers; no job is left running beyond a command's lifetime.

## Next actions

1. Merge this branch so the corrected proxy and the new client reach production.
2. With Render access: confirm the build logs, then re-run
   `node scripts/smoke_deploy.mjs --base https://helix-web-lemon.vercel.app` and record
   the output here.
3. Add signing secrets, tag a release, and attach the CI-produced signed APK with its
   `apk.sha256`.
4. Continue Phase D only where it is measurable on available hardware: larger SFT set,
   retrieval quality measurement on a labelled question set, and tool-level evaluations.
