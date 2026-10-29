# Beta by helix — Android client

Expo (SDK 58) + React Native 0.88 + TypeScript client for the HELIX API. It shares the
web client's design tokens, API contract and honesty rules: every control performs a
real request, and nothing in the UI claims a capability the backend does not report.

```
mobile/
├── App.tsx                 app shell, Android back-button policy, safe areas
├── src/api/client.ts       HTTP/SSE client, typed errors, single-retry fallback
├── src/components/         sidebar, composer, bubbles, activity row, quick actions
├── src/lib/                SSE parser, base64/Basic auth, formatting, markdown splitter
├── src/screens/ChatScreen  conversation state, streaming, image analysis, settings
├── src/state/sessions.ts   device-local history with AsyncStorage
├── src/theme/tokens.ts     palette/spacing/radii mirrored from public/static/styles.css
└── tests/                  node:test suites over the pure logic modules
```

## Run it

```bash
cd mobile
npm ci
npm run typecheck      # tsc --noEmit
npm run test:logic     # compiles src/lib + tests, runs node:test
npx expo start --android   # needs an emulator/device with Expo Go or a dev build
```

The API base URL defaults to the deployment in `app.json` → `expo.extra.apiBaseUrl`
(`https://helix-web-lemon.vercel.app`). It can be changed at runtime in **Settings**
along with HTTP Basic credentials — the deployment sets `HELIX_REQUIRE_AUTH`, so the
app sends `Authorization: Basic …` when credentials are configured. Credentials are
stored with `expo-secure-store` (Android Keystore) and are never written to logs,
AsyncStorage or the bundle.

## What each control does

| Control | Real action |
| --- | --- |
| Composer send | `POST /api/chat/stream` (SSE), falls back to `POST /api/chat` if the stream transport fails before any token arrives |
| Stop | aborts the XHR (server releases its compute slot when the client disconnects) |
| Attach / "Analyse an image" | `expo-image-picker` → `POST /api/image/analyze` (multipart), output rendered field-by-field, no invented values |
| Tools toggle | sends the same conversation to `POST /api/agent`-style tool routing context; the tool list comes from `/api/status` |
| Quick actions | fill the composer with a concrete prompt, or open the image tool; they never fake a result |
| Settings → Test | `GET /api/health`, reports the server's own `model_ready` flag |
| Sidebar | history from AsyncStorage, delete per chat, "Delete all chats on this device", theme cycle, live server status |

There is no microphone button: the deployment exposes no speech endpoint, and a
decorative one would be a lie. There is no live web search for the same reason
(`/api/status` reports `web_search: false`).

## Streaming

React Native has no `ReadableStream` on the response, so `HelixClient.stream` uses
`XMLHttpRequest` with incremental `onprogress` reads. `parseSseChunk` buffers partial
frames, tolerates CRLF and multi-line `data:` fields, and surfaces malformed events as
errors instead of silently dropping text. If the stream cannot start, the client does a
single non-streaming request and renders that reply, marked as a normal answer.

## Testing and builds

* `npm run test:logic` — 29 assertions over SSE parsing, session persistence, error
  mapping (HTML error pages are never parsed as JSON), token drift versus the web CSS,
  Basic auth encoding and analysis formatting.
* `npx expo export --platform android` was verified locally: 770 modules bundled to
  1.9 MB Hermes bytecode.
* `npx expo prebuild --platform android` was verified locally; the generated manifest
  removes `CAMERA` and `RECORD_AUDIO` and keeps only `INTERNET` plus storage-read for
  the picker.
* GitHub Actions (`.github/workflows/mobile-android.yml`) validates types, tests, Expo
  config and SDK-compatible dependency versions, then builds `assembleDebug` (or
  `assembleRelease` when the signing secrets exist), verifies the APK exists, prints its
  badging, writes `apk.sha256` and uploads it as an artifact. Tagged signed builds are
  attached to a GitHub release.

No APK is committed to this repository, and no unsigned release build is presented as
installable. To produce a signed release, add `ANDROID_KEYSTORE_BASE64`,
`ANDROID_KEYSTORE_PASSWORD`, `ANDROID_KEY_ALIAS`, `ANDROID_KEY_PASSWORD` as repository
secrets; keystores and passwords must never be committed.

## Known limitations

* The HELIX models are small and experimental; replies can be wrong or incoherent.
  The UI states this in the composer footer and never claims otherwise.
* Server-side retrieval only exists when `HELIX_RETRIEVAL_ROOT` is configured; the app
  shows "Reading local documents" only when the stream's `meta` event carries sources.
* Device history is not synced anywhere; clearing app data deletes it.
