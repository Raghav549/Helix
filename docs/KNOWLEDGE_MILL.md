# Knowledge Mill

The Knowledge Mill accepts reviewed material and makes it available through the local retrieval index. It **does not train model weights**. Training remains a separate, versioned job with a reviewed dataset, holdout evaluation, model artifact and measurement report.

## Persistent data and provenance

Set `HELIX_DATA_ROOT` to persistent storage and optionally set `HELIX_KNOWLEDGE_ROOT`. The service writes:

- source bytes under `knowledge-mill/documents/`;
- immutable SHA-256, source URL, supplied licence note, submitter id and timestamp in `knowledge.sqlite3`;
- a deterministic retrieval index in `knowledge-mill/index/`.

Identical normalized content is deduplicated by SHA-256. Text, HTML, JSON, CSV and PDF are indexed; PNG, JPEG and WebP are retained for a later image-analysis pipeline but never misrepresented as text retrieval.

## Agent API

Configure one or more comma-separated secrets in `HELIX_AGENT_INGEST_KEYS`. Agents send one secret only in `X-HELIX-Agent-Key` to the Vercel proxy. The browser never receives the backend's Basic Auth credentials.

`POST /api/knowledge/ingest` accepts exactly one of:

- JSON `{ "text": "...", "name": "notes.txt", "license_note": "..." }`;
- JSON `{ "url": "https://public.example/doc", "license_note": "..." }`;
- multipart file field `file`, with optional `license_note`.

URLs must be public HTTP(S), cannot carry credentials, cannot redirect, reject private/loopback targets and are capped at 4 MiB. API results state whether a document was deduplicated and indexed, and always return `training_updated: false`.
