#!/usr/bin/env node
'use strict';

/**
 * End-to-end smoke test for a deployed Beta by helix instance.
 *
 *   node scripts/smoke_deploy.mjs --base https://helix-web-lemon.vercel.app
 *   HELIX_WEB_URL=https://example --json node scripts/smoke_deploy.mjs
 *
 * Verifies the exact failure that produced
 *   "Unexpected token 'T', "The page c"... is not valid JSON"
 * by asserting that every API path answers with JSON or server-sent events and
 * never with HTML. No credentials are required or printed: the browser-facing
 * proxy holds them server side.
 */

const DEFAULT_TIMEOUT_MS = 90_000;

function parseArgs(argv) {
  const args = { base: process.env.HELIX_WEB_URL || 'https://helix-web-lemon.vercel.app', json: false };
  for (let i = 0; i < argv.length; i += 1) {
    if (argv[i] === '--base' && argv[i + 1]) args.base = argv[++i];
    else if (argv[i] === '--json') args.json = true;
  }
  args.base = args.base.replace(/\/$/, '');
  return args;
}

function classifyContentType(value) {
  const type = String(value || '').toLowerCase();
  if (type.includes('application/json')) return 'json';
  if (type.includes('text/event-stream')) return 'sse';
  if (type.startsWith('image/')) return 'image';
  if (type.includes('text/html')) return 'html';
  return type || 'unknown';
}

async function request(base, path, options = {}) {
  const started = Date.now();
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new Error('timeout')), options.timeoutMs || DEFAULT_TIMEOUT_MS);
  try {
    const response = await fetch(base + path, { ...options, signal: controller.signal });
    return {
      response,
      ms: Date.now() - started,
      kind: classifyContentType(response.headers.get('content-type')),
    };
  } finally {
    clearTimeout(timer);
  }
}

function record(results, name, ok, detail) {
  results.push({ name, ok, detail });
}

async function expectJson(results, base, path, options = {}) {
  try {
    const { response, kind, ms } = await request(base, path, options);
    const text = await response.text();
    let parsed = null;
    try { parsed = JSON.parse(text); } catch { /* handled below */ }
    const ok = kind === 'json' && parsed !== null && (response.status < 500);
    record(results, `${options.method || 'GET'} ${path}`, ok, {
      status: response.status,
      content_type: kind,
      request_id: response.headers.get('x-request-id'),
      latency_ms: ms,
      error: parsed && parsed.error ? String(parsed.error).slice(0, 200) : undefined,
    });
    return { response, parsed, kind };
  } catch (error) {
    record(results, `${options.method || 'GET'} ${path}`, false, { error: error.message });
    return { response: null, parsed: null, kind: null };
  }
}

async function main() {
  const args = parseArgs(process.argv.slice(2));
  const results = [];
  const base = args.base;

  // 1. The static shell must be the Beta by helix client.
  try {
    const { response, kind } = await request(base, '/');
    const html = await response.text();
    const isShell = kind === 'html' && /Beta by helix/i.test(html);
    record(results, 'GET / (Beta by helix shell)', isShell, {
      status: response.status, content_type: kind, bytes: html.length,
    });
  } catch (error) {
    record(results, 'GET / (Beta by helix shell)', false, { error: error.message });
  }

  // 2. Health and readiness must be JSON and must expose no secrets.
  const health = await expectJson(results, base, '/api/health');
  if (health.parsed) {
    const serialized = JSON.stringify(health.parsed).toLowerCase();
    const leaks = ['password', 'authorization', 'basic ', 'helix_auth'].filter((needle) => serialized.includes(needle));
    record(results, 'GET /api/health leaks no secrets', leaks.length === 0, { leaks });
  }

  // 3. Model registry status.
  const status = await expectJson(results, base, '/api/status');
  if (status.parsed) {
    record(results, 'GET /api/status schema', typeof status.parsed.models === 'object', {
      models: status.parsed.models,
      loaded: status.parsed.loaded,
    });
  }

  // 4. Unknown API paths must answer with JSON, not Vercel's HTML 404 page.
  const missing = await expectJson(results, base, '/api/definitely-not-a-route');
  record(results, 'GET /api/* unknown route returns JSON', missing.kind === 'json', {
    status: missing.response ? missing.response.status : null,
    content_type: missing.kind,
  });

  // 5. Streaming chat: the exact path that produced the browser error.
  try {
    const started = Date.now();
    const { response, kind, ms } = await request(base, '/api/chat/stream', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({ message: 'Reply with one short sentence about recursion.', max_new_tokens: 24 }),
    });
    let text = '';
    let firstByteMs = null;
    if (response.body) {
      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      for (;;) {
        const { value, done } = await reader.read();
        if (done) break;
        if (firstByteMs === null) firstByteMs = Date.now() - started;
        text += decoder.decode(value, { stream: true });
      }
    }
    const events = text.split('\n')
      .filter((line) => line.startsWith('data:'))
      .map((line) => {
        try { return JSON.parse(line.slice(5).trim()); } catch { return null; }
      })
      .filter(Boolean);
    const sawDone = events.some((event) => event.done === true);
    const sawError = events.some((event) => typeof event.error === 'string');
    const ok = kind === 'sse' && !/<!doctype html|<html/i.test(text) && sawDone && !sawError;
    record(results, 'POST /api/chat/stream (SSE, not HTML)', ok, {
      status: response.status,
      content_type: kind,
      first_byte_ms: firstByteMs,
      total_ms: ms,
      events: events.length,
      done_event: sawDone,
      backend_error: sawError ? text.slice(0, 200) : undefined,
    });
  } catch (error) {
    record(results, 'POST /api/chat/stream (SSE, not HTML)', false, { error: error.message });
  }

  // 6. Non-streaming chat must return an actual answer, not merely valid JSON.
  const chat = await expectJson(results, base, '/api/chat', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ message: 'Explain recursion in one short sentence.', max_new_tokens: 24 }),
  });
  const chatReply = chat.parsed && (chat.parsed.reply || chat.parsed.answer);
  record(results, 'POST /api/chat returns a non-empty answer', Boolean(
    chat.response && chat.response.status === 200 && typeof chatReply === 'string' && chatReply.trim()
  ), {
    status: chat.response ? chat.response.status : null,
    answer_chars: typeof chatReply === 'string' ? chatReply.trim().length : 0,
    request_id: chat.parsed?.request_id || chat.response?.headers.get('x-request-id') || null,
  });

  // 7. Oversized payloads must be rejected as JSON before the backend.
  const oversized = await expectJson(results, base, '/api/chat', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ message: 'x'.repeat(5 * 1024 * 1024) }),
  });
  record(results, 'POST /api/chat rejects oversized input', Boolean(
    oversized.response && oversized.response.status === 413 && oversized.kind === 'json'
  ), {
    status: oversized.response ? oversized.response.status : null,
    content_type: oversized.kind,
  });

  const failed = results.filter((entry) => !entry.ok);
  if (args.json) {
    console.log(JSON.stringify({ base, results, ok: failed.length === 0 }, null, 2));
  } else {
    for (const entry of results) {
      const marker = entry.ok ? 'PASS' : 'FAIL';
      const detail = Object.entries(entry.detail || {})
        .filter(([, value]) => value !== undefined && value !== null)
        .map(([key, value]) => `${key}=${typeof value === 'object' ? JSON.stringify(value) : value}`)
        .join(' ');
      console.log(`${marker}  ${entry.name}  ${detail}`);
    }
    console.log(`\n${results.length - failed.length}/${results.length} checks passed against ${base}`);
  }
  process.exit(failed.length === 0 ? 0 : 1);
}

main().catch((error) => {
  console.error('smoke test failed:', error.message);
  process.exit(1);
});
