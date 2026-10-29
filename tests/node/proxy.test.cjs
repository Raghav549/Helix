'use strict';

/**
 * Regression tests for the Vercel -> Render proxy layer.
 *
 * These tests reproduce the production incident ("Unexpected token 'T', "The
 * page c"... is not valid JSON") by returning the same HTML interstitials that
 * Render and Vercel serve, and assert that the browser only ever receives JSON,
 * SSE or image bytes.
 *
 * Run with: node --test tests/node/
 */

const test = require('node:test');
const assert = require('node:assert/strict');
const http = require('node:http');

const { createProxy, classifyHtml } = require('../../lib/helix-proxy');

const RENDER_LOADING_HTML = `<!doctype html><html><head><title>Render - Application loading</title></head>
<body><p>Your application is starting up.</p></body></html>`;
const VERCEL_404_HTML = `<!doctype html><html><head><title>404: This page could not be found.</title></head>
<body>The page could not be found.</body></html>`;

/** Start an upstream mock; `handler(req, res, state)` defines its behaviour. */
async function upstream(handler) {
  const state = { requests: [], closed: false };
  const server = http.createServer((req, res) => {
    state.requests.push({ method: req.method, url: req.url, headers: req.headers });
    req.on('close', () => { state.closed = true; });
    handler(req, res, state);
  });
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const base = `http://127.0.0.1:${server.address().port}`;
  return { server, base, state };
}

async function proxyServer(upstreamBase, options = {}) {
  const logs = [];
  const handler = createProxy({
    env: {
      HELIX_API_BASE_URL: upstreamBase,
      HELIX_AUTH_USERNAME: 'helix-test',
      HELIX_AUTH_PASSWORD: 'super-secret-value',
      HELIX_PROXY_COLD_START_RETRIES: String(options.retries ?? 0),
      HELIX_PROXY_RETRY_BUDGET_MS: String(options.retryBudget ?? 5000),
      HELIX_PROXY_MAX_BODY_BYTES: String(options.maxBody ?? 4 * 1024 * 1024),
      HELIX_PROXY_TIMEOUT_MS: String(options.timeout ?? 5000),
      HELIX_PROXY_STREAM_TIMEOUT_MS: String(options.streamTimeout ?? 10000),
    },
    log: (event) => logs.push(event),
  });
  const server = http.createServer((req, res) => handler(req, res, (path) => path.startsWith('/api/')));
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  return { server, base: `http://127.0.0.1:${server.address().port}`, logs, handler };
}

async function close(...servers) {
  await Promise.all(servers.map((server) => new Promise((resolve) => {
    server.closeAllConnections?.();
    server.close(resolve);
  })));
}

test('classifyHtml recognises the two production interstitials', () => {
  const render = classifyHtml(RENDER_LOADING_HTML);
  assert.equal(render.kind, 'render_cold_start');
  assert.equal(render.retryable, true);
  const vercel = classifyHtml(VERCEL_404_HTML);
  assert.equal(vercel.kind, 'vercel_not_found');
  assert.equal(vercel.retryable, false);
  assert.equal(classifyHtml('{"ok":true}').kind, 'unexpected_html');
});

test('missing configuration fails closed with JSON, never HTML', async () => {
  const handler = createProxy({ env: {}, log: () => {} });
  const server = http.createServer((req, res) => handler(req, res, () => true));
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  const { port } = server.address();
  const response = await fetch(`http://127.0.0.1:${port}/api/status`);
  assert.equal(response.status, 503);
  assert.match(response.headers.get('content-type'), /application\/json/);
  const body = await response.json();
  assert.equal(body.code, 'proxy_not_configured');
  assert.match(body.detail, /HELIX_API_BASE_URL/);
  assert.ok(response.headers.get('x-request-id'));
  await close(server);
});

test('Render cold-start HTML becomes a retryable JSON 503', async () => {
  const mock = await upstream((req, res) => {
    res.writeHead(502, { 'content-type': 'text/html; charset=utf-8' });
    res.end(RENDER_LOADING_HTML);
  });
  const proxy = await proxyServer(mock.base, { retries: 0 });
  const response = await fetch(`${proxy.base}/api/status`);
  assert.equal(response.status, 503);
  assert.match(response.headers.get('content-type'), /application\/json/);
  const body = await response.json();
  assert.equal(body.code, 'render_cold_start');
  assert.equal(body.retryable, true);
  assert.equal(body.layer, 'vercel-proxy');
  assert.ok(!JSON.stringify(body).includes('super-secret-value'));
  await close(proxy.server, mock.server);
});

test('a cold start that finishes is retried transparently', async () => {
  let calls = 0;
  const mock = await upstream((req, res) => {
    calls += 1;
    if (calls === 1) {
      res.writeHead(502, { 'content-type': 'text/html' });
      res.end(RENDER_LOADING_HTML);
      return;
    }
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ ok: true, ready: true }));
  });
  const proxy = await proxyServer(mock.base, { retries: 2, retryBudget: 4000 });
  const response = await fetch(`${proxy.base}/api/status`);
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { ok: true, ready: true });
  assert.equal(calls, 2);
  assert.ok(proxy.logs.some((line) => line.event === 'upstream_cold_start_retry'));
  await close(proxy.server, mock.server);
});

test('upstream HTML 200 on a JSON endpoint is converted to a JSON error', async () => {
  const mock = await upstream((req, res) => {
    res.writeHead(200, { 'content-type': 'text/html; charset=utf-8' });
    res.end('<html><body>Some proxy page</body></html>');
  });
  const proxy = await proxyServer(mock.base);
  const response = await fetch(`${proxy.base}/api/status`);
  assert.notEqual(response.status, 200);
  assert.match(response.headers.get('content-type'), /application\/json/);
  const body = await response.json();
  assert.ok(['render_cold_start', 'unexpected_html'].includes(body.code));
  await close(proxy.server, mock.server);
});

test('a plain-text 401 from the backend never leaks through as non-JSON', async () => {
  const mock = await upstream((req, res) => {
    res.writeHead(401, { 'content-type': 'text/plain; charset=utf-8', 'www-authenticate': 'Basic realm="HELIX"' });
    res.end('Authentication required');
  });
  const proxy = await proxyServer(mock.base);
  const response = await fetch(`${proxy.base}/api/status`);
  assert.match(response.headers.get('content-type'), /application\/json/);
  const body = await response.json();
  assert.equal(body.code, 'upstream_non_json');
  assert.equal(body.upstream_content_type, 'text/plain; charset=utf-8');
  await close(proxy.server, mock.server);
});

test('valid JSON is relayed byte-for-byte with status and credentials', async () => {
  const mock = await upstream((req, res) => {
    const expected = 'Basic ' + Buffer.from('helix-test:super-secret-value').toString('base64');
    assert.equal(req.headers.authorization, expected);
    res.writeHead(201, { 'content-type': 'application/json' });
    res.end(JSON.stringify({ id: 'note-1' }));
  });
  const proxy = await proxyServer(mock.base);
  const response = await fetch(`${proxy.base}/api/memory`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', 'x-request-id': 'req-abcdefgh' },
    body: JSON.stringify({ text: 'remember this' }),
  });
  assert.equal(response.status, 201);
  assert.equal(response.headers.get('x-request-id'), 'req-abcdefgh');
  assert.deepEqual(await response.json(), { id: 'note-1' });
  assert.equal(mock.state.requests[0].headers['x-request-id'], 'req-abcdefgh');
  await close(proxy.server, mock.server);
});

test('SSE is streamed incrementally with anti-buffering headers', async () => {
  const mock = await upstream((req, res) => {
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.write('data: {"delta":"Hel"}\n\n');
    setTimeout(() => {
      res.write('data: {"delta":"ix"}\n\n');
      setTimeout(() => { res.write('data: {"done":true}\n\n'); res.end(); }, 120);
    }, 120);
  });
  const proxy = await proxyServer(mock.base);
  const response = await fetch(`${proxy.base}/api/chat/stream`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: '{"message":"hi"}' });
  assert.equal(response.status, 200);
  assert.match(response.headers.get('content-type'), /text\/event-stream/);
  assert.equal(response.headers.get('cache-control'), 'no-cache, no-transform');
  assert.equal(response.headers.get('x-accel-buffering'), 'no');

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  const timestamps = [];
  let text = '';
  for (;;) {
    const { value, done } = await reader.read();
    if (done) break;
    timestamps.push(Date.now());
    text += decoder.decode(value, { stream: true });
  }
  text += decoder.decode();
  assert.match(text, /"delta":"Hel"/);
  assert.match(text, /"done":true/);
  assert.ok(timestamps.length >= 2, 'chunks must not be buffered into a single write');
  assert.ok(timestamps[timestamps.length - 1] - timestamps[0] > 100, 'stream must arrive progressively');
  await close(proxy.server, mock.server);
});

test('client disconnect aborts the upstream stream', async () => {
  let upstreamClosed = false;
  let finished = false;
  const mock = await upstream((req, res) => {
    res.writeHead(200, { 'content-type': 'text/event-stream' });
    res.write('data: {"delta":"a"}\n\n');
    req.on('close', () => { upstreamClosed = true; });
    const interval = setInterval(() => {
      if (finished || upstreamClosed) { clearInterval(interval); return; }
      res.write('data: {"delta":"b"}\n\n');
    }, 50);
    setTimeout(() => { finished = true; clearInterval(interval); res.end(); }, 3000);
  });
  const proxy = await proxyServer(mock.base, { streamTimeout: 5000 });
  const controller = new AbortController();
  const response = await fetch(`${proxy.base}/api/chat/stream`, { method: 'DELETE', headers: { 'content-type': 'application/json' }, signal: controller.signal });
  const reader = response.body.getReader();
  await reader.read();
  controller.abort();
  await new Promise((resolve) => setTimeout(resolve, 400));
  assert.equal(upstreamClosed, true, 'upstream request must be cancelled when the browser disconnects');
  finished = true;
  await close(proxy.server, mock.server);
});

test('oversized request bodies are rejected with JSON 413 before reaching upstream', async () => {
  const mock = await upstream((req, res) => {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end('{"ok":true}');
  });
  const proxy = await proxyServer(mock.base, { maxBody: 1024 });
  const response = await fetch(`${proxy.base}/api/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ message: 'x'.repeat(4096) }),
  });
  assert.equal(response.status, 413);
  const body = await response.json();
  assert.equal(body.code, 'request_too_large');
  assert.equal(mock.state.requests.length, 0);
  await close(proxy.server, mock.server);
});

test('unmatched paths and methods are rejected as JSON', async () => {
  const mock = await upstream((req, res) => {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end('{"ok":true}');
  });
  const proxy = await proxyServer(mock.base);
  const notFound = await fetch(`${proxy.base}/other/route`);
  assert.equal(notFound.status, 404);
  assert.equal((await notFound.json()).code, 'route_not_found');
  const notAllowedStatus = await new Promise((resolve, reject) => {
    const request = http.request(`${proxy.base}/api/status`, { method: 'TRACE' }, (response) => {
      response.resume();
      resolve(response.statusCode);
    });
    request.on('error', reject);
    request.end();
  });
  assert.equal(notAllowedStatus, 405);
  await close(proxy.server, mock.server);
});

test('unreachable backend returns a retryable JSON 502', async () => {
  const proxy = await proxyServer('http://127.0.0.1:9');
  const response = await fetch(`${proxy.base}/api/status`);
  assert.equal(response.status, 502);
  const body = await response.json();
  assert.equal(body.code, 'upstream_unreachable');
  assert.equal(body.retryable, true);
  await close(proxy.server);
});

test('logs never contain credentials, prompts or authorization headers', async () => {
  const mock = await upstream((req, res) => {
    res.writeHead(200, { 'content-type': 'application/json' });
    res.end('{"reply":"hello"}');
  });
  const proxy = await proxyServer(mock.base);
  const prompt = 'my private prompt about a secret project';
  await fetch(`${proxy.base}/api/chat`, {
    method: 'POST',
    headers: { 'content-type': 'application/json', authorization: 'Bearer client-token-should-not-be-forwarded' },
    body: JSON.stringify({ message: prompt }),
  });
  const serialized = JSON.stringify(proxy.logs);
  assert.ok(!serialized.includes('super-secret-value'));
  assert.ok(!serialized.includes(prompt));
  assert.ok(!serialized.includes('client-token-should-not-be-forwarded'));
  const upstreamHeaders = mock.state.requests[0].headers;
  assert.equal(upstreamHeaders.authorization.startsWith('Basic '), true);
  assert.equal(upstreamHeaders['x-forwarded-proto'], 'https');
  await close(proxy.server, mock.server);
});
