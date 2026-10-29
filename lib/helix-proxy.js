'use strict';

/**
 * Server-side Vercel -> HELIX backend proxy.
 *
 * Contract guarantees enforced here:
 *   1. Every `/api/*` response is either application/json, text/event-stream or
 *      an image. An upstream HTML page (Render cold-start interstitial, Vercel
 *      404 page, proxy error page) is NEVER relayed to the browser; it is
 *      converted into a structured JSON error that the UI can render.
 *   2. Credentials stay server side. Only allow-listed request headers and a
 *      Basic Authorization header built from Vercel env vars are forwarded.
 *   3. Server-sent events are streamed chunk by chunk, without buffering, and
 *      the upstream request is aborted when the browser disconnects.
 *   4. A cold-start HTML interstitial is retried (bounded, budgeted) because it
 *      proves the application never received the request.
 *   5. Every request carries a request id that appears in the response body,
 *      the response header and the sanitized JSON log line.
 */

const crypto = require('node:crypto');

const DEFAULT_MAX_BODY_BYTES = 4 * 1024 * 1024;
const MAX_JSON_BYTES = 2 * 1024 * 1024;
const SNIFF_BYTES = 64 * 1024;
const ALLOWED_METHODS = new Set(['GET', 'HEAD', 'POST', 'PUT', 'PATCH', 'DELETE', 'OPTIONS']);

const JSON_TYPE = /^application\/(?:[\w.+-]*\+)?json\b/i;
const SSE_TYPE = /^text\/event-stream\b/i;
const IMAGE_TYPE = /^image\/(?:png|jpeg|webp)\b/i;

// ---------------------------------------------------------------------------
// configuration
// ---------------------------------------------------------------------------

function readConfig(env = process.env) {
  const numeric = (name, fallback, min, max) => {
    const raw = Number.parseInt(env[name] || '', 10);
    if (!Number.isFinite(raw)) return fallback;
    return Math.min(max, Math.max(min, raw));
  };
  return {
    base: (env.HELIX_API_BASE_URL || '').trim(),
    username: env.HELIX_AUTH_USERNAME || '',
    password: env.HELIX_AUTH_PASSWORD || '',
    maxBodyBytes: numeric('HELIX_PROXY_MAX_BODY_BYTES', DEFAULT_MAX_BODY_BYTES, 1024, 32 * 1024 * 1024),
    requestTimeoutMs: numeric('HELIX_PROXY_TIMEOUT_MS', 55_000, 1_000, 300_000),
    streamTimeoutMs: numeric('HELIX_PROXY_STREAM_TIMEOUT_MS', 115_000, 1_000, 300_000),
    coldStartRetries: numeric('HELIX_PROXY_COLD_START_RETRIES', 2, 0, 5),
    retryBudgetMs: numeric('HELIX_PROXY_RETRY_BUDGET_MS', 25_000, 0, 120_000),
  };
}

function validateOrigin(base) {
  let url;
  try {
    url = new URL(base);
  } catch {
    return { error: 'HELIX_API_BASE_URL is not a valid URL.' };
  }
  const loopback = ['127.0.0.1', 'localhost', '::1', '[::1]'].includes(url.hostname);
  if (url.protocol !== 'https:' && !(url.protocol === 'http:' && loopback)) {
    return { error: 'HELIX_API_BASE_URL must use https:// (plain http is allowed only for loopback test servers).' };
  }
  if (url.pathname !== '/' || url.search || url.hash) {
    return { error: 'HELIX_API_BASE_URL must be an origin without a path, query or fragment.' };
  }
  return { origin: url };
}

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

function requestIdFrom(headers) {
  const candidate = String((headers && headers['x-request-id']) || '');
  if (/^[A-Za-z0-9._-]{8,80}$/.test(candidate)) return candidate;
  return crypto.randomUUID();
}

function logLine(event) {
  try {
    process.stdout.write(JSON.stringify({ ts: new Date().toISOString(), service: 'helix-proxy', ...event }) + '\n');
  } catch {
    /* logging must never break a request */
  }
}

function setCommonHeaders(res) {
  res.setHeader('X-Content-Type-Options', 'nosniff');
  res.setHeader('Referrer-Policy', 'no-referrer');
  res.setHeader('X-Upstream-Host', 'helix-backend');
}

function sendJson(res, status, payload) {
  if (res.headersSent) {
    try { res.end(); } catch { /* already closed */ }
    return;
  }
  res.statusCode = status;
  setCommonHeaders(res);
  res.setHeader('Content-Type', 'application/json; charset=utf-8');
  res.setHeader('Cache-Control', 'no-store, max-age=0');
  res.end(JSON.stringify(payload));
}

/**
 * Build the single error envelope used by the proxy layer.
 * `error` stays a plain string for UI compatibility; `code` is the stable key.
 */
function apiError(message, code, options = {}) {
  const payload = {
    error: message,
    code,
    request_id: options.requestId || null,
    retryable: Boolean(options.retryable),
    layer: 'vercel-proxy',
  };
  if (options.status) payload.upstream_status = options.status;
  if (options.contentType) payload.upstream_content_type = options.contentType;
  if (options.detail) payload.detail = options.detail;
  if (options.retryAfter) payload.retry_after_s = options.retryAfter;
  return payload;
}

/** Identify known HTML interstitials that prove the app never handled the request. */
function classifyHtml(bodyText) {
  const text = (bodyText || '').slice(0, SNIFF_BYTES);
  if (/Render\s*-\s*Application loading/i.test(text) || /data:image\/svg\+xml[^"]*Block A/.test(text)) {
    return { kind: 'render_cold_start', retryable: true, message: 'The HELIX backend is starting up (Render cold start).' };
  }
  if (/The page could not be found/i.test(text)) {
    return { kind: 'vercel_not_found', retryable: false, message: 'The Vercel deployment has no route for this API path.' };
  }
  if (/Application failed to respond|Internal Server Error|Bad Gateway/i.test(text)) {
    return { kind: 'upstream_error_page', retryable: true, message: 'The HELIX backend returned an HTML error page instead of JSON.' };
  }
  return { kind: 'unexpected_html', retryable: false, message: 'The HELIX backend returned HTML instead of JSON.' };
}

async function readLimited(stream, limit) {
  const chunks = [];
  let total = 0;
  for await (const chunk of stream) {
    total += chunk.length;
    if (total > limit) return { tooLarge: true, body: Buffer.concat(chunks), total };
    chunks.push(chunk);
  }
  return { tooLarge: false, body: Buffer.concat(chunks), total };
}

async function sleep(ms) {
  await new Promise((resolve) => setTimeout(resolve, ms));
}

function isStreamingPath(pathname) {
  return pathname.endsWith('/stream');
}

// ---------------------------------------------------------------------------
// proxy
// ---------------------------------------------------------------------------

function createProxy(options = {}) {
  const env = options.env || process.env;
  const log = options.log || logLine;

  return async function proxyHelix(req, res, allowedPath, rewritePath) {
    const requestId = requestIdFrom(req.headers);
    const started = Date.now();
    const method = (req.method || 'GET').toUpperCase();
    const rawUrl = req.url || '/';
    let pathname = '/';
    try {
      pathname = new URL(rawUrl, 'https://vercel.invalid').pathname;
    } catch {
      pathname = '/';
    }
    const finish = (status, extra = {}) => log({
      request_id: requestId,
      method,
      path: pathname,
      status,
      duration_ms: Date.now() - started,
      ...extra,
    });

    res.setHeader('X-Request-ID', requestId);

    if (!ALLOWED_METHODS.has(method)) {
      sendJson(res, 405, apiError('Method not allowed.', 'method_not_allowed', { requestId }));
      finish(405);
      return;
    }

    const config = readConfig(env);
    if (!config.base || !config.username || !config.password) {
      const missing = ['HELIX_API_BASE_URL', 'HELIX_AUTH_USERNAME', 'HELIX_AUTH_PASSWORD']
        .filter((name) => !String(env[name] || '').trim());
      sendJson(res, 503, apiError(
        'The HELIX API proxy is not configured in this deployment.',
        'proxy_not_configured',
        { requestId, retryable: false, detail: `Missing Vercel environment variables: ${missing.join(', ')}.` },
      ));
      finish(503, { error_code: 'proxy_not_configured' });
      return;
    }

    const origin = validateOrigin(config.base);
    if (origin.error) {
      sendJson(res, 500, apiError(origin.error, 'invalid_proxy_configuration', { requestId }));
      finish(500, { error_code: 'invalid_proxy_configuration' });
      return;
    }

    if (typeof allowedPath === 'function' && !allowedPath(pathname)) {
      sendJson(res, 404, apiError('Unknown HELIX API route.', 'route_not_found', { requestId }));
      finish(404, { error_code: 'route_not_found' });
      return;
    }

    let body;
    if (!['GET', 'HEAD'].includes(method)) {
      let buffered;
      try {
        buffered = await readLimited(req, config.maxBodyBytes);
      } catch {
        sendJson(res, 400, apiError('Could not read the request body.', 'body_read_failed', { requestId }));
        finish(400, { error_code: 'body_read_failed' });
        return;
      }
      if (buffered.tooLarge) {
        sendJson(res, 413, apiError(
          `Request body exceeds the ${Math.round(config.maxBodyBytes / 1024)} KiB proxy limit.`,
          'request_too_large',
          { requestId },
        ));
        finish(413, { error_code: 'request_too_large' });
        return;
      }
      body = buffered.body;
    }

    const target = new URL(rawUrl, origin.origin);
    const rewritten = typeof rewritePath === 'function' ? rewritePath(pathname) : pathname;
    target.pathname = rewritten;
    // Never let a client control the upstream search string; only chat/status
    // style endpoints are used and they take their arguments in the body.
    target.search = '';

    const authHeader = 'Basic ' + Buffer.from(`${config.username}:${config.password}`, 'utf8').toString('base64');
    const streaming = isStreamingPath(pathname);
    const controller = new AbortController();
    const totalTimeoutMs = streaming ? config.streamTimeoutMs : config.requestTimeoutMs;
    const timer = setTimeout(() => controller.abort(new Error('timeout')), totalTimeoutMs);
    let clientGone = false;
    const onClose = () => {
      if (!res.writableEnded) {
        clientGone = true;
        controller.abort(new Error('client_disconnected'));
      }
    };
    res.on('close', onClose);

    const upstreamHeaders = {
      Authorization: authHeader,
      Accept: streaming ? 'text/event-stream' : 'application/json',
      'X-Forwarded-Proto': 'https',
      'X-Request-ID': requestId,
      'User-Agent': 'helix-vercel-proxy/2',
    };
    if (req.headers['content-type']) upstreamHeaders['Content-Type'] = req.headers['content-type'];
    if (req.headers.accept && !streaming) upstreamHeaders.Accept = req.headers.accept;
    if (body && body.length) upstreamHeaders['Content-Length'] = String(body.length);

    const deadline = Date.now() + config.retryBudgetMs;
    const cleanup = () => {
      clearTimeout(timer);
      res.off('close', onClose);
    };
    let upstream = null;
    let attempts = 0;
    try {
      for (attempts = 1; attempts <= config.coldStartRetries + 1; attempts += 1) {
        upstream = await fetch(target, {
          method,
          headers: upstreamHeaders,
          body: body && body.length ? body : undefined,
          redirect: 'manual',
          signal: controller.signal,
        });
        const probeType = String(upstream.headers.get('content-type') || '').toLowerCase();
        const retryableReply = upstream.status >= 500 || upstream.status === 200;
        if (retryableReply && !JSON_TYPE.test(probeType) && !SSE_TYPE.test(probeType) && !IMAGE_TYPE.test(probeType)) {
          const sniff = await readLimited(upstream.body, SNIFF_BYTES).catch(() => ({ body: Buffer.alloc(0) }));
          const interstitial = classifyHtml(sniff.body ? sniff.body.toString('utf8') : '');
          const budgetLeft = deadline - Date.now();
          if (interstitial.retryable && attempts <= config.coldStartRetries && budgetLeft > 1_500) {
            const wait = Math.min(4_000, 700 * attempts);
            log({
              request_id: requestId,
              event: 'upstream_cold_start_retry',
              attempt: attempts,
              upstream_status: upstream.status,
              wait_ms: wait,
            });
            await sleep(Math.min(wait, Math.max(0, budgetLeft - 500)));
            continue;
          }
          sendJson(res, 503, apiError(interstitial.message, interstitial.kind, {
            requestId,
            status: upstream.status,
            contentType: probeType || 'unknown',
            retryable: interstitial.retryable,
            detail: 'The browser received JSON, not the upstream HTML page. Retry once the backend has finished starting.',
          }));
          finish(503, { error_code: interstitial.kind, upstream_status: upstream.status, attempts });
          return;
        }
        break;
      }

      const type = String(upstream.headers.get('content-type') || '').toLowerCase();
      const status = upstream.status;

      // ---- JSON -----------------------------------------------------------
      const isErrorPayload = status >= 400 && !SSE_TYPE.test(type) && !IMAGE_TYPE.test(type) && !streaming;
      if (JSON_TYPE.test(type) || isErrorPayload) {
        let text = '';
        try {
          const buffered = await readLimited(upstream.body, MAX_JSON_BYTES);
          if (buffered.tooLarge) {
            sendJson(res, 502, apiError('Backend response exceeded the proxy JSON limit.', 'upstream_too_large', { requestId, status }));
            finish(502, { error_code: 'upstream_too_large', upstream_status: status });
            return;
          }
          text = buffered.body.toString('utf8');
        } catch {
          if (clientGone) { finish(499, { error_code: 'client_disconnected' }); return; }
          sendJson(res, 502, apiError('Backend response was interrupted.', 'upstream_stream_interrupted', { requestId, retryable: true, status }));
          finish(502, { error_code: 'upstream_stream_interrupted', upstream_status: status });
          return;
        }
        try {
          JSON.parse(text || '{}');
        } catch {
          const classified = classifyHtml(text);
          sendJson(res, 502, apiError(classified.message, 'upstream_non_json', {
            requestId,
            status,
            contentType: type || 'unknown',
            retryable: classified.retryable,
            detail: 'Upstream returned a body that is not valid JSON.',
          }));
          finish(502, { error_code: 'upstream_non_json', upstream_status: status });
          return;
        }
        res.statusCode = status;
        setCommonHeaders(res);
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        res.setHeader('Cache-Control', 'no-store, max-age=0');
        res.setHeader('X-Upstream-Status', String(status));
        res.end(text);
        finish(status, { upstream_status: status, bytes: Buffer.byteLength(text), kind: 'json' });
        return;
      }

      // ---- server-sent events --------------------------------------------
      if (SSE_TYPE.test(type)) {
        res.statusCode = status;
        setCommonHeaders(res);
        res.setHeader('Content-Type', 'text/event-stream; charset=utf-8');
        res.setHeader('Cache-Control', 'no-cache, no-transform');
        res.setHeader('Connection', 'keep-alive');
        res.setHeader('X-Accel-Buffering', 'no');
        res.setHeader('X-Upstream-Status', String(status));
        if (typeof res.flushHeaders === 'function') res.flushHeaders();
        try {
          for await (const chunk of upstream.body) {
            if (clientGone) break;
            if (!res.write(chunk)) {
              await new Promise((resolve) => {
                const done = () => { res.off('drain', done); res.off('close', done); resolve(); };
                res.on('drain', done);
                res.on('close', done);
              });
            }
          }
          if (!res.writableEnded) res.end();
          finish(status, { upstream_status: status, kind: 'sse', client_disconnected: clientGone });
        } catch {
          if (!res.writableEnded) {
            // The status line is already sent: emit a terminal error event the
            // client understands instead of silently truncating the stream.
            try {
              res.write(`data: ${JSON.stringify({ error: 'Stream interrupted by the proxy.', code: 'upstream_stream_interrupted', request_id: requestId })}\n\n`);
            } catch { /* socket already gone */ }
            res.end();
          }
          finish(status, { upstream_status: status, kind: 'sse', error_code: 'upstream_stream_interrupted' });
        }
        return;
      }

      // ---- images ---------------------------------------------------------
      if (IMAGE_TYPE.test(type)) {
        res.statusCode = status;
        setCommonHeaders(res);
        res.setHeader('Content-Type', type);
        res.setHeader('Cache-Control', 'private, max-age=300');
        try {
          for await (const chunk of upstream.body) {
            if (clientGone) break;
            if (!res.write(chunk)) await new Promise((resolve) => res.once('drain', resolve));
          }
        } catch { /* client abort */ }
        if (!res.writableEnded) res.end();
        finish(status, { upstream_status: status, kind: 'image' });
        return;
      }

      // ---- anything else ---------------------------------------------------
      sendJson(res, 502, apiError(
        'The HELIX backend returned an unsupported content type.',
        'upstream_unsupported_type',
        { requestId, status, contentType: type || 'unknown', retryable: false },
      ));
      finish(502, { error_code: 'upstream_unsupported_type', upstream_status: status });
    } catch (error) {
      if (clientGone) {
        finish(499, { error_code: 'client_disconnected' });
        return;
      }
      const reason = controller.signal.reason;
      const timedOut = reason instanceof Error && /timeout/i.test(reason.message);
      const status = timedOut ? 504 : 502;
      sendJson(res, status, apiError(
        timedOut ? 'The HELIX backend did not respond in time.' : 'Could not reach the HELIX backend.',
        timedOut ? 'upstream_timeout' : 'upstream_unreachable',
        {
          requestId,
          retryable: true,
          detail: timedOut
            ? `No upstream response within ${Math.round(totalTimeoutMs / 1000)}s.`
            : 'DNS, TCP or TLS failure between Vercel and the backend origin.',
        },
      ));
      finish(status, { error_code: timedOut ? 'upstream_timeout' : 'upstream_unreachable', attempts });
    } finally {
      cleanup();
    }
  };
}

module.exports = createProxy();
module.exports.createProxy = createProxy;
module.exports.classifyHtml = classifyHtml;
module.exports.apiError = apiError;
module.exports.readConfig = readConfig;
