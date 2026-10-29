'use strict';

const MAX_BODY_BYTES = 4 * 1024 * 1024;

module.exports = async function proxyHelix(req, res, allowedPath, rewritePath = path => path) {
  const base = process.env.HELIX_API_BASE_URL;
  const username = process.env.HELIX_AUTH_USERNAME;
  const password = process.env.HELIX_AUTH_PASSWORD;
  if (!base || !username || !password) {
    res.statusCode = 503;
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.end(JSON.stringify({ error: 'HELIX proxy is not configured. Set HELIX_API_BASE_URL, HELIX_AUTH_USERNAME, and HELIX_AUTH_PASSWORD in Vercel.' }));
    return;
  }

  let targetBase;
  try { targetBase = new URL(base); } catch {
    res.statusCode = 500; res.end('Invalid HELIX_API_BASE_URL'); return;
  }
  if (targetBase.protocol !== 'https:' || targetBase.pathname !== '/' || targetBase.search || targetBase.hash) {
    res.statusCode = 500; res.end('HELIX_API_BASE_URL must be an HTTPS origin'); return;
  }

  const pathname = new URL(req.url, 'https://vercel.invalid').pathname;
  if (!allowedPath(pathname)) {
    res.statusCode = 404; res.end('Not found'); return;
  }

  let body;
  if (!['GET', 'HEAD'].includes(req.method || 'GET')) {
    const chunks = [];
    let total = 0;
    for await (const chunk of req) {
      total += chunk.length;
      if (total > MAX_BODY_BYTES) {
        res.statusCode = 413;
        res.setHeader('Content-Type', 'application/json; charset=utf-8');
        res.end(JSON.stringify({ error: 'Request exceeds the Vercel proxy 4 MiB limit.' }));
        return;
      }
      chunks.push(chunk);
    }
    body = Buffer.concat(chunks);
  }

  const target = new URL(req.url, targetBase);
  target.pathname = rewritePath(pathname);
  const headers = {
    'Authorization': 'Basic ' + Buffer.from(username + ':' + password, 'utf8').toString('base64'),
    'Accept': req.headers.accept || 'application/json',
    'X-Forwarded-Proto': 'https',
  };
  if (req.headers['content-type']) headers['Content-Type'] = req.headers['content-type'];
  let upstream;
  try {
    upstream = await fetch(target, {
      method: req.method,
      headers,
      body: body && body.length ? body : undefined,
      redirect: 'manual',
      signal: AbortSignal.timeout(55_000),
    });
  } catch (error) {
    res.statusCode = 502;
    res.setHeader('Content-Type', 'application/json; charset=utf-8');
    res.end(JSON.stringify({ error: 'Could not reach HELIX backend.', detail: error.name === 'TimeoutError' ? 'Backend request timed out.' : 'Upstream connection failed.' }));
    return;
  }

  res.statusCode = upstream.status;
  for (const key of ['content-type', 'cache-control', 'content-disposition', 'etag', 'last-modified']) {
    const value = upstream.headers.get(key);
    if (value) res.setHeader(key, value);
  }
  res.setHeader('X-Content-Type-Options', 'nosniff');
  if (upstream.headers.get('content-type') === 'text/event-stream') {
    res.setHeader('Cache-Control', 'no-cache, no-transform');
    res.setHeader('X-Accel-Buffering', 'no');
  }
  if (req.method === 'HEAD' || !upstream.body) { res.end(); return; }
  try {
    for await (const chunk of upstream.body) {
      if (!res.write(chunk)) await new Promise(resolve => res.once('drain', resolve));
    }
    res.end();
  } catch {
    if (!res.headersSent) res.statusCode = 502;
    res.end();
  }
};
