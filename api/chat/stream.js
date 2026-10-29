'use strict';

// Explicit registration for the streaming chat endpoint. Keeping a dedicated
// function guarantees server-sent events are proxied without buffering.
const proxy = require('../../lib/helix-proxy');

module.exports = (req, res) => proxy(
  req,
  res,
  (path) => path === '/api/chat/stream' || path === '/chat/stream',
  (path) => (path.startsWith('/api/') ? path : '/api' + path),
);
module.exports.config = { maxDuration: 60, api: { bodyParser: false } };
