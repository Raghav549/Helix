'use strict';
const proxy = require('../../lib/helix-proxy');

// Explicit route ensures the streaming chat endpoint is registered on Vercel.
module.exports = (req, res) => proxy(
  req,
  res,
  path => path === '/api/chat/stream' || path === '/chat/stream',
  path => path.startsWith('/api/') ? path : '/api' + path
);
module.exports.config = { maxDuration: 60, api: { bodyParser: false } };
