'use strict';

// Catch-all Vercel serverless entrypoint for the HELIX API.
// Every /api/* request is authenticated with server-side Basic credentials and
// converted to a structured JSON error when the backend misbehaves, so the
// browser never parses an HTML page as JSON.
const proxy = require('../lib/helix-proxy');

module.exports = (req, res) => proxy(req, res, (path) => path.startsWith('/api/'));
module.exports.config = { maxDuration: 60, api: { bodyParser: false } };
