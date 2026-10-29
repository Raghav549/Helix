'use strict';

// Serves generated PNGs through the same authenticated proxy as the API.
const proxy = require('../../../lib/helix-proxy');

module.exports = (req, res) => proxy(
  req,
  res,
  (path) => /^\/api\/out\/web_images\/[0-9a-f]{32}\.png$/.test(path),
);
module.exports.config = { maxDuration: 60, api: { bodyParser: false } };
