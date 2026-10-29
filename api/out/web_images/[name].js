'use strict';
const proxy = require('../../../lib/helix-proxy');

module.exports = (req, res) => proxy(req, res, path => /^\/out\/web_images\/[0-9a-f]{32}\.png$/.test(path));
module.exports.config = { maxDuration: 60, api: { bodyParser: false } };
