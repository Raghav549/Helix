'use strict';
const proxy = require('../lib/helix-proxy');

module.exports = (req, res) => proxy(req, res, path => path.startsWith('/api/'));
module.exports.config = { maxDuration: 60, api: { bodyParser: false } };
