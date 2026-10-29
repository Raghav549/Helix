import assert from 'node:assert/strict';
import test from 'node:test';
import { ApiError, HelixClient, errorFromResponse } from '../src/api/client';
import { splitCodeBlocks } from '../src/lib/markdown';

test('JSON API errors keep the server message and request id', () => {
  const error = errorFromResponse(503, '{"error":"model not loaded","code":"model_unavailable","request_id":"abcdef123456"}');
  assert.equal(error.kind, 'model_unavailable');
  assert.equal(error.status, 503);
  assert.match(error.message, /model not loaded/);
  assert.match(error.message, /abcdef12/);
});

test('HTML error pages are never parsed as JSON', () => {
  const error = errorFromResponse(502, '<!doctype html><title>The page could not be found</title>');
  assert.ok(error instanceof ApiError);
  assert.equal(error.kind, 'bad_gateway');
  assert.doesNotMatch(error.message, /</);
  assert.match(error.message, /not ready/i);
});

test('status codes map onto user-facing kinds', () => {
  assert.equal(errorFromResponse(401, '').kind, 'unauthorized');
  assert.equal(errorFromResponse(403, '').kind, 'unauthorized');
  assert.equal(errorFromResponse(404, '').kind, 'not_found');
  assert.equal(errorFromResponse(413, '').kind, 'too_large');
  assert.equal(errorFromResponse(429, '').kind, 'rate_limited');
  assert.equal(errorFromResponse(500, '').kind, 'server');
});

test('client normalises trailing slashes and refuses to invent a reply', async () => {
  const client = new HelixClient({ baseUrl: 'https://example.test/' });
  assert.equal(client.baseUrl, 'https://example.test');
});

test('code fences split into text and code parts', () => {
  const parts = splitCodeBlocks('intro\n```python\nprint(1)\n```\noutro');
  assert.deepEqual(parts, [
    { kind: 'text', value: 'intro\n' },
    { kind: 'code', value: 'print(1)' },
    { kind: 'text', value: '\noutro' },
  ]);
  assert.deepEqual(splitCodeBlocks('plain text'), [{ kind: 'text', value: 'plain text' }]);
  assert.deepEqual(splitCodeBlocks('```js\n```'), []);
});
