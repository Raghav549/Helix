import assert from 'node:assert/strict';
import test from 'node:test';
import { extractReply, greetingFor, normalizeBaseUrl, parseSseChunk, titleFor } from '../src/lib/sse';

test('parses complete events and keeps the trailing partial buffer', () => {
  const { events, rest } = parseSseChunk('data: {"delta":"he"}\n\ndata: {"delta":"llo"}\n\ndata: {"del');
  assert.deepEqual(events, [{ delta: 'he' }, { delta: 'llo' }]);
  assert.equal(rest, 'data: {"del');
});

test('handles CRLF frames and multi-line data fields', () => {
  const { events } = parseSseChunk('data: {"delta":"a"}\r\n\r\n');
  assert.deepEqual(events, [{ delta: 'a' }]);
  const multi = parseSseChunk('data: {"meta":\ndata: {"model":"x"}}\n\n');
  assert.equal(multi.events.length, 1);
  assert.deepEqual(multi.events[0], { meta: { model: 'x' } });
});

test('reports malformed events instead of dropping them', () => {
  const { events } = parseSseChunk('data: {oops}\n\n');
  assert.equal(events.length, 1);
  assert.equal(events[0]!.code, 'malformed_event');
});

test('ignores comment and empty frames', () => {
  const { events } = parseSseChunk(': keep-alive\n\nevent: message\n\ndata: {"done":true}\n\n');
  assert.deepEqual(events, [{ done: true }]);
});

test('extractReply concatenates deltas only', () => {
  assert.equal(extractReply([{ delta: 'a' }, { meta: {} }, { delta: 'b' }, { done: true }]), 'ab');
});

test('titleFor produces a stable short title', () => {
  assert.equal(titleFor('  hello   world '), 'hello world');
  assert.equal(titleFor(''), 'New chat');
  assert.equal(titleFor('x'.repeat(60)).length, 48);
});

test('greetingFor buckets the day', () => {
  assert.match(greetingFor(9), /Good morning\./);
  assert.equal(greetingFor(15, 'Ada'), 'Good afternoon, Ada.');
  assert.match(greetingFor(23), /Good evening\./);
});

test('normalizeBaseUrl accepts https and loopback http only', () => {
  assert.equal(normalizeBaseUrl('https://helix-web-lemon.vercel.app/'), 'https://helix-web-lemon.vercel.app');
  assert.equal(normalizeBaseUrl('http://127.0.0.1:5000'), 'http://127.0.0.1:5000');
  assert.equal(normalizeBaseUrl('http://10.0.2.2:8080'), 'http://10.0.2.2:8080');
  assert.equal(normalizeBaseUrl('http://evil.example.com'), null);
  assert.equal(normalizeBaseUrl('https://example.com/api'), null);
  assert.equal(normalizeBaseUrl('not a url'), null);
  assert.equal(normalizeBaseUrl(''), null);
});
