import assert from 'node:assert/strict';
import test from 'node:test';
import { formatBytes, formatRelativeTime, summarizeHealth, truncate } from '../src/lib/format';
import { formatAnalysis, quickActionPrompt } from '../src/lib/analysis';
import { basicAuthHeader, encodeBase64 } from '../src/lib/base64';

test('relative time is human and non-negative', () => {
  const now = 1_000_000_000;
  assert.equal(formatRelativeTime(now - 5_000, now), 'just now');
  assert.equal(formatRelativeTime(now + 60_000, now), 'just now');
  assert.equal(formatRelativeTime(now - 5 * 60_000, now), '5m ago');
  assert.equal(formatRelativeTime(now - 3 * 3_600_000, now), '3h ago');
  assert.equal(formatRelativeTime(now - 26 * 3_600_000, now), 'yesterday');
  assert.equal(formatRelativeTime(now - 5 * 86_400_000, now), '5d ago');
});

test('truncate and formatBytes', () => {
  assert.equal(truncate('  a  b  ', 10), 'a b');
  assert.equal(truncate('abcdefghij', 5), 'abcd…');
  assert.equal(formatBytes(-1), 'unknown');
  assert.equal(formatBytes(512), '512 B');
  assert.equal(formatBytes(2048), '2.0 KB');
  assert.equal(formatBytes(5 * 1024 * 1024), '5.0 MB');
});

test('summarizeHealth never claims readiness without model_ready', () => {
  assert.deepEqual(summarizeHealth(null), { label: 'Offline', tone: 'down' });
  assert.deepEqual(summarizeHealth({ model_ready: true }), { label: 'Ready', tone: 'ready' });
  assert.deepEqual(summarizeHealth({ model_ready: false, degraded_reason: 'no bundle' }), { label: 'Degraded', tone: 'degraded' });
  assert.deepEqual(summarizeHealth({ model_ready: false }), { label: 'Unknown', tone: 'degraded' });
});

test('formatAnalysis only prints fields that exist', () => {
  const text = formatAnalysis({ brightness: 0.42, description: 'uniform grey field' }, 'shot.png');
  assert.match(text, /shot\.png/);
  assert.match(text, /Brightness: 0\.42/);
  assert.match(text, /Description/);
  assert.match(text, /not by the chat model/i);
  assert.doesNotMatch(text, /Contrast/);
});

test('formatAnalysis reports errors and empty payloads honestly', () => {
  assert.match(formatAnalysis({ error: 'unsupported format' }), /Analysis failed: unsupported format/);
  assert.match(formatAnalysis(null), /No analysis was returned/);
  assert.match(formatAnalysis({}), /no recognised fields/);
  assert.match(formatAnalysis({ caption_warning: 'synthetic only' }), /synthetic only/);
});

test('quick action prompts are concrete', () => {
  assert.match(quickActionPrompt('summarize'), /Summarize the following text/);
  assert.match(quickActionPrompt('math'), /Differentiate/);
  assert.equal(quickActionPrompt('image'), '');
});

test('base64 and basic auth headers match the RFC test vector', () => {
  assert.equal(encodeBase64('Aladdin:open sesame'), 'QWxhZGRpbjpvcGVuIHNlc2FtZQ==');
  assert.equal(encodeBase64('a'), 'YQ==');
  assert.equal(encodeBase64('héllo'), 'aMOpbGxv');
  assert.equal(basicAuthHeader('u', 'p'), 'Basic dTpw');
});
