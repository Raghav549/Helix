import assert from 'node:assert/strict';
import test from 'node:test';
import {
  MAX_CONTEXT_MESSAGES,
  SessionRepository,
  appendMessage,
  contextFor,
  emptySession,
  parseSessions,
  sortSessions,
  updateMessage,
  type StorageLike,
} from '../src/state/sessions';

function memoryStorage(): StorageLike & { data: Map<string, string> } {
  const data = new Map<string, string>();
  return {
    data,
    getItem: async (key) => data.get(key) ?? null,
    setItem: async (key, value) => {
      data.set(key, value);
    },
    removeItem: async (key) => {
      data.delete(key);
    },
  };
}

test('appending the first user message titles the session', () => {
  const session = emptySession(1000);
  const next = appendMessage(session, { id: 'm1', role: 'user', content: 'Explain KV caching', createdAt: 1001 });
  assert.equal(next.title, 'Explain KV caching');
  const later = appendMessage(next, { id: 'm2', role: 'user', content: 'second', createdAt: 1002 });
  assert.equal(later.title, 'Explain KV caching');
});

test('updateMessage patches only the target message', () => {
  let session = appendMessage(emptySession(1), { id: 'a', role: 'assistant', content: '', createdAt: 2, streaming: true });
  session = updateMessage(session, 'a', { content: 'hi', streaming: false });
  assert.equal(session.messages[0]!.content, 'hi');
  assert.equal(session.messages[0]!.streaming, false);
});

test('contextFor drops failures and empty placeholders and trims to the tail', () => {
  let session = emptySession(1);
  session = appendMessage(session, { id: 'u', role: 'user', content: 'hello', createdAt: 2 });
  session = appendMessage(session, { id: 'a', role: 'assistant', content: '', createdAt: 3, error: 'boom' });
  session = appendMessage(session, { id: 'b', role: 'assistant', content: '', createdAt: 4, streaming: true });
  assert.deepEqual(contextFor(session), [{ role: 'user', content: 'hello' }]);

  let long = emptySession(1);
  for (let i = 0; i < 20; i += 1) {
    long = appendMessage(long, { id: `m${i}`, role: i % 2 === 0 ? 'user' : 'assistant', content: `msg ${i}`, createdAt: i });
  }
  const context = contextFor(long);
  assert.equal(context.length, MAX_CONTEXT_MESSAGES);
  assert.equal(context[context.length - 1]!.content, 'msg 19');
});

test('parseSessions survives corrupt or hostile payloads', () => {
  assert.deepEqual(parseSessions('not json'), []);
  assert.deepEqual(parseSessions('{"a":1}'), []);
  const parsed = parseSessions(
    JSON.stringify([
      { id: 's1', title: '', messages: [{ role: 'user', content: 'kept' }, { role: 'hacker', content: 'dropped' }, null] },
      { id: null, messages: [] },
    ]),
  );
  assert.equal(parsed.length, 1);
  assert.equal(parsed[0]!.title, 'New chat');
  assert.equal(parsed[0]!.messages.length, 1);
  assert.equal(parsed[0]!.messages[0]!.content, 'kept');
});

test('sortSessions orders by most recent update', () => {
  const a = { ...emptySession(1), id: 'a', updatedAt: 10 };
  const b = { ...emptySession(1), id: 'b', updatedAt: 30 };
  assert.deepEqual(sortSessions([a, b]).map((s) => s.id), ['b', 'a']);
});

test('repository round-trips through storage and clears', async () => {
  const storage = memoryStorage();
  const repo = new SessionRepository(storage);
  const session = appendMessage(emptySession(5), { id: 'm', role: 'user', content: 'persist me', createdAt: 6 });
  await repo.save([session]);
  const reloaded = await new SessionRepository(storage).load();
  assert.equal(reloaded[0]!.title, 'persist me');
  await repo.clear();
  assert.deepEqual(await repo.load(), []);
  assert.equal(storage.data.has('helix.sessions.v1'), false);
});

test('repository strips the streaming flag on save', async () => {
  const storage = memoryStorage();
  const repo = new SessionRepository(storage);
  const session = appendMessage(emptySession(5), { id: 'm', role: 'assistant', content: 'x', createdAt: 6, streaming: true });
  await repo.save([session]);
  const stored = JSON.parse(storage.data.get('helix.sessions.v1')!) as Array<{ messages: Array<{ streaming?: boolean }> }>;
  assert.equal(stored[0]!.messages[0]!.streaming, false);
});
