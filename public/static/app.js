'use strict';

/* ==========================================================================
   Beta by helix — web client
   Plain ES2020, no build step, no third-party runtime. Every request goes to a
   same-origin /api path served by the authenticated Vercel proxy (or Flask
   directly). Non-JSON responses are converted into precise, actionable errors
   instead of confusing JSON parse failures.
   ========================================================================== */

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => Array.from(root.querySelectorAll(selector));

const STORAGE = { threads: 'helix.threads.v2', settings: 'helix.settings.v2', active: 'helix.active.v2' };
const LIMITS = { files: 6 * 1024 * 1024, message: 4000, history: 12 };

const state = {
  view: 'chat',
  threads: [],
  activeThreadId: null,
  streaming: null,
  attachment: null,
  status: null,
  activity: {},
  settings: {
    theme: 'system',
    reduceMotion: false,
    displayName: '',
    useSources: true,
    useMemory: false,
    agentMode: false,
  },
};

/* ------------------------------------------------------------------ utils */

function safeParse(raw, fallback) {
  try { return JSON.parse(raw); } catch { return fallback; }
}

function loadStore() {
  const threads = safeParse(localStorage.getItem(STORAGE.threads) || '[]', []);
  state.threads = Array.isArray(threads) ? threads.filter((thread) => thread && typeof thread.id === 'string') : [];
  const settings = safeParse(localStorage.getItem(STORAGE.settings) || '{}', {});
  state.settings = { ...state.settings, ...(settings && typeof settings === 'object' ? settings : {}) };
  const active = localStorage.getItem(STORAGE.active);
  state.activeThreadId = state.threads.some((thread) => thread.id === active) ? active : (state.threads[0]?.id || null);
}

function saveStore() {
  try {
    localStorage.setItem(STORAGE.threads, JSON.stringify(state.threads.slice(0, 40)));
    localStorage.setItem(STORAGE.settings, JSON.stringify(state.settings));
    if (state.activeThreadId) localStorage.setItem(STORAGE.active, state.activeThreadId);
  } catch {
    toast('This browser refused to store conversation data.', 'error');
  }
}

function uid() {
  if (window.crypto?.randomUUID) return window.crypto.randomUUID();
  return 't-' + Date.now().toString(36) + Math.random().toString(36).slice(2, 8);
}

function formatTime(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
}

function truncate(text, max) {
  const clean = String(text || '').replace(/\s+/g, ' ').trim();
  return clean.length > max ? clean.slice(0, max - 1) + '…' : clean;
}

/* ------------------------------------------------------------------ toast */

function toast(message, kind = 'info', timeout = 5200) {
  const stack = $('#toast-stack');
  if (!stack) return;
  const node = document.createElement('div');
  node.className = `toast is-${kind}`;
  node.setAttribute('role', kind === 'error' ? 'alert' : 'status');
  const icon = document.createElement('span');
  icon.innerHTML = `<svg viewBox="0 0 24 24" aria-hidden="true"><use href="#${kind === 'error' ? 'i-alert' : kind === 'success' ? 'i-check' : 'i-spark'}"/></svg>`;
  const text = document.createElement('span');
  text.textContent = message;
  node.append(icon.firstElementChild, text);
  stack.append(node);
  setTimeout(() => node.remove(), timeout);
}

/* -------------------------------------------------------------------- api */

class ApiError extends Error {
  constructor(message, options = {}) {
    super(message);
    this.name = 'ApiError';
    this.status = options.status ?? 0;
    this.code = options.code || 'unknown_error';
    this.requestId = options.requestId || null;
    this.layer = options.layer || null;
    this.retryable = Boolean(options.retryable);
    this.contentType = options.contentType || null;
    this.detail = options.detail || null;
    this.upstreamStatus = options.upstreamStatus ?? null;
  }
}

function describeTransportFailure(response, excerpt) {
  const status = response.status;
  if (status === 404) return 'The API route does not exist on this deployment (HTTP 404).';
  if (status === 401 || status === 403) return 'The deployment proxy rejected the request (authentication).';
  if (status === 413) return 'The request was too large.';
  if (status === 429) return 'Too many requests right now. Wait a moment and retry.';
  if (status >= 500) return `The backend returned an error page instead of JSON (HTTP ${status}).`;
  return `The server returned ${response.headers.get('content-type') || 'an unknown content type'} instead of JSON (HTTP ${status}).${excerpt ? ' ' + excerpt : ''}`;
}

async function parseResponse(response) {
  const contentType = String(response.headers.get('content-type') || '').toLowerCase();
  const requestId = response.headers.get('x-request-id');
  if (contentType.includes('application/json')) {
    const text = await response.text();
    try {
      return { kind: 'json', data: text ? JSON.parse(text) : {}, requestId };
    } catch {
      throw new ApiError('The server sent malformed JSON.', {
        status: response.status, code: 'malformed_json', requestId, contentType,
      });
    }
  }
  const text = await response.text().catch(() => '');
  const excerpt = truncate(text.replace(/<[^>]*>/g, ' '), 160);
  throw new ApiError(describeTransportFailure(response, excerpt), {
    status: response.status,
    code: contentType.includes('text/html') ? 'html_instead_of_json' : 'unexpected_content_type',
    requestId, contentType, retryable: response.status >= 500 || response.status === 429,
  });
}

async function requestJson(path, options = {}) {
  const { method = 'GET', body, json, signal, timeoutMs = 120000 } = options;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(new DOMException('timeout', 'TimeoutError')), timeoutMs);
  const onAbort = () => controller.abort(signal?.reason);
  if (signal) signal.addEventListener('abort', onAbort, { once: true });
  const init = { method, signal: controller.signal, headers: { Accept: 'application/json' } };
  if (json !== undefined) {
    init.headers['Content-Type'] = 'application/json';
    init.body = JSON.stringify(json);
  } else if (body) {
    init.body = body;
  }
  try {
    const response = await fetch(path, init);
    const parsed = await parseResponse(response);
    if (!response.ok) {
      const data = parsed.data || {};
      throw new ApiError(data.error || `Request failed (HTTP ${response.status}).`, {
        status: response.status, code: data.code, requestId: data.request_id || parsed.requestId,
        layer: data.layer, retryable: Boolean(data.retryable), detail: data.detail,
        upstreamStatus: data.upstream_status,
      });
    }
    return parsed.data;
  } catch (error) {
    if (error instanceof ApiError) throw error;
    if (error.name === 'AbortError' || error.name === 'TimeoutError') {
      throw new ApiError('The request was cancelled or timed out.', { code: 'cancelled', retryable: true });
    }
    throw new ApiError('The browser could not reach the HELIX API.', {
      code: 'network_error', retryable: true, detail: error.message,
    });
  } finally {
    clearTimeout(timer);
    if (signal) signal.removeEventListener('abort', onAbort);
  }
}

/**
 * Consume an SSE chat stream.
 * Handles CRLF, multi-line data fields, error events and mid-stream aborts.
 */
async function streamChat({ message, history, useSources, useMemory, signal, onMeta, onDelta }) {
  const response = await fetch('/api/chat/stream', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify({
      message,
      history: history.slice(-LIMITS.history),
      use_sources: useSources,
      use_memory: useMemory,
      max_new_tokens: 120,
    }),
    signal,
  });

  const contentType = String(response.headers.get('content-type') || '').toLowerCase();
  if (!response.ok || !contentType.includes('text/event-stream')) {
    let data = {};
    try { data = await parseResponse(response).then((parsed) => parsed.data); }
    catch (error) {
      if (error instanceof ApiError) throw error;
    }
    throw new ApiError(data.error || `Chat failed (HTTP ${response.status}).`, {
      status: response.status, code: data.code, requestId: data.request_id || response.headers.get('x-request-id'),
      layer: data.layer, retryable: Boolean(data.retryable), detail: data.detail,
    });
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = '';
  let reply = '';
  let meta = {};

  const handleEvent = (raw) => {
    const dataLines = raw.split('\n').filter((line) => line.startsWith('data:')).map((line) => line.slice(5).trim());
    if (!dataLines.length) return false;
    const payload = safeParse(dataLines.join('\n'), null);
    if (!payload) return false;
    if (payload.error) {
      throw new ApiError(payload.error, {
        code: payload.code || 'generation_failed', requestId: payload.request_id || null,
        layer: 'helix-backend', retryable: true,
      });
    }
    if (payload.meta) { meta = payload.meta; onMeta?.(meta); }
    if (typeof payload.delta === 'string') { reply += payload.delta; onDelta?.(payload.delta, reply); }
    return Boolean(payload.done);
  };

  for (;;) {
    const { value, done } = await reader.read();
    buffer += decoder.decode(value || new Uint8Array(), { stream: !done });
    buffer = buffer.replace(/\r\n/g, '\n');
    let boundary;
    while ((boundary = buffer.indexOf('\n\n')) !== -1) {
      const raw = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      if (handleEvent(raw)) { onMeta?.({ ...meta, done: true }); return { reply, meta }; }
    }
    if (done) {
      if (buffer.trim()) handleEvent(buffer.trim());
      return { reply, meta };
    }
  }
}

/* --------------------------------------------------------------- activity */

function setActivity(kind, mode) {
  // mode: 'active' | 'done' | 'idle'
  state.activity[kind] = mode;
  const card = $(`.activity-card[data-activity="${kind}"]`);
  if (!card) return;
  card.classList.toggle('is-active', mode === 'active');
  card.classList.toggle('is-done', mode === 'done');
}

function resetActivity(keep = []) {
  ['thinking', 'generating', 'image', 'searching'].forEach((kind) => {
    if (!keep.includes(kind)) setActivity(kind, 'idle');
  });
}

/* ------------------------------------------------------------------ views */

const VIEW_TITLES = { chat: 'Chat', image: 'Image studio', code: 'Code', research: 'Research', settings: 'Settings' };

function showView(view, options = {}) {
  if (!VIEW_TITLES[view]) view = 'chat';
  state.view = view;
  $$('.view').forEach((section) => {
    const active = section.id === `view-${view}`;
    section.hidden = !active;
    section.classList.toggle('is-active', active);
  });
  $$('.nav-item[data-view]').forEach((item) => {
    const active = item.dataset.view === view;
    item.classList.toggle('is-active', active);
    if (active) item.setAttribute('aria-current', 'page');
    else item.removeAttribute('aria-current');
  });
  $('#view-title').textContent = VIEW_TITLES[view];
  document.title = view === 'chat' ? 'Beta by helix' : `${VIEW_TITLES[view]} · Beta by helix`;
  if (view !== 'chat') closeNav();
  if (view === 'research') { refreshExperiments(); refreshRetrievalStatus(); }
  if (view === 'settings') { renderMemory().catch(() => {}); }
  if (!options.silent && location.hash !== `#${view}`) {
    history.replaceState(null, '', `#${view}`);
  }
  window.scrollTo({ top: 0, behavior: state.settings.reduceMotion ? 'auto' : 'smooth' });
}

function openNav() {
  document.body.classList.add('nav-open');
  $('#sidebar-scrim').hidden = false;
  $('#menu-button').setAttribute('aria-expanded', 'true');
  $('#sidebar-close').focus();
}

function closeNav() {
  document.body.classList.remove('nav-open');
  $('#sidebar-scrim').hidden = true;
  $('#menu-button').setAttribute('aria-expanded', 'false');
}

/* --------------------------------------------------------------- theming */

function applyTheme() {
  const media = window.matchMedia('(prefers-color-scheme: dark)');
  const resolved = state.settings.theme === 'system' ? (media.matches ? 'dark' : 'light') : state.settings.theme;
  document.body.dataset.theme = resolved;
  document.body.dataset.motion = state.settings.reduceMotion ? 'reduced' : 'full';
  $('#theme-icon').innerHTML = `<use href="#${resolved === 'dark' ? 'i-moon' : 'i-sun'}"/>`;
  $('#theme-toggle').setAttribute('aria-label', resolved === 'dark' ? 'Switch to light appearance' : 'Switch to dark appearance');
  $$('[data-theme-choice]').forEach((button) => {
    button.setAttribute('aria-checked', String(button.dataset.themeChoice === state.settings.theme));
  });
  const reduce = $('#reduce-motion');
  if (reduce) reduce.checked = state.settings.reduceMotion;
  if (media.addEventListener) {
    media.addEventListener('change', () => { if (state.settings.theme === 'system') applyTheme(); });
  }
}

function cycleTheme() {
  const order = ['light', 'dark', 'system'];
  state.settings.theme = order[(order.indexOf(state.settings.theme) + 1) % order.length];
  saveStore();
  applyTheme();
  toast(`Appearance: ${state.settings.theme}`, 'info', 2200);
}

/* ------------------------------------------------------------- greeting -- */

function renderGreeting() {
  const hour = new Date().getHours();
  const part = hour < 5 ? 'night' : hour < 12 ? 'morning' : hour < 18 ? 'afternoon' : 'evening';
  const name = (state.settings.displayName || '').trim();
  $('#greeting-title').textContent = name ? `Good ${part}, ${name}.` : `Good ${part}.`;
}

/* -------------------------------------------------------------- history -- */

function activeThread() {
  return state.threads.find((thread) => thread.id === state.activeThreadId) || null;
}

function ensureThread() {
  let thread = activeThread();
  if (thread) return thread;
  thread = { id: uid(), title: 'New chat', createdAt: Date.now(), updatedAt: Date.now(), messages: [] };
  state.threads.unshift(thread);
  state.activeThreadId = thread.id;
  saveStore();
  return thread;
}

function createThread() {
  if (state.streaming) stopGeneration();
  const thread = { id: uid(), title: 'New chat', createdAt: Date.now(), updatedAt: Date.now(), messages: [] };
  state.threads.unshift(thread);
  state.activeThreadId = thread.id;
  saveStore();
  renderHistory();
  renderThread();
  $('#composer-input').value = '';
  clearAttachment();
  showView('chat');
  $('#composer-input').focus();
}

function selectThread(id) {
  if (state.streaming) stopGeneration();
  state.activeThreadId = id;
  saveStore();
  renderHistory();
  renderThread();
  showView('chat');
}

function deleteThread(id) {
  state.threads = state.threads.filter((thread) => thread.id !== id);
  if (state.activeThreadId === id) state.activeThreadId = state.threads[0]?.id || null;
  saveStore();
  renderHistory();
  renderThread();
}

function renderHistory() {
  const list = $('#history-list');
  list.replaceChildren();
  if (!state.threads.length) {
    const empty = document.createElement('p');
    empty.className = 'history-empty';
    empty.textContent = 'Your conversations stay in this browser.';
    list.append(empty);
    return;
  }
  state.threads.slice(0, 40).forEach((thread) => {
    const row = document.createElement('div');
    row.className = 'history-item' + (thread.id === state.activeThreadId ? ' is-active' : '');
    row.setAttribute('role', 'listitem');

    const open = document.createElement('button');
    open.type = 'button';
    open.className = 'history-title';
    open.dataset.thread = thread.id;
    open.textContent = thread.title || 'New chat';
    open.title = `${thread.title || 'New chat'} · ${formatTime(thread.updatedAt)}`;
    open.addEventListener('click', () => selectThread(thread.id));

    const remove = document.createElement('button');
    remove.type = 'button';
    remove.className = 'history-delete';
    remove.setAttribute('aria-label', `Delete conversation ${thread.title || 'New chat'}`);
    remove.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><use href="#i-trash"/></svg>';
    remove.addEventListener('click', (event) => { event.stopPropagation(); deleteThread(thread.id); });

    row.append(open, remove);
    list.append(row);
  });
}

/* --------------------------------------------------------------- thread --- */

function scrollThread() {
  const thread = $('#thread');
  if (!thread) return;
  const nearBottom = thread.scrollHeight - thread.scrollTop - window.innerHeight < 400;
  if (nearBottom || document.activeElement === document.body) {
    window.scrollTo({ top: document.body.scrollHeight, behavior: state.settings.reduceMotion ? 'auto' : 'smooth' });
  }
}

function messageNode(message) {
  const wrap = document.createElement('article');
  wrap.className = `message ${message.role}`;
  const avatar = document.createElement('span');
  avatar.className = 'message-avatar';
  avatar.setAttribute('aria-hidden', 'true');
  avatar.textContent = message.role === 'user' ? 'You' : 'β';

  const body = document.createElement('div');
  body.className = 'message-body';

  const role = document.createElement('div');
  role.className = 'message-role';
  role.textContent = message.role === 'user' ? 'You' : message.kind === 'tool' ? 'Tool result' : 'Beta by helix';

  const text = document.createElement('div');
  text.className = 'message-text';
  text.textContent = message.content;

  body.append(role, text);

  if (message.sources?.length) {
    const meta = document.createElement('div');
    meta.className = 'message-meta';
    message.sources.forEach((source, index) => {
      const chip = document.createElement('a');
      chip.className = 'citation';
      chip.href = `#sources`;
      chip.textContent = `[${index + 1}] ${source.title || source.source}`;
      chip.title = `${source.citation} · relevance ${source.score}`;
      chip.addEventListener('click', (event) => { event.preventDefault(); showSources(source); });
      meta.append(chip);
    });
    body.append(meta);
  }
  if (message.kind === 'tool') {
    const badge = document.createElement('div');
    badge.className = 'message-meta';
    const chip = document.createElement('span');
    chip.className = 'chip';
    chip.textContent = message.tool || 'tool';
    badge.append(chip);
    body.append(badge);
  }
  wrap.append(avatar, body);
  return wrap;
}

function renderThread() {
  const container = $('#thread');
  container.replaceChildren();
  const thread = activeThread();
  if (!thread || !thread.messages.length) {
    const empty = document.createElement('div');
    empty.className = 'thread-empty';
    empty.id = 'thread-empty';
    const icon = document.createElement('span');
    icon.className = 'empty-orbit';
    icon.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><use href="#i-spark"/></svg>';
    const text = document.createElement('p');
    const strong = document.createElement('strong');
    strong.textContent = 'No messages yet. ';
    text.append(strong, document.createTextNode('Ask anything, or start from a quick action above.'));
    empty.append(icon, text);
    container.append(empty);
    return;
  }
  thread.messages.forEach((message) => container.append(messageNode(message)));
}

function showSources(source) {
  toast(`${source.citation} — ${truncate(source.snippet || '', 220)}`, 'info', 9000);
}

function appendMessage(message) {
  const thread = ensureThread();
  thread.messages.push(message);
  thread.updatedAt = Date.now();
  if (thread.title === 'New chat' && message.role === 'user') {
    thread.title = truncate(message.content, 48);
  }
  saveStore();
  renderHistory();
  if ($('#thread-empty')) $('#thread-empty').remove();
  const node = messageNode(message);
  $('#thread').append(node);
  scrollThread();
  return node;
}

/* ----------------------------------------------------------------- chat --- */

function setStreamingUI(streaming) {
  $('#send-button').hidden = streaming;
  $('#stop-button').hidden = !streaming;
  $('#composer-input').disabled = false;
  $$('.quick-card').forEach((card) => { card.disabled = streaming; });
}

function stopGeneration() {
  if (!state.streaming) return;
  state.streaming.controller.abort(new DOMException('user stopped', 'AbortError'));
}

async function sendMessage(text) {
  const message = String(text || '').trim();
  if (!message) return;
  if (message.length > LIMITS.message) { toast('Messages are limited to 4000 characters.', 'error'); return; }
  if (state.streaming) return;

  if (state.settings.agentMode) {
    appendMessage({ role: 'user', content: message });
    $('#composer-input').value = '';
    autoGrow($('#composer-input'));
    await runAgent(message);
    return;
  }

  if (state.attachment) {
    const attachment = state.attachment;
    appendMessage({ role: 'user', content: `${message}\n\n[attached: ${attachment.name}]` });
    clearAttachment();
    $('#composer-input').value = '';
    await analyzeAttachment(attachment, message);
    return;
  }

  const thread = ensureThread();
  const history = thread.messages.filter((item) => item.role === 'user' || item.role === 'assistant').slice(-LIMITS.history);
  appendMessage({ role: 'user', content: message });
  $('#composer-input').value = '';
  autoGrow($('#composer-input'));

  const output = appendMessage({ role: 'assistant', content: '' });
  const textNode = $('.message-text', output);
  textNode.classList.add('cursor');

  const controller = new AbortController();
  state.streaming = { controller, threadId: thread.id };
  setStreamingUI(true);
  resetActivity(['thinking']);
  setActivity('thinking', 'active');

  let reply = '';
  try {
    const result = await streamChat({
      message,
      history,
      useSources: state.settings.useSources,
      useMemory: state.settings.useMemory,
      signal: controller.signal,
      onMeta: (meta) => {
        if (meta.sources?.length) setActivity('searching', 'done');
        setActivity('thinking', 'done');
        setActivity('generating', 'active');
      },
      onDelta: (_delta, full) => {
        reply = full;
        textNode.textContent = full;
        scrollThread();
      },
    });
    reply = result.reply || reply;
    textNode.classList.remove('cursor');
    if (!reply.trim()) {
      textNode.textContent = 'The model produced no text. This is a limitation of the small bundled checkpoint.';
      textNode.classList.add('muted');
    }
    const sources = result.meta?.sources || [];
    if (sources.length) {
      const body = $('.message-body', output);
      const meta = document.createElement('div');
      meta.className = 'message-meta';
      sources.forEach((source, index) => {
        const chip = document.createElement('a');
        chip.className = 'citation';
        chip.href = '#sources';
        chip.textContent = `[${index + 1}] ${source.title || source.source}`;
        chip.title = `${source.citation} · relevance ${source.score}`;
        chip.addEventListener('click', (event) => { event.preventDefault(); showSources(source); });
        meta.append(chip);
      });
      body.append(meta);
    }
    const current = state.threads.find((item) => item.id === state.streaming.threadId) || thread;
    current.messages.push({ role: 'assistant', content: reply, sources });
    current.updatedAt = Date.now();
    saveStore();
    renderHistory();
    setActivity('generating', 'done');
    refreshHealth(false);
  } catch (error) {
    textNode.classList.remove('cursor');
    output.remove();
    if (error.name === 'AbortError' || (error.code === 'cancelled')) {
      if (reply) {
        const current = state.threads.find((item) => item.id === state.streaming?.threadId) || thread;
        current.messages.push({ role: 'assistant', content: reply });
        saveStore();
        toast('Generation stopped.', 'info', 2200);
      } else {
        toast('Generation stopped before any text arrived.', 'info', 2200);
      }
    } else {
      renderChatError(thread, message, error);
    }
    setActivity('thinking', 'idle');
    setActivity('generating', 'idle');
  } finally {
    state.streaming = null;
    setStreamingUI(false);
    setTimeout(() => resetActivity(), 2500);
    $('#composer-input').focus();
  }
}

function describeFailure(error) {
  const code = String(error?.code || '');
  const upstreamStatus = Number(error?.upstreamStatus || error?.upstream_status || 0);
  if (code === 'render_cold_start') return 'The backend is starting up. Retry in a few seconds.';
  if (code === 'upstream_non_json' && upstreamStatus === 404) {
    return 'This deployment\'s backend is older than this client (the API route is missing).';
  }
  if (code === 'upstream_non_json' || code === 'html_instead_of_json' || code === 'unexpected_html') {
    return 'The deployment returned a web page instead of JSON.';
  }
  if (code === 'model_unavailable') return 'The model is not loaded on this deployment.';
  if (code === 'rate_limited' || code === 'compute_busy') return 'The server is busy. Retry in a moment.';
  if (code === 'unauthorized') return 'This deployment rejected the request credentials.';
  if (code === 'route_not_found') return 'This API route does not exist on the deployment.';
  return 'Unable to generate a reply.';
}

function renderChatError(thread, message, error) {
  const node = document.createElement('div');
  node.className = 'message-error';
  const title = document.createElement('strong');
  title.textContent = describeFailure(error);
  const detail = document.createElement('span');
  detail.textContent = error.message;
  const meta = document.createElement('code');
  const bits = [];
  if (error.status) bits.push(`HTTP ${error.status}`);
  if (error.code) bits.push(`code: ${error.code}`);
  if (error.layer) bits.push(`layer: ${error.layer}`);
  if (error.requestId) bits.push(`request id: ${error.requestId}`);
  if (error.detail) bits.push(error.detail);
  meta.textContent = bits.join(' · ') || 'No further detail available.';

  const actions = document.createElement('div');
  actions.className = 'error-actions';
  const retry = document.createElement('button');
  retry.type = 'button';
  retry.className = 'secondary';
  retry.textContent = 'Retry';
  retry.addEventListener('click', () => { node.remove(); sendMessage(message); });
  const diagnostics = document.createElement('button');
  diagnostics.type = 'button';
  diagnostics.className = 'secondary';
  diagnostics.textContent = 'Open diagnostics';
  diagnostics.addEventListener('click', runDiagnostics);
  actions.append(retry, diagnostics);

  node.append(title, detail, meta, actions);
  $('#thread').append(node);
  scrollThread();
  toast(title.textContent, 'error', 7000);
}

async function analyzeAttachment(file, prompt) {
  resetActivity(['thinking']);
  setActivity('thinking', 'active');
  const form = new FormData();
  form.append('image', file);
  const node = appendMessage({ role: 'assistant', kind: 'tool', tool: 'image/analyze', content: 'Analyzing the image…' });
  const textNode = $('.message-text', node);
  try {
    const data = await requestJson('/api/image/analyze', { method: 'POST', body: form });
    const lines = [];
    lines.push(`Size: ${data.width}×${data.height}`);
    if (data.measurements) lines.push(`Measurements: ${JSON.stringify(data.measurements)}`);
    if (data.prediction) lines.push(`Synthetic-domain prediction: ${JSON.stringify(data.prediction)}`);
    if (data.synthetic_domain_caption) lines.push(`Learned caption: ${data.synthetic_domain_caption}`);
    if (data.caption_warning) lines.push(`Warning: ${data.caption_warning}`);
    textNode.textContent = lines.join('\n');
    const thread = activeThread();
    if (thread) {
      const stored = thread.messages[thread.messages.length - 1];
      if (stored && stored.kind === 'tool') stored.content = textNode.textContent;
    }
    saveStore();
    setActivity('thinking', 'done');
  } catch (error) {
    textNode.textContent = `Image analysis failed: ${error.message}`;
    setActivity('thinking', 'idle');
    toast('Image analysis failed.', 'error');
  }
  if (prompt) toast('The bundled model cannot see images; only the analysis tool ran.', 'info', 6000);
}

/* ------------------------------------------------------------ attachments */

function clearAttachment() {
  state.attachment = null;
  $('#attachment-tray').replaceChildren();
  $('#attachment-tray').hidden = true;
  $('#file-input').value = '';
}

function setAttachment(file) {
  if (!file) return;
  const allowed = ['image/png', 'image/jpeg', 'image/webp'];
  if (!allowed.includes(file.type)) { toast('Attach a PNG, JPEG or WebP image.', 'error'); return; }
  if (file.size > LIMITS.files) { toast('Images must be 6 MiB or smaller.', 'error'); return; }
  state.attachment = file;
  const tray = $('#attachment-tray');
  tray.replaceChildren();
  const chip = document.createElement('span');
  chip.className = 'attachment-chip';
  const preview = document.createElement('img');
  preview.alt = '';
  preview.src = URL.createObjectURL(file);
  const label = document.createElement('span');
  label.textContent = `${truncate(file.name, 26)} · ${Math.round(file.size / 1024)} KB`;
  const remove = document.createElement('button');
  remove.type = 'button';
  remove.setAttribute('aria-label', 'Remove attachment');
  remove.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><use href="#i-close"/></svg>';
  remove.addEventListener('click', clearAttachment);
  chip.append(preview, label, remove);
  tray.append(chip);
  tray.hidden = false;
}

/* ---------------------------------------------------------------- health -- */

async function refreshHealth(verbose = false) {
  const dot = $('#status-dot');
  const label = $('#status-label');
  const healthDot = $('#health-dot');
  const healthText = $('#health-text');
  try {
    const health = await requestJson('/api/health', { timeoutMs: 15000 });
    state.status = health;
    const ready = Boolean(health.model_ready);
    dot.className = `dot ${ready ? 'is-ready' : 'is-degraded'}`;
    label.textContent = ready ? `Backend ready · ${health.version || ''}` : 'Backend degraded';
    healthDot.className = `dot ${ready ? 'is-ready' : 'is-degraded'}`;
    healthText.textContent = ready ? 'Ready' : 'Degraded';
    $('#health-pill').title = ready
      ? `Backend ready · build ${health.build || 'unknown'}`
      : `Degraded: ${health.degraded_reason || 'model unavailable'}`;
    if (!ready && verbose) toast(`Backend degraded: ${health.degraded_reason || 'model unavailable'}`, 'error', 8000);
    const retrieval = await requestJson('/api/status', { timeoutMs: 15000 }).catch(() => null);
    if (retrieval) applyCapabilities(retrieval);
  } catch (error) {
    dot.className = 'dot is-down';
    label.textContent = error.code === 'upstream_non_json' ? 'Backend outdated'
      : error.code === 'render_cold_start' ? 'Backend starting' : 'Backend unreachable';
    healthDot.className = 'dot is-down';
    healthText.textContent = error.code === 'upstream_non_json' ? 'Backend outdated'
      : error.code === 'render_cold_start' ? 'Starting'
      : error.code === 'html_instead_of_json' ? 'Proxy error' : 'Offline';
    $('#health-pill').title = `${error.message}${error.requestId ? ` (request ${error.requestId})` : ''}`;
    if (verbose) toast(error.message, 'error', 8000);
  }
}

function applyCapabilities(status) {
  state.status = { ...(state.status || {}), ...status };
  const searchCard = $('.quick-card[data-action="sources"]');
  const available = Boolean(status.retrieval?.configured);
  if (searchCard) {
    searchCard.disabled = false;
    searchCard.title = available ? 'Search the local document index' : 'No local document index is configured on this deployment';
    searchCard.classList.toggle('is-unavailable', !available);
  }
  const searchingCard = $('.activity-card[data-activity="searching"]');
  if (searchingCard) searchingCard.classList.toggle('is-unavailable', !available);
  $('#retrieval-status').textContent = available
    ? `Index ready: ${status.retrieval.documents} documents, ${status.retrieval.chunks} chunks.`
    : (status.retrieval?.hint || 'No local document index is configured; answers use model knowledge only.');
}

/* ------------------------------------------------------------- quick acts */

async function runQuickAction(action) {
  switch (action) {
    case 'image':
    case 'code':
    case 'math':
    case 'sources':
      showView(action === 'math' || action === 'sources' ? 'research' : action);
      if (action === 'sources') $('#search-query').focus();
      return;
    case 'summarize': {
      const selection = String(window.getSelection() || '').trim();
      const value = $('#composer-input').value.trim() || selection;
      if (!value) {
        toast('Paste the text you want summarized into the composer first.', 'info', 5000);
        $('#composer-input').focus();
        return;
      }
      $('#composer-input').value = `Summarize the following text in five bullet points:\n\n${value}`;
      autoGrow($('#composer-input'));
      $('#composer-input').focus();
      return;
    }
    case 'brainstorm': {
      $('#composer-input').value = 'Brainstorm five distinct approaches to ';
      autoGrow($('#composer-input'));
      const input = $('#composer-input');
      input.focus();
      input.setSelectionRange(input.value.length, input.value.length);
      return;
    }
    default:
      toast('That action is not wired to a backend capability.', 'info');
  }
}

/* ----------------------------------------------------------------- agent -- */

function parseSeries(text) {
  return String(text || '')
    .split(/[,\s]+/)
    .map((piece) => piece.trim())
    .filter(Boolean)
    .map(Number)
    .filter((value) => Number.isFinite(value));
}

async function runAgent(message) {
  const node = appendMessage({ role: 'assistant', kind: 'tool', tool: 'orchestrator', content: 'Planning…' });
  const textNode = $('.message-text', node);
  setActivity('thinking', 'active');
  try {
    const body = { message, use_sources: state.settings.useSources };
    const numbers = parseSeries(message);
    if (numbers.length >= 3) body.series = numbers;
    const data = await requestJson('/api/agent', { method: 'POST', json: body, timeoutMs: 180000 });
    textNode.textContent = data.answer || '(no answer)';
    const bodyNode = $('.message-body', node);
    const meta = document.createElement('div');
    meta.className = 'message-meta';
    (data.tool_calls || []).forEach((call) => {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = `${call.tool} · ${call.ok ? 'ok' : 'failed'} · ${Math.round(call.duration_ms)}ms`;
      meta.append(chip);
    });
    (data.refused || []).forEach((refusal) => {
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = 'refused';
      chip.title = refusal;
      meta.append(chip);
    });
    if (data.classification) {
      const kindChip = document.createElement('span');
      kindChip.className = 'chip';
      kindChip.textContent = data.classification;
      meta.prepend(kindChip);
    }
    bodyNode.append(meta);
    const thread = activeThread();
    if (thread) {
      const stored = thread.messages[thread.messages.length - 1];
      if (stored && stored.kind === 'tool') {
        stored.content = textNode.textContent;
        stored.tool = 'orchestrator';
      }
      thread.updatedAt = Date.now();
      saveStore();
    }
    setActivity('thinking', 'done');
  } catch (error) {
    textNode.textContent = `Agent request failed: ${error.message}${error.requestId ? ` (request ${error.requestId})` : ''}`;
    setActivity('thinking', 'idle');
  } finally {
    setTimeout(() => resetActivity(), 2500);
  }
}

async function runForecast(event) {
  event.preventDefault();
  const output = $('#forecast-output');
  const series = parseSeries($('#forecast-series').value);
  if (series.length < 3) { output.textContent = 'Enter at least three numbers.'; return; }
  const button = $('#forecast-submit');
  button.disabled = true;
  output.textContent = 'Fitting…';
  try {
    const data = await requestJson('/api/research/forecast', {
      method: 'POST',
      json: {
        series,
        horizon: Number($('#forecast-horizon').value || 3),
        interval: Number($('#forecast-interval').value || 0.9),
      },
    });
    const lines = data.predictions.map((point) => (
      `step ${point.x.toFixed(0)}: ${point.yhat} (${Math.round(data.interval * 100)}% interval ${point.lower} … ${point.upper})`
    ));
    output.textContent = [
      ...lines,
      '',
      `hold-out MAE ${data.holdout.mae} · RMSE ${data.holdout.rmse} · last-value baseline MAE ${data.holdout.baseline_mae_last_value}`,
      `beats baseline: ${data.holdout.beats_baseline} · backend: ${data.backend}`,
      '',
      ...data.assumptions.map((line) => `• ${line}`),
    ].join('\n');
  } catch (error) {
    output.textContent = `Forecast failed: ${error.message}`;
  } finally {
    button.disabled = false;
  }
}

/* ----------------------------------------------------------------- image -- */

const IMAGE_MODE_HELP = {
  physics: 'The renderer supports one sphere or cube, a material and a light direction. Output is deterministic for a given seed.',
  physics_guided: 'A rendered reference guides the 32×32 diffusion model, so the reference adds information the prompt alone does not carry.',
  learned: 'A tiny from-scratch diffusion model. Prompt adherence and visual quality are very limited.',
};

async function generateImage(event) {
  event.preventDefault();
  const button = $('#image-submit');
  button.disabled = true;
  $('#image-chip').textContent = 'working';
  setActivity('image', 'active');
  try {
    const data = await requestJson('/api/image/generate', {
      method: 'POST',
      json: {
        prompt: $('#image-prompt').value,
        mode: $('#image-mode').value,
        seed: Number($('#image-seed').value || 0),
      },
    });
    const image = document.createElement('img');
    image.src = data.image_url;
    image.alt = `Generated image: ${$('#image-prompt').value}`;
    image.width = 256;
    $('#image-output').replaceChildren(image);
    $('#image-caption').textContent = `${data.notice} Mode: ${data.mode} · seed ${data.seed}.`;
    const download = $('#image-download');
    download.href = data.image_url;
    download.download = `helix-${data.seed}.png`;
    download.hidden = false;
    $('#image-chip').textContent = 'done';
    setActivity('image', 'done');
  } catch (error) {
    $('#image-caption').textContent = `Generation failed: ${error.message}`;
    $('#image-chip').textContent = 'error';
    setActivity('image', 'idle');
  } finally {
    button.disabled = false;
    setTimeout(() => resetActivity(), 3000);
  }
}

async function analyzeUpload(edit) {
  const file = $('#upload-input').files[0];
  if (!file) { toast('Choose an image first.', 'error'); return; }
  const form = new FormData();
  form.append('image', file);
  if (edit) {
    form.append('prompt', $('#image-prompt').value);
    form.append('seed', $('#image-seed').value || '1337');
    form.append('strength', $('#edit-strength').value);
  }
  $('#analysis-output').textContent = edit ? 'Generating an edit…' : 'Analyzing…';
  setActivity('image', 'active');
  try {
    const data = await requestJson(edit ? '/api/image/edit' : '/api/image/analyze', { method: 'POST', body: form });
    if (edit) {
      const image = document.createElement('img');
      image.src = data.image_url;
      image.alt = 'Edited image';
      $('#image-output').replaceChildren(image);
      $('#analysis-output').textContent = data.notice;
    } else {
      $('#analysis-output').textContent = JSON.stringify(data, null, 2);
    }
    setActivity('image', 'done');
  } catch (error) {
    $('#analysis-output').textContent = `Failed: ${error.message}`;
    setActivity('image', 'idle');
  } finally {
    setTimeout(() => resetActivity(), 2500);
  }
}

/* ------------------------------------------------------------------ code -- */

async function codeAction(kind) {
  const code = $('#code-input').value;
  const chip = $('#code-chip');
  const output = $('#code-output');
  if (kind === 'execute' && !$('#code-approval').checked) {
    output.textContent = 'Tick the approval box before each execution.';
    chip.textContent = 'not run';
    return;
  }
  chip.textContent = 'working';
  try {
    if (kind === 'analyze') {
      const data = await requestJson('/api/code/analyze', { method: 'POST', json: { code } });
      output.textContent = JSON.stringify(data, null, 2);
      chip.textContent = data.syntax_error ? 'syntax error' : 'analyzed';
      $('#code-note').textContent = 'Static AST analysis only; nothing was executed.';
    } else if (kind === 'explain') {
      const data = await requestJson('/api/code/explain', { method: 'POST', json: { code } });
      output.textContent = `${data.reply || '(no completion)'}\n\n[Unverified model text — not an execution result.]`;
      chip.textContent = 'unverified';
      $('#code-note').textContent = 'Model explanation only; nothing was executed.';
    } else if (kind === 'tests') {
      const data = await requestJson('/api/code/test-skeleton', { method: 'POST', json: { code } });
      output.textContent = data.tests;
      chip.textContent = 'skeleton';
      $('#code-note').textContent = data.notice;
    } else if (kind === 'patch') {
      const data = await requestJson('/api/code/patch', {
        method: 'POST', json: { original: code, modified: $('#code-modified').value },
      });
      output.textContent = data.diff;
      chip.textContent = 'preview';
      $('#code-note').textContent = data.note;
    } else if (kind === 'execute') {
      $('#code-approval').checked = false;
      const data = await requestJson('/api/code/execute', { method: 'POST', json: { code, approved: true } });
      output.textContent = `${data.output || '(no output)'}${data.error ? '\n\n' + data.error : ''}\n\n[isolation: ${data.isolation}; ${Number(data.duration_s || 0).toFixed(2)}s]`;
      chip.textContent = data.ok ? 'complete' : 'failed';
      $('#code-note').textContent = 'Execution ran in the sandbox; output above is real interpreter output.';
    }
  } catch (error) {
    output.textContent = `${error.message}${error.requestId ? `\n(request ${error.requestId})` : ''}`;
    chip.textContent = 'error';
  }
}

/* -------------------------------------------------------------- research -- */

async function runMath(event) {
  event.preventDefault();
  const button = $('#math-submit');
  button.disabled = true;
  $('#math-chip').textContent = 'working';
  try {
    const data = await requestJson('/api/research/math', {
      method: 'POST',
      json: {
        operation: $('#math-operation').value,
        expression: $('#math-expression').value,
        variable: $('#math-variable').value,
        point: $('#math-point').value,
        save_experiment: $('#save-experiment').checked,
      },
    });
    const steps = Array.isArray(data.steps) ? data.steps.join('\n') : '';
    $('#math-output').textContent = `${data.result}\n\n${steps}${data.experiment_id ? `\n\nexperiment: ${data.experiment_id}` : ''}`;
    $('#math-chip').textContent = 'sympy';
    if (data.experiment_id) refreshExperiments();
  } catch (error) {
    $('#math-output').textContent = `Failed: ${error.message}`;
    $('#math-chip').textContent = 'error';
  } finally {
    button.disabled = false;
  }
}

async function runSearch(event) {
  event.preventDefault();
  const query = $('#search-query').value.trim();
  if (!query) return;
  const results = $('#search-results');
  results.replaceChildren();
  setActivity('searching', 'active');
  try {
    const data = await requestJson('/api/research/search', { method: 'POST', json: { query, k: 6 } });
    if (!data.results.length) {
      const empty = document.createElement('p');
      empty.className = 'field-help';
      empty.textContent = 'No indexed chunk matched that query.';
      results.append(empty);
    }
    data.results.forEach((item, index) => {
      const node = document.createElement('div');
      node.className = 'result-item';
      const title = document.createElement('strong');
      title.textContent = `[${index + 1}] ${item.title}`;
      const source = document.createElement('span');
      source.className = 'result-source';
      source.textContent = `${item.citation} · relevance ${item.score}`;
      const text = document.createElement('p');
      text.textContent = item.snippet;
      node.append(title, source, text);
      results.append(node);
    });
    setActivity('searching', 'done');
  } catch (error) {
    const node = document.createElement('p');
    node.className = 'field-help';
    node.textContent = `Search failed: ${error.message}`;
    results.append(node);
    setActivity('searching', 'idle');
  } finally {
    setTimeout(() => resetActivity(), 3000);
  }
}

async function refreshRetrievalStatus() {
  const label = $('#retrieval-status');
  try {
    const data = await requestJson('/api/research/sources', { timeoutMs: 15000 });
    label.textContent = data.configured
      ? `Index ready: ${data.index?.documents ?? data.documents.length} documents, ${data.index?.chunks ?? 0} chunks (checksum ${(data.index?.checksum || '').slice(0, 12)}…).`
      : 'No local document index is configured on this deployment. Answer text will not cite sources.';
  } catch (error) {
    label.textContent = `Could not read the document index: ${error.message}`;
  }
}

async function refreshExperiments() {
  const list = $('#experiment-list');
  list.replaceChildren();
  try {
    const data = await requestJson('/api/research/experiments', { timeoutMs: 20000 });
    if (!data.experiments?.length) {
      const empty = document.createElement('p');
      empty.className = 'field-help';
      empty.textContent = 'No recorded experiments yet.';
      list.append(empty);
      return;
    }
    data.experiments.slice(0, 12).forEach((experiment) => {
      const node = document.createElement('div');
      node.className = 'result-item';
      const title = document.createElement('strong');
      title.textContent = experiment.name || 'experiment';
      const detail = document.createElement('span');
      detail.className = 'result-source';
      detail.textContent = `${experiment.stage || 'stage'} · ${experiment.device || 'cpu'}${typeof experiment.best_loss === 'number' ? ` · loss ${experiment.best_loss.toFixed(3)}` : ''}`;
      node.append(title, detail);
      list.append(node);
    });
  } catch (error) {
    const node = document.createElement('p');
    node.className = 'field-help';
    node.textContent = error.message;
    list.append(node);
  }
}

/* ------------------------------------------------------------ diagnostics -- */

async function runDiagnostics() {
  const output = $('#diagnostics-output');
  const chip = $('#diagnostics-chip');
  chip.textContent = 'running';
  output.textContent = 'Querying deployment endpoints…';
  const report = {};
  for (const path of ['/api/health', '/api/version', '/api/status', '/api/diagnostics']) {
    try {
      const data = await requestJson(path, { timeoutMs: 20000 });
      report[path] = data;
    } catch (error) {
      report[path] = {
        error: error.message, status: error.status, code: error.code,
        request_id: error.requestId, layer: error.layer, detail: error.detail,
      };
    }
  }
  output.textContent = JSON.stringify(report, null, 2);
  const failed = Object.values(report).filter((entry) => entry && entry.error).length;
  chip.textContent = failed ? `${failed} failed` : 'ok';
  await refreshHealth();
}

/* ----------------------------------------------------------------- memory -- */

async function renderMemory() {
  const list = $('#memory-list');
  list.replaceChildren();
  try {
    const data = await requestJson('/api/memory', { timeoutMs: 15000 });
    if (!data.memories?.length) {
      const empty = document.createElement('p');
      empty.className = 'field-help';
      empty.textContent = 'No saved notes.';
      list.append(empty);
      return;
    }
    data.memories.forEach((memory) => {
      const node = document.createElement('div');
      node.className = 'result-item';
      const text = document.createElement('span');
      text.textContent = memory.text;
      const actions = document.createElement('div');
      actions.className = 'row';
      const remove = document.createElement('button');
      remove.type = 'button';
      remove.className = 'text-button';
      remove.textContent = 'Delete';
      remove.addEventListener('click', async () => {
        try {
          await requestJson(`/api/memory/${memory.id}`, { method: 'DELETE' });
          await renderMemory();
        } catch (error) { toast(error.message, 'error'); }
      });
      actions.append(remove);
      node.append(text, actions);
      list.append(node);
    });
  } catch (error) {
    const node = document.createElement('p');
    node.className = 'field-help';
    node.textContent = `Could not load notes: ${error.message}`;
    list.append(node);
  }
}

async function addMemory() {
  const input = $('#memory-input');
  const text = input.value.trim();
  if (!text) return;
  if (!$('#memory-consent').checked) { toast('Tick the consent box before saving a note.', 'error'); return; }
  try {
    await requestJson('/api/memory', { method: 'POST', json: { text, consent: true } });
    input.value = '';
    $('#memory-consent').checked = false;
    await renderMemory();
    toast('Note saved.', 'success');
  } catch (error) {
    toast(error.message, 'error');
  }
}

/* ------------------------------------------------------------------ init -- */

function autoGrow(textarea) {
  textarea.style.height = 'auto';
  textarea.style.height = `${Math.min(textarea.scrollHeight, 200)}px`;
}

function bindEvents() {
  $$('[data-view]').forEach((button) => button.addEventListener('click', () => showView(button.dataset.view)));
  $$('.quick-card').forEach((card) => card.addEventListener('click', () => runQuickAction(card.dataset.action)));
  $('#new-chat').addEventListener('click', createThread);
  $('#clear-history').addEventListener('click', () => {
    if (!state.threads.length) return;
    if (!window.confirm('Delete every conversation stored in this browser?')) return;
    state.threads = [];
    state.activeThreadId = null;
    saveStore();
    renderHistory();
    renderThread();
  });
  $('#menu-button').addEventListener('click', openNav);
  $('#sidebar-close').addEventListener('click', closeNav);
  $('#sidebar-scrim').addEventListener('click', closeNav);
  $('#theme-toggle').addEventListener('click', cycleTheme);
  $('#health-pill').addEventListener('click', () => { showView('settings'); runDiagnostics(); });

  const input = $('#composer-input');
  input.addEventListener('input', () => autoGrow(input));
  input.addEventListener('keydown', (event) => {
    if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $('#composer').requestSubmit();
    }
    if (event.key === 'Escape' && state.streaming) stopGeneration();
  });
  $('#composer').addEventListener('submit', (event) => {
    event.preventDefault();
    const value = input.value;
    if (!value.trim() && !state.attachment) return;
    sendMessage(value);
  });
  $('#stop-button').addEventListener('click', stopGeneration);
  $('#file-input').addEventListener('change', (event) => setAttachment(event.target.files?.[0]));
  $('#composer-image').addEventListener('click', () => showView('image'));

  const voice = $('#voice-button');
  const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!Recognition) {
    voice.disabled = true;
    voice.title = 'Dictation is not available in this browser.';
  } else {
    let recognition = null;
    voice.addEventListener('click', () => {
      if (recognition) { recognition.stop(); recognition = null; voice.setAttribute('aria-pressed', 'false'); return; }
      recognition = new Recognition();
      recognition.lang = document.documentElement.lang || 'en';
      recognition.interimResults = true;
      recognition.continuous = false;
      const start = input.value;
      voice.setAttribute('aria-pressed', 'true');
      recognition.onresult = (event) => {
        let transcript = '';
        for (let i = 0; i < event.results.length; i += 1) transcript += event.results[i][0].transcript;
        input.value = `${start}${start ? ' ' : ''}${transcript}`;
        autoGrow(input);
      };
      recognition.onerror = () => toast('Dictation failed. Check microphone permission.', 'error');
      recognition.onend = () => { recognition = null; voice.setAttribute('aria-pressed', 'false'); input.focus(); };
      recognition.start();
    });
  }

  $('#forecast-form').addEventListener('submit', runForecast);

  const agentToggle = $('#agent-toggle');
  const syncAgentToggle = () => {
    agentToggle.setAttribute('aria-pressed', String(Boolean(state.settings.agentMode)));
    $('#composer-hint').textContent = state.settings.agentMode
      ? 'Tool mode: math, forecast, sources and code are routed explicitly'
      : 'Enter to send · Shift+Enter for a new line';
  };
  agentToggle.addEventListener('click', () => {
    state.settings.agentMode = !state.settings.agentMode;
    saveStore();
    syncAgentToggle();
    toast(state.settings.agentMode
      ? 'Tool mode on: answers report which tools actually ran.'
      : 'Tool mode off: messages use the language model.', 'info', 3200);
  });
  syncAgentToggle();

  $('#image-form').addEventListener('submit', generateImage);
  $('#image-mode').addEventListener('change', (event) => {
    $('#image-mode-help').textContent = IMAGE_MODE_HELP[event.target.value] || '';
  });
  $('#upload-input').addEventListener('change', (event) => {
    $('#upload-name').textContent = event.target.files?.[0]?.name || 'No file selected';
  });
  $('#edit-strength').addEventListener('input', (event) => { $('#strength-value').textContent = event.target.value; });
  $('#analyze-image').addEventListener('click', () => analyzeUpload(false));
  $('#edit-image').addEventListener('click', () => analyzeUpload(true));

  $('#code-input').addEventListener('input', () => { $('#code-approval').checked = false; });
  $('#analyze-code').addEventListener('click', () => codeAction('analyze'));
  $('#explain-code').addEventListener('click', () => codeAction('explain'));
  $('#test-skeleton').addEventListener('click', () => codeAction('tests'));
  $('#run-code').addEventListener('click', () => codeAction('execute'));
  $('#preview-patch').addEventListener('click', () => codeAction('patch'));

  $('#math-form').addEventListener('submit', runMath);
  $('#search-form').addEventListener('submit', runSearch);
  $('#summarize-csv').addEventListener('click', async () => {
    try {
      const data = await requestJson('/api/research/csv', { method: 'POST', json: { csv: $('#csv-input').value } });
      $('#csv-output').textContent = JSON.stringify(data, null, 2);
    } catch (error) { $('#csv-output').textContent = `Failed: ${error.message}`; }
  });
  $('#plot-csv').addEventListener('click', async () => {
    try {
      const data = await requestJson('/api/research/plot', {
        method: 'POST', json: { csv: $('#csv-input').value, x: $('#csv-x').value, y: $('#csv-y').value },
      });
      const image = document.createElement('img');
      image.src = data.image_url;
      image.alt = `Plot of ${data.points} points`;
      $('#csv-plot').replaceChildren(image);
      $('#csv-output').textContent = `${data.points} points plotted. ${data.note}`;
    } catch (error) { $('#csv-output').textContent = `Failed: ${error.message}`; }
  });
  $('#refresh-experiments').addEventListener('click', refreshExperiments);

  $$('[data-theme-choice]').forEach((button) => button.addEventListener('click', () => {
    state.settings.theme = button.dataset.themeChoice;
    saveStore();
    applyTheme();
  }));
  $('#reduce-motion').addEventListener('change', (event) => {
    state.settings.reduceMotion = event.target.checked;
    saveStore();
    applyTheme();
  });
  $('#display-name').addEventListener('input', (event) => {
    state.settings.displayName = event.target.value;
    saveStore();
    renderGreeting();
  });
  $('#use-sources').addEventListener('change', (event) => { state.settings.useSources = event.target.checked; saveStore(); });
  $('#use-memory').addEventListener('change', (event) => { state.settings.useMemory = event.target.checked; saveStore(); });
  $('#clear-local').addEventListener('click', () => {
    if (!window.confirm('Delete conversations and settings stored in this browser?')) return;
    localStorage.removeItem(STORAGE.threads);
    localStorage.removeItem(STORAGE.settings);
    localStorage.removeItem(STORAGE.active);
    state.threads = [];
    state.activeThreadId = null;
    state.settings = { theme: 'system', reduceMotion: false, displayName: '', useSources: true, useMemory: false };
    saveStore();
    applyTheme();
    renderGreeting();
    renderHistory();
    renderThread();
    toast('Local data cleared.', 'success');
  });
  $('#run-diagnostics').addEventListener('click', runDiagnostics);
  $('#memory-add').addEventListener('click', addMemory);

  window.addEventListener('hashchange', () => {
    const view = location.hash.replace('#', '');
    if (VIEW_TITLES[view] && view !== state.view) showView(view, { silent: true });
  });
  document.addEventListener('keydown', (event) => {
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'k') { event.preventDefault(); createThread(); }
    if (event.key === 'Escape') closeNav();
  });
  window.addEventListener('beforeunload', (event) => {
    if (state.streaming) { event.preventDefault(); event.returnValue = ''; }
  });
}

function init() {
  loadStore();
  applyTheme();
  renderGreeting();
  renderHistory();
  renderThread();
  bindEvents();

  $('#display-name').value = state.settings.displayName || '';
  $('#use-sources').checked = Boolean(state.settings.useSources);
  $('#use-memory').checked = Boolean(state.settings.useMemory);
  $('#image-mode-help').textContent = IMAGE_MODE_HELP[$('#image-mode').value] || '';

  const initial = VIEW_TITLES[location.hash.replace('#', '')] ? location.hash.replace('#', '') : 'chat';
  showView(initial, { silent: true });
  refreshHealth(true);
  renderMemory().catch(() => {});
  setInterval(() => { if (!state.streaming) refreshHealth(false); }, 60000);
}

document.addEventListener('DOMContentLoaded', init);
