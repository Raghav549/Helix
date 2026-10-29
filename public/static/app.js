'use strict';
const $ = (selector) => document.querySelector(selector);
const all = (selector) => [...document.querySelectorAll(selector)];
let conversationHistory = [];
let activeChatController = null;

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const raw = await response.text();
  let data;
  try { data = raw ? JSON.parse(raw) : {}; }
  catch (_) {
    const excerpt = raw.replace(/<[^>]*>/g, ' ').replace(/\\s+/g, ' ').trim().slice(0, 180);
    throw new Error(`API route returned non-JSON (HTTP ${response.status}): ${excerpt || 'empty response'}`);
  }
  if (!response.ok) throw new Error(data.error || data.result || `Request failed (${response.status})`);
  return data;
}
function jsonOptions(body, method = 'POST') {
  return {method, headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body)};
}
function go(view) {
  all('.view').forEach((el) => { const active = el.id === `view-${view}`; el.hidden = !active; el.classList.toggle('active-view', active); });
  all('.nav-item').forEach((el) => { const active = el.dataset.view === view; el.classList.toggle('active', active); if (active) el.setAttribute('aria-current', 'page'); else el.removeAttribute('aria-current'); });
  const names = {chat: 'Chat', image: 'Image Generation', code: 'Coding Agent', research: 'Research Lab'};
  $('#breadcrumb').textContent = names[view] || view;
  if (view === 'research') refreshExperiments();
  window.scrollTo({top: 0, behavior: 'smooth'});
}
all('[data-view]').forEach((el) => el.addEventListener('click', () => go(el.dataset.view)));
all('[data-go]').forEach((el) => el.addEventListener('click', () => go(el.dataset.go)));
all('.suggestion').forEach((el) => el.addEventListener('click', () => { $('#chat-input').value = el.dataset.prompt; $('#chat-input').focus(); }));

async function refreshStatus() {
  try {
    const status = await api('/api/status');
    const ready = Object.values(status.models).filter(Boolean).length;
    $('#top-status').textContent = `${ready} of ${Object.keys(status.models).length} models available`;
    $('#top-dot').classList.toggle('off', ready !== Object.keys(status.models).length);
  } catch (_) { $('#top-status').textContent = 'Server unavailable'; $('#top-dot').classList.add('off'); }
}
function addMessage(role, text) {
  const welcome = $('#chat-welcome'); if (welcome) welcome.remove();
  const message = document.createElement('div'); message.className = `message ${role}`;
  const icon = document.createElement('div'); icon.className = 'message-icon'; icon.textContent = role === 'user' ? 'Y' : '✳';
  const body = document.createElement('div'); body.className = 'message-body';
  const label = document.createElement('div'); label.className = 'message-role'; label.textContent = role === 'user' ? 'YOU' : 'HELIX · UNVERIFIED';
  const content = document.createElement('div'); content.className = 'message-text'; content.textContent = text;
  body.append(label, content); message.append(icon, body); $('#chat-messages').append(message);
  $('#chat-messages').scrollTop = $('#chat-messages').scrollHeight;
  return content;
}
$('#chat-input').addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && !event.shiftKey && !event.isComposing) { event.preventDefault(); $('#chat-form').requestSubmit(); }
});
$('#clear-chat').addEventListener('click', () => {
  if (activeChatController) activeChatController.abort();
  activeChatController = null; $('#chat-send').disabled = false;
  conversationHistory = [];
  $('#chat-messages').replaceChildren();
  const note = document.createElement('div'); note.className = 'chat-welcome'; note.id = 'chat-welcome';
  const title = document.createElement('h3'); title.textContent = 'A fresh page.';
  const detail = document.createElement('p'); detail.textContent = 'Conversation cleared. Saved memory notes are unchanged.';
  note.append(title, detail); $('#chat-messages').append(note);
});
$('#chat-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  const input = $('#chat-input'); const button = $('#chat-send');
  const text = input.value.trim(); if (!text || button.disabled) return;
  const controller = new AbortController(); activeChatController = controller;
  const prior = conversationHistory.slice(-12); input.value = ''; addMessage('user', text);
  const output = addMessage('assistant', ''); button.disabled = true;
  let reply = '';
  try {
    const response = await fetch('/api/chat/stream', {...jsonOptions({message: text, history: prior, use_memory: $('#use-memory').checked}), signal: controller.signal});
    if (!response.ok) {
      const raw = await response.text();
      let err = {};
      try { err = JSON.parse(raw); } catch (_) {
        const excerpt = raw.replace(/<[^>]*>/g, ' ').replace(/\\s+/g, ' ').trim().slice(0, 180);
        throw new Error(`API route returned non-JSON (HTTP ${response.status}): ${excerpt || 'empty response'}`);
      }
      throw new Error(err.error || 'Chat unavailable');
    }
    const reader = response.body.getReader(); const decoder = new TextDecoder(); let buffer = '';
    while (true) {
      const part = await reader.read(); buffer += decoder.decode(part.value || new Uint8Array(), {stream: !part.done});
      let boundary;
      while ((boundary = buffer.indexOf('\n\n')) !== -1) {
        const eventText = buffer.slice(0, boundary); buffer = buffer.slice(boundary + 2);
        for (const line of eventText.split('\n')) if (line.startsWith('data: ')) {
          const data = JSON.parse(line.slice(6)); if (data.error) throw new Error(data.error);
          if (data.delta) { reply += data.delta; output.textContent = reply; $('#chat-messages').scrollTop = $('#chat-messages').scrollHeight; }
        }
      }
      if (part.done) break;
    }
    if (!reply) output.textContent = '(The model produced no text.)';
    if (activeChatController === controller) {
      conversationHistory.push({role: 'user', text}, {role: 'assistant', text: reply});
      conversationHistory = conversationHistory.slice(-12);
    }
  } catch (error) {
    output.textContent = error.name === 'AbortError' ? 'Generation stopped.' : `Unable to generate: ${error.message}`;
  } finally {
    if (activeChatController === controller) { activeChatController = null; button.disabled = false; input.focus(); }
  }
});

async function refreshMemory() {
  const list = $('#memory-list'); list.replaceChildren();
  try {
    const data = await api('/api/memory');
    for (const memory of data.memories) {
      const row = document.createElement('div'); row.className = 'memory-item';
      const text = document.createElement('span'); text.textContent = memory.text;
      const edit = document.createElement('button'); edit.type = 'button'; edit.textContent = 'Edit'; edit.setAttribute('aria-label', 'Edit memory');
      edit.addEventListener('click', async () => {
        const next = window.prompt('Edit this locally saved note:', memory.text); if (next === null) return;
        try { await api(`/api/memory/${memory.id}`, jsonOptions({text: next, consent: true}, 'PUT')); refreshMemory(); }
        catch (error) { $('#memory-error').textContent = error.message; }
      });
      const remove = document.createElement('button'); remove.type = 'button'; remove.textContent = '×'; remove.setAttribute('aria-label', 'Delete memory');
      remove.addEventListener('click', async () => {
        try { await api(`/api/memory/${memory.id}`, {method: 'DELETE'}); refreshMemory(); }
        catch (error) { $('#memory-error').textContent = error.message; }
      });
      row.append(text, edit, remove); list.append(row);
    }
  } catch (error) { $('#memory-error').textContent = error.message; }
}
$('#memory-form').addEventListener('submit', async (event) => {
  event.preventDefault(); $('#memory-error').textContent = '';
  try {
    await api('/api/memory', jsonOptions({text: $('#memory-input').value, consent: $('#memory-consent').checked}));
    $('#memory-input').value = ''; $('#memory-consent').checked = false; await refreshMemory();
  } catch (error) { $('#memory-error').textContent = error.message; }
});

const modeNotes = {
  physics: 'Try a color, sphere or cube, position, material, and light direction.',
  physics_guided: 'Uses a physics-rendered reference to guide the 32×32 model; the reference provides extra information.',
  learned: 'A tiny from-scratch diffusion model. Prompt adherence and visual quality are very limited.'
};
$('#image-mode').addEventListener('change', () => { $('#image-mode-note').textContent = modeNotes[$('#image-mode').value]; });
function displayImage(url, caption) {
  const image = document.createElement('img'); image.src = url; image.alt = 'Image generated locally by HELIX';
  $('#image-output').replaceChildren(image); $('#image-caption').textContent = caption;
  const link = $('#image-download'); link.href = url; link.hidden = false;
}
$('#image-form').addEventListener('submit', async (event) => {
  event.preventDefault(); $('#image-error').textContent = ''; const button = $('#image-submit'); button.disabled = true;
  $('#image-caption').textContent = 'GENERATING LOCALLY…';
  try {
    const data = await api('/api/image/generate', jsonOptions({prompt: $('#image-prompt').value, mode: $('#image-mode').value, seed: Number($('#image-seed').value)}));
    displayImage(data.image_url, data.notice);
    if (data.metadata && data.metadata.scene) $('#image-caption').textContent += ` Scene: ${data.metadata.scene.color} ${data.metadata.scene.shape}, ${data.metadata.scene.position}.`;
  } catch (error) { $('#image-error').textContent = error.message; $('#image-caption').textContent = 'GENERATION FAILED'; }
  finally { button.disabled = false; }
});
let imageObjectUrl = null;
$('#upload-image').addEventListener('change', () => {
  const file = $('#upload-image').files[0]; $('#upload-filename').textContent = file ? file.name : 'No file selected';
  if (imageObjectUrl) URL.revokeObjectURL(imageObjectUrl);
  if (file) { imageObjectUrl = URL.createObjectURL(file); const image = document.createElement('img'); image.src = imageObjectUrl; image.alt = 'Selected image preview'; $('#upload-preview').replaceChildren(image); }
});
$('#edit-strength').addEventListener('input', () => { $('#strength-value').textContent = $('#edit-strength').value; });
async function uploadImage(edit) {
  const file = $('#upload-image').files[0]; $('#upload-error').textContent = '';
  if (!file) { $('#upload-error').textContent = 'Choose an image first.'; return; }
  const button = $(edit ? '#edit-image' : '#analyze-image'); button.disabled = true;
  const body = new FormData(); body.append('image', file);
  if (edit) { body.append('prompt', $('#image-prompt').value); body.append('seed', $('#image-seed').value); body.append('strength', $('#edit-strength').value); }
  try {
    const data = await api(edit ? '/api/image/edit' : '/api/image/analyze', {method: 'POST', body});
    if (edit) { displayImage(data.image_url, data.notice); $('#analysis-output').textContent = 'Image-to-image edit generated. See canvas above.'; }
    else $('#analysis-output').textContent = JSON.stringify(data, null, 2);
  } catch (error) { $('#upload-error').textContent = error.message; }
  finally { button.disabled = false; }
}
$('#analyze-image').addEventListener('click', () => uploadImage(false));
$('#edit-image').addEventListener('click', () => uploadImage(true));

$('#code-input').addEventListener('input', () => { $('#code-approval').checked = false; });
$('#analyze-code').addEventListener('click', async () => {
  $('#code-result-note').textContent = 'STATIC AST ANALYSIS ONLY; NO CODE WAS EXECUTED.';
  const button = $('#analyze-code'); button.disabled = true; $('#code-badge').textContent = 'ANALYZING';
  try {
    const data = await api('/api/code/analyze', jsonOptions({code: $('#code-input').value}));
    $('#code-output').textContent = JSON.stringify(data, null, 2); $('#code-badge').textContent = data.syntax_error ? 'SYNTAX ERROR' : 'ANALYZED';
  } catch (error) { $('#code-output').textContent = error.message; $('#code-badge').textContent = 'ERROR'; }
  finally { button.disabled = false; }
});
$('#run-code').addEventListener('click', async () => {
  const button = $('#run-code'); const approved = $('#code-approval').checked;
  $('#code-approval').checked = false;
  if (!approved) { $('#code-output').textContent = 'Check the approval box before each execution.'; $('#code-badge').textContent = 'NOT RUN'; return; }
  button.disabled = true; $('#code-badge').textContent = 'RUNNING';
  $('#code-result-note').textContent = 'EXECUTION IS ISOLATED ON SUPPORTED LINUX HOSTS; IT FAILS CLOSED OTHERWISE.';
  try {
    const data = await api('/api/code/execute', jsonOptions({code: $('#code-input').value, approved}));
    $('#code-output').textContent = `${data.output || '(no output)'}${data.error ? '\n\n' + data.error : ''}\n\n[${data.isolation}; ${data.duration_s.toFixed(2)}s]`;
    $('#code-badge').textContent = data.ok ? 'COMPLETE' : 'FAILED';
  } catch (error) { $('#code-output').textContent = error.message; $('#code-badge').textContent = 'NOT RUN'; }
  finally { button.disabled = false; }
});
$('#preview-patch').addEventListener('click', async () => {
  const button = $('#preview-patch'); button.disabled = true; $('#code-badge').textContent = 'PREVIEW';
  $('#code-result-note').textContent = 'DIFF PREVIEW ONLY; NO REPOSITORY FILE WAS CHANGED.';
  try { const data = await api('/api/code/patch', jsonOptions({original: $('#code-input').value, modified: $('#code-modified').value})); $('#code-output').textContent = data.diff; }
  catch (error) { $('#code-output').textContent = error.message; $('#code-badge').textContent = 'ERROR'; }
  finally { button.disabled = false; }
});
$('#test-skeleton').addEventListener('click', async () => {
  const button = $('#test-skeleton'); button.disabled = true; $('#code-badge').textContent = 'SKELETON';
  $('#code-result-note').textContent = 'GENERATED IMPORT/CALLABLE CHECKS ONLY; NOT EXECUTED OR BEHAVIORALLY VERIFIED.';
  try { const data = await api('/api/code/test-skeleton', jsonOptions({code: $('#code-input').value})); $('#code-output').textContent = data.tests; }
  catch (error) { $('#code-output').textContent = error.message; $('#code-badge').textContent = 'ERROR'; }
  finally { button.disabled = false; }
});
$('#explain-code').addEventListener('click', async () => {
  $('#code-result-note').textContent = 'UNVERIFIED MODEL OUTPUT; NO CODE WAS EXECUTED.';
  const button = $('#explain-code'); button.disabled = true; $('#code-badge').textContent = 'GENERATING';
  try {
    const data = await api('/api/code/explain', jsonOptions({code: $('#code-input').value}));
    $('#code-output').textContent = `${data.reply || '(no completion)'}\n\n[Unverified experimental model output.]`;
    $('#code-badge').textContent = 'UNVERIFIED';
  } catch (error) { $('#code-output').textContent = error.message; $('#code-badge').textContent = 'ERROR'; }
  finally { button.disabled = false; }
});

$('#math-form').addEventListener('submit', async (event) => {
  event.preventDefault(); const button = $('#math-submit'); button.disabled = true; $('#math-badge').textContent = 'COMPUTING';
  try {
    const data = await api('/api/research/math', jsonOptions({operation: $('#math-operation').value, expression: $('#math-expression').value,
      variable: $('#math-variable').value, point: $('#math-point').value, save_experiment: $('#save-experiment').checked}));
    $('#math-output').textContent = `${data.result}\n\n${data.steps.join('\n')}${data.experiment_id ? '\n\nExperiment: ' + data.experiment_id : ''}`;
    $('#math-badge').textContent = 'SYMPY'; if (data.experiment_id) refreshExperiments();
  } catch (error) { $('#math-output').textContent = error.message; $('#math-badge').textContent = 'ERROR'; }
  finally { button.disabled = false; }
});
$('#summarize-csv').addEventListener('click', async () => {
  const button = $('#summarize-csv'); button.disabled = true;
  try { const data = await api('/api/research/csv', jsonOptions({csv: $('#csv-input').value})); $('#csv-output').textContent = JSON.stringify(data, null, 2); }
  catch (error) { $('#csv-output').textContent = error.message; }
  finally { button.disabled = false; }
});
$('#plot-csv').addEventListener('click', async () => {
  const button = $('#plot-csv'); button.disabled = true;
  try {
    const data = await api('/api/research/plot', jsonOptions({csv: $('#csv-input').value, x: $('#csv-x').value, y: $('#csv-y').value}));
    const image = document.createElement('img'); image.src = data.image_url; image.alt = `Line plot with ${data.points} paired numeric points`;
    $('#csv-plot').replaceChildren(image); $('#csv-output').textContent = `${data.points} paired points plotted. ${data.note}`;
  } catch (error) { $('#csv-output').textContent = error.message; }
  finally { button.disabled = false; }
});
async function refreshExperiments() {
  const list = $('#experiment-list'); list.replaceChildren();
  try {
    const data = await api('/api/research/experiments');
    if (!data.experiments.length) { const msg = document.createElement('p'); msg.className = 'field-note'; msg.textContent = 'No recorded experiments yet.'; list.append(msg); return; }
    for (const exp of data.experiments.slice(0, 12)) {
      const row = document.createElement('div'); row.className = 'experiment-item';
      const text = document.createElement('div'); const title = document.createElement('strong'); title.textContent = exp.name;
      const detail = document.createElement('small'); detail.textContent = `${exp.stage} · ${exp.device}`; text.append(title, detail);
      const metric = document.createElement('div'); metric.className = 'exp-metric'; metric.textContent = typeof exp.best_loss === 'number' ? `loss ${exp.best_loss.toFixed(3)}` : 'RECORDED';
      row.append(text, metric); list.append(row);
    }
  } catch (error) { const msg = document.createElement('p'); msg.textContent = error.message; list.append(msg); }
}
$('#refresh-experiments').addEventListener('click', refreshExperiments);
refreshStatus(); refreshMemory();
