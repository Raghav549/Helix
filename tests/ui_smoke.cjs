/* Optional DOM-to-live-API smoke test. Run only against a trusted local server.
 * npm install --prefix out/jsdom --no-save jsdom@24.1.3
 * NODE_PATH=out/jsdom/node_modules HELIX_URL=http://127.0.0.1:8766/ node tests/ui_smoke.cjs
 * Requires Node >= 18; this optional test is separate from the Python suite.
 */
const fs = require('fs');
const assert = require('assert');
const {randomUUID} = require('node:crypto');
const {JSDOM} = require('jsdom');
const html = fs.readFileSync('helix/static/index.html', 'utf8');
const script = fs.readFileSync('helix/static/app.js', 'utf8');
const base = process.env.HELIX_URL || 'http://127.0.0.1:8766/';
const dom = new JSDOM(html, {url: base, runScripts: 'outside-only'});
const window = dom.window;
window.TextDecoder = TextDecoder;
window.scrollTo = () => {};
window.fetch = (path, options) => global.fetch(new URL(path, base), options);
window.eval(script);
const $ = (selector) => window.document.querySelector(selector);
const note = 'HELIX UI smoke ' + randomUUID();
const wait = async (predicate, label, limit = 10000) => {
  const end = Date.now() + limit;
  while (Date.now() < end) {
    if (predicate()) return;
    await new Promise(resolve => setTimeout(resolve, 50));
  }
  throw new Error('Timed out waiting for ' + label);
};
const submit = (selector) => $(selector).dispatchEvent(new window.Event('submit', {bubbles: true, cancelable: true}));
(async () => {
  try {
    await wait(() => $('#top-status').textContent.includes('4 of 4'), 'model status');
    assert($('.security-pill').textContent.includes('NO AUTH'));
    $('#memory-input').value = note; $('#memory-consent').checked = true; submit('#memory-form');
    await wait(() => $('#memory-list').textContent.includes(note), 'memory save');
    const row = [...$('#memory-list').querySelectorAll('.memory-item')].find(el => el.textContent.includes(note));
    row.querySelector('[aria-label="Delete memory"]').click();
    await wait(() => !$('#memory-list').textContent.includes(note), 'memory delete');
    $('#chat-input').value = 'Explain recursion'; submit('#chat-form');
    await wait(() => !$('#chat-send').disabled && !!$('.message.assistant .message-text'), 'chat stream');
    assert.equal(window.document.querySelectorAll('.message').length, 2);
    $('#clear-chat').click(); assert(!$('.message'));
    $('[data-view="image"]').click(); assert(!$('#view-image').hidden);
    submit('#image-form'); await wait(() => !!$('#image-output img'), 'renderer');
    $('[data-view="code"]').click();
    $('#code-input').value = 'def add(a, b):\n    return a+b\n';
    $('#test-skeleton').click(); await wait(() => $('#code-output').textContent.includes('test_add_is_callable'), 'test skeleton');
    $('#code-modified').value = 'def add(a, b):\n    return a-b\n';
    $('#preview-patch').click(); await wait(() => $('#code-output').textContent.includes('return a-b'), 'patch preview');
    $('#code-input').value = 'result = 42'; $('#run-code').click();
    assert($('#code-output').textContent.includes('approval box'));
    $('#code-approval').checked = true; $('#run-code').click();
    await wait(() => !$('#run-code').disabled && $('#code-output').textContent.includes('42'), 'approved code');
    assert(!$('#code-approval').checked);
    $('[data-view="research"]').click();
    submit('#math-form'); await wait(() => $('#math-output').textContent.includes('3*x**2 + 2'), 'symbolic math');
    $('#plot-csv').click(); await wait(() => !!$('#csv-plot img'), 'CSV plot');
    console.log('UI smoke passed: status, consent/delete, streaming chat, navigation, renderer, test/diff preview, approval, isolated code, math and plotting');
  } finally {
    // Only clean up the unique note this test created; never delete user data.
    try {
      const data = await (await global.fetch(new URL('/api/memory', base))).json();
      for (const entry of data.memories || []) if (entry.text === note)
        await global.fetch(new URL('/api/memory/' + entry.id, base), {method: 'DELETE'});
    } catch (_) {}
    dom.window.close();
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
