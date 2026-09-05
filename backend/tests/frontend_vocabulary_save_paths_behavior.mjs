import assert from 'node:assert/strict';
import fs from 'node:fs';
import test from 'node:test';
import vm from 'node:vm';

const readPage = name => fs.readFileSync(new URL(`../../frontend/app/${name}.html`, import.meta.url), 'utf8');
const reading = readPage('reading');
const speaking = readPage('index');
const quota = { detail: { error: 'vocab_limit_reached', limit: 30 } };
const response = (status, body) => ({ status, ok: status >= 200 && status < 300, json: async () => body });

function extract(source, start, end) {
  const left = source.indexOf(start);
  const right = source.indexOf(end, left);
  assert.ok(left >= 0 && right > left, `Missing execution markers: ${start}`);
  return source.slice(left, right);
}

class Element {
  constructor() {
    this.children = [];
    this.style = {};
    this.listeners = {};
    this.classes = new Set();
    this.classList = { add: value => this.classes.add(value) };
    this.textContent = '';
    this.innerHTML = '';
    this.disabled = false;
    this.hidden = true;
  }
  addEventListener(name, handler) { this.listeners[name] = handler; }
  appendChild(child) { child.parentNode = this; this.children.push(child); }
}

function readingPage(result) {
  const elements = new Map();
  const $ = id => {
    if (!elements.has(id)) elements.set(id, new Element());
    return elements.get(id);
  };
  const events = [];
  const requests = [];
  const anchor = new Element();
  anchor.dataset = { word: 'syntheticword' };
  const context = {
    $, console, CSS: { escape: value => value },
    popoverAnchor: anchor,
    sentenceContaining: () => 'A synthetic sentence.',
    state: { savedWords: new Set() },
    defnCache: new Map([['syntheticword', { definition: 'Synthetic definition.' }]]),
    window: { analytics: { track: (...args) => events.push(args) } },
    document: { querySelectorAll: () => [anchor] },
    authFetch: async (method, path, payload) => {
      requests.push({ method, path, payload });
      return path === '/api/vocabulary/save_word'
        ? result : response(200, { zh_meaning: 'meaning', definition: 'definition' });
    },
  };
  vm.createContext(context);
  vm.runInContext(extract(reading, '        const PRO_MODAL_COPY', "        $('pro-modal-dismiss')"), context);
  vm.runInContext(extract(reading, "        $('vocab-popover-save').addEventListener", '        //  Questions rendering'), context);
  return { context, $, events, requests, anchor, click: () => $('vocab-popover-save').listeners.click() };
}

async function speakingPage(result) {
  const block = new Element();
  const list = new Element();
  const events = [];
  const requests = [];
  const context = {
    API_BASE: 'https://api.example.test',
    document: {
      getElementById: id => id === 'vocab-suggestions' ? block : list,
      createElement: () => new Element(),
    },
    sb: { auth: { getSession: async () => ({ data: { session: { access_token: 'verified-token' } } }) } },
    window: { analytics: { track: (...args) => events.push(args) } },
    fetch: async (url, options) => { requests.push({ url, options }); return result; },
  };
  vm.createContext(context);
  vm.runInContext(extract(speaking, '        async function renderVocabSuggestions', '        // Target copy bank'), context);
  await context.renderVocabSuggestions({ suggested_vocab: { id: 'item', word: 'syntheticword' } });
  const row = list.children[0];
  const button = row.children.find(child => child.className === 'vocab-suggestion-add');
  return { row, button, events, requests, click: () => button.listeners.click() };
}

test('Reading quota 403 opens vocabulary paywall without recording a save', async () => {
  const page = readingPage(response(403, quota));
  await page.click();
  assert.equal(page.$('pro-modal-overlay').hidden, false);
  assert.equal(page.$('pro-modal-cta').href, '/upgrade.html?source=vocab_limit');
  assert.match(page.$('pro-modal-sub').textContent, /30/);
  assert.equal(page.context.state.savedWords.size, 0);
  assert.equal(page.anchor.classes.has('is-saved'), false);
  assert.equal(page.events.length, 0);
  assert.equal(page.$('vocab-popover-save').disabled, false);
});

test('Speaking parses nested quota error and shows existing upgrade nudge', async () => {
  const page = await speakingPage(response(403, quota));
  await page.click();
  assert.equal(page.button.textContent, '已達上限');
  assert.equal(page.button.classes.has('added'), false);
  assert.equal(page.button.disabled, true);
  assert.ok(page.row.children.some(child => child.innerHTML.includes('/upgrade.html?source=vocab_limit')));
  assert.equal(page.events.length, 0);
  assert.equal(page.requests[0].options.headers.Authorization, 'Bearer verified-token');
});

for (const status of [401, 403, 503]) {
  test(`Non-quota ${status} never opens a quota paywall or marks a save`, async () => {
    const payload = { detail: 'Not a quota error' };
    const reader = readingPage(response(status, payload));
    await reader.click();
    assert.equal(reader.$('pro-modal-overlay').hidden, true);
    assert.equal(reader.context.state.savedWords.size, 0);
    assert.equal(reader.$('vocab-popover-save').disabled, false);
    const speaker = await speakingPage(response(status, payload));
    await speaker.click();
    assert.equal(speaker.row.children.length, 2);
    assert.equal(speaker.button.classes.has('added'), false);
    assert.equal(speaker.events.length, 0);
  });
}

for (const status of ['added', 'exists']) {
  test(`Reading ${status} response retains saved state and telemetry`, async () => {
    const page = readingPage(response(200, { status }));
    await page.click();
    assert.equal(page.context.state.savedWords.has('syntheticword'), true);
    assert.equal(page.anchor.classes.has('is-saved'), true);
    assert.equal(page.events[0][0], 'reading_word_saved');
    assert.equal(page.$('pro-modal-overlay').hidden, true);
  });
}

test('Speaking success retains existing saved state and telemetry', async () => {
  const page = await speakingPage(response(200, { id: 'saved' }));
  await page.click();
  assert.equal(page.button.classes.has('added'), true);
  assert.equal(page.button.disabled, true);
  assert.equal(page.events[0][0], 'vocabulary_item_added');
});
