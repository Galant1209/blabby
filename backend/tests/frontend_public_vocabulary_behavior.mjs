import assert from 'node:assert/strict';
import fs from 'node:fs';
import { JSDOM } from 'jsdom';
const html = fs.readFileSync(new URL('../../frontend/app/vocabulary.html', import.meta.url), 'utf8');
const code = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)].map(m => m[1]).find(s => s.includes('const cfg = window.BLABBY_CONFIG'));
assert.ok(code);
const dom = new JSDOM(html, { runScripts: 'outside-only', url: 'https://local.invalid/vocabulary.html' });
const w = dom.window, d = w.document;
w.BLABBY_CONFIG = { supabaseUrl: 'https://local.invalid', supabaseAnonKey: 'dummy' };
w.BLABBY_API_BASE = 'https://api.invalid';
let session = null;
w.supabase = { createClient: () => ({ auth: { getSession: async () => ({ data: { session } }) }, from: () => { throw Error('raw table query'); } }) };
w.analytics = { track() {}, identify() {} };
w.alert = () => { throw Error('unexpected alert'); };
const calls = [], pending = [];
w.fetch = (url, options) => {
    calls.push({ url: new URL(url), options });
    return new Promise((resolve, reject) => pending.push({ resolve, reject }));
};
const flush = async () => { for (let i = 0; i < 8; i++) await new Promise(r => setImmediate(r)); };
const reply = (request, items, next = null) => request.resolve({ ok: true, json: async () => ({ items, has_more: next !== null, next_offset: next }) });
const item = (word = 'hello') => ({ id: 'one', word, zh_meaning: '你好', topic: 'study', ielts_band_level: '6.0' });
w.eval(code);
assert.match(d.querySelector('#vocab-grid').textContent, /載入/);
await flush();
assert.equal(calls.length, 2); // Anonymous bank + bounded recommendations only.
assert.ok(calls.every(c => !c.options?.headers?.Authorization));
assert.ok(calls.every(c => c.url.searchParams.get('limit') === '50'));
reply(pending.shift(), [item()], 50);
reply(pending.shift(), [item('recommended')]);
await flush();
assert.match(d.querySelector('#vocab-grid').textContent, /hello/);
assert.match(d.querySelector('#recommended-grid').textContent, /recommended/);
d.querySelector('#bank-next').click();
assert.equal(calls.at(-1).url.searchParams.get('offset'), '50');
reply(pending.shift(), [item('second page')]); await flush();
assert.equal(d.querySelector('#bank-next').disabled, true);
d.querySelector('#bank-prev').click();
assert.equal(calls.at(-1).url.searchParams.get('offset'), '0');
const stale = pending.shift();
// Search invalidates in-flight pages before debounce, and never queries each key.
const input = d.querySelector('#vocab-search');
const count = calls.length;
for (const value of ['學', '學習']) { input.value = value; input.dispatchEvent(new w.Event('input')); }
assert.equal(calls.length, count);
reply(stale, [item('STALE')]); await flush();
assert.doesNotMatch(d.querySelector('#vocab-grid').textContent, /STALE/);
await new Promise(r => setTimeout(r, 650));
assert.equal(calls.length, count + 1);
assert.equal(calls.at(-1).url.searchParams.get('q'), '學習');
reply(pending.shift(), [item('whole corpus match')]); await flush();
assert.match(d.querySelector('#vocab-grid').textContent, /whole corpus match/);
d.querySelector('#topic-filters button[data-topic="study"]').click();
assert.equal(calls.at(-1).url.searchParams.get('topic'), 'study');
reply(pending.shift(), []); await flush();
assert.match(d.querySelector('#vocab-grid').textContent, /No words/);
d.querySelector('[data-level="6.0"]').click();
assert.equal(calls.at(-1).url.searchParams.get('band'), '6.0');
assert.equal(calls.at(-1).url.searchParams.get('offset'), '0');
pending.shift().reject(new TypeError('offline')); await flush();
assert.match(d.querySelector('#vocab-grid').textContent, /檢查網路/);
d.querySelector('#bank-retry').click();
reply(pending.shift(), [item()]); await flush();
// A later authenticated save still uses the existing Free30 quota paywall.
session = { access_token: 'test-token' };
d.querySelector('#vocab-grid .vocab-card__add').click(); await flush();
assert.equal(calls.at(-1).url.pathname, '/api/vocabulary/my');
pending.shift().resolve({ ok: false, status: 403, json: async () => ({ detail: { error: 'vocab_limit_reached', limit: 30 } }) });
await flush();
assert.equal(d.querySelector('#vocab-pro-modal-overlay').hidden, false);
assert.equal(d.querySelector('#vocab-pro-modal-cta').getAttribute('href'), '/upgrade.html?source=vocab_limit');
assert.equal(d.querySelector('#vocab-grid .vocab-card__add').disabled, false);
assert.ok(!/\.from\s*\(\s*['"]vocabulary_items/.test(html));
// Unpublished generated cards are consumed directly from the owner's response;
// saving must not depend on finding the item again in the public bank.
d.querySelector('#generate-vocab-btn').click(); await flush();
assert.equal(calls.at(-1).url.pathname, '/api/vocabulary/generate');
assert.equal(calls.at(-1).options.headers.Authorization, 'Bearer test-token');
pending.shift().resolve({ ok: true, json: async () => ({
    items: [{ ...item('unpublished-generated'), id: 'generated-id', is_public: false }],
    generated_count: 1, topic: 'study', weakness_tag: 'weak_vocab',
}) });
await flush();
assert.match(d.querySelector('#recommended-grid').textContent, /unpublished-generated/);
assert.doesNotMatch(d.querySelector('#vocab-grid').textContent, /unpublished-generated/);
const generatedButton = d.querySelector('#recommended-grid .vocab-card__add');
assert.equal(generatedButton.disabled, false);
generatedButton.click(); await flush();
assert.equal(calls.at(-1).url.pathname, '/api/vocabulary/my');
assert.equal(JSON.parse(calls.at(-1).options.body).vocabulary_item_id, 'generated-id');
pending.shift().resolve({ ok: false, status: 403, json: async () => ({ detail: { error: 'vocab_limit_reached', limit: 30 } }) });
await flush();
assert.equal(generatedButton.disabled, false);
w.close();
console.log('PASS public vocabulary DOM: anonymous loading, bounded recommendations, paging, server search, debounce, stale response, topic/band, empty/error/retry, quota CTA, unpublished generation direct render/save');
