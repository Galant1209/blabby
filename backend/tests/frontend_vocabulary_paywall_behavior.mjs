import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const source = fs.readFileSync(path.join(here, '../../frontend/app/vocabulary.html'), 'utf8');
const start = source.indexOf('    class ApiResponseError');
const end = source.indexOf('    // Identify on auth ready', start);
assert.ok(start >= 0 && end > start, 'API helper extraction markers');

const requests = [];
const context = {
  API_BASE: 'https://api.example.test',
  getToken: async () => 'session-token',
  Promise,
  Error,
  JSON,
  fetch: async (url, options) => {
    requests.push({ url, options });
    return context.nextResponse;
  },
};
vm.runInNewContext(
  `${source.slice(start, end)}\nglobalThis.apiJsonForTest = authJson;\nglobalThis.quotaForTest = isVocabularyQuotaError;\nglobalThis.addErrorMessageForTest = vocabularyAddErrorMessage;`,
  context,
);

const response = (status, payload) => ({
  status,
  ok: status >= 200 && status < 300,
  json: async () => payload,
});

context.nextResponse = response(403, {
  detail: {
    error: 'vocab_limit_reached',
    limit: 30,
    message: 'Free users may save up to 30 words.',
  },
});
await assert.rejects(
  context.apiJsonForTest('POST', '/api/vocabulary/my', { vocabulary_item_id: 'new-item' }),
  (error) => {
    assert.equal(error.status, 403);
    assert.equal(error.code, 'vocab_limit_reached');
    assert.equal(error.detail.limit, 30);
    assert.equal(context.quotaForTest(error), true);
    return true;
  },
);

context.nextResponse = response(401, {});
await assert.rejects(
  context.apiJsonForTest('POST', '/api/vocabulary/my', { vocabulary_item_id: 'new-item' }),
  (error) => {
    assert.equal(error.status, 401);
    assert.equal(error.code, 'auth_failed');
    assert.equal(context.quotaForTest(error), false);
    return true;
  },
);

context.nextResponse = response(403, { detail: 'Forbidden' });
await assert.rejects(
  context.apiJsonForTest('POST', '/api/vocabulary/my', { vocabulary_item_id: 'new-item' }),
  (error) => {
    assert.equal(error.status, 403);
    assert.equal(error.code, null);
    assert.equal(context.quotaForTest(error), false);
    return true;
  },
);

context.nextResponse = response(200, { id: 'saved-item' });
const saved = await context.apiJsonForTest(
  'POST',
  '/api/vocabulary/my',
  { vocabulary_item_id: 'saved-item' },
);
assert.deepEqual(saved, { id: 'saved-item' });
assert.equal(requests[0].options.headers.Authorization, 'Bearer session-token');
assert.equal(requests[0].options.method, 'POST');
assert.equal(requests.length, 4);
assert.equal(context.addErrorMessageForTest({ message: 'not_signed_in' }), '請先登入');
assert.equal(context.addErrorMessageForTest({ status: 401 }), '登入已過期，請重新登入。');
assert.equal(context.addErrorMessageForTest({ status: 403 }), '目前無法加入這個單字，請稍後再試。');
assert.equal(context.addErrorMessageForTest({ status: 404 }), '這個單字已不存在，請重新整理後再試。');
assert.equal(context.addErrorMessageForTest({ status: 503 }), '單字服務暫時無法使用，請稍後再試。');
assert.equal(context.addErrorMessageForTest({ name: 'TypeError' }), '無法連線，請檢查網路後再試。');
console.log('vocabulary paywall API classification: PASS');
