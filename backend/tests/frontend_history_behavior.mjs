import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import vm from 'node:vm';
import { fileURLToPath } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const source = fs.readFileSync(path.join(here, '../../frontend/app/history.html'), 'utf8');
const start = source.indexOf('    async function loadExtendedHistory');
const end = source.indexOf('    function renderEmpty', start);
assert.ok(start >= 0 && end > start, 'loadExtendedHistory extraction markers');

const events = [];
const context = {
  Promise,
  renderWritingHistory(body) { events.push(['writing', body]); },
  renderReadingHistory(body) { events.push(['reading', body]); },
  renderHistoryError(containerId, section, error) {
    events.push(['error', containerId, section, error.message]);
  },
};
vm.runInNewContext(
  `${source.slice(start, end)}\nglobalThis.loadExtendedHistoryForTest = loadExtendedHistory;`,
  context,
);

const calls = [];
const requester = async (route, token) => {
  calls.push([route, token]);
  if (route.startsWith('/api/writing/history')) {
    throw new Error('writing unavailable');
  }
  return { attempts: [{ attempt_id: 'reading-1' }] };
};

await context.loadExtendedHistoryForTest('session-token', requester);
assert.deepEqual(calls, [
  ['/api/writing/history?limit=10', 'session-token'],
  ['/reading/history?limit=20', 'session-token'],
]);
assert.deepEqual(events, [
  ['error', 'writing-history-list', 'Writing', 'writing unavailable'],
  ['reading', { attempts: [{ attempt_id: 'reading-1' }] }],
]);
console.log('history partial-failure contract: PASS');
