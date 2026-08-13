import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source = fs.readFileSync(
  new URL('../../frontend/app/anonymous-conversion.js', import.meta.url),
  'utf8',
);
const context = { globalThis: {} };
vm.createContext(context);
vm.runInContext(source, context);
const conversion = context.globalThis.BlabbyAnonymousConversion;

assert.equal(conversion.stateFor(0, 10).status, '可免費體驗 10 次口說，不需註冊');

for (const used of [1, 2, 3, 4]) {
  const state = conversion.stateFor(used, 10);
  assert.equal(state.card, 'none');
  assert.equal(state.remaining, 10 - used);
  assert.match(state.status, new RegExp(`剩餘 ${10 - used} 次`));
}

const milestone5 = conversion.stateFor(5, 10);
assert.equal(milestone5.card, 'milestone_5');
assert.match(milestone5.body, /保存你的練習紀錄/);

const milestone8 = conversion.stateFor(8, 10);
assert.equal(milestone8.card, 'milestone_8');
assert.match(milestone8.body, /保存歷史/);
assert.match(milestone8.body, /追蹤反覆出現的弱點/);

const completed = conversion.stateFor(10, 10);
assert.equal(completed.card, 'complete');
assert.equal(completed.lockRecorder, true);
assert.equal(completed.remaining, 0);
assert.equal(completed.status, '免費體驗完成');

const props = conversion.eventProps(milestone8, 'speaking_part1', false);
assert.deepEqual(
  Object.keys(props).sort(),
  ['anonymous_used_count', 'authenticated', 'remaining_count', 'source'].sort(),
);
assert.deepEqual(JSON.parse(JSON.stringify(props)), {
  anonymous_used_count: 8,
  remaining_count: 2,
  source: 'speaking_part1',
  authenticated: false,
});

const serialized = JSON.stringify(props).toLowerCase();
for (const forbidden of ['raw_ip', 'transcript', 'audio', 'email', 'visitor_id']) {
  assert.equal(serialized.includes(forbidden), false);
}

console.log('Anonymous conversion behavior: all assertions passed.');
