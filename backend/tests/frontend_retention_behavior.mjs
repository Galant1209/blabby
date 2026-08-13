import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const source = fs.readFileSync(
  new URL('../../frontend/app/retention.js', import.meta.url),
  'utf8',
);
const context = { globalThis: {} };
vm.createContext(context);
vm.runInContext(source, context);
const retention = context.globalThis.BlabbyRetention;
const index = fs.readFileSync(new URL('../../frontend/app/index.html', import.meta.url), 'utf8');

const expected = {
  weak_vocab: /替換一個.*簡單詞/,
  safe_answer: /明確立場/,
  lack_detail: /具體原因或例子/,
  grammar_minor: /文法問題是否重複出現/,
  off_topic: /直接回答問題/,
};

for (const [tag, nextAction] of Object.entries(expected)) {
  const model = retention.resumeModel({ id: `record-${tag}`, weakness_tag: tag, resolved: false });
  assert.equal(model.hasFocus, true);
  assert.equal(model.focus.tag, tag);
  assert.equal(model.focus.status, '正在處理');
  assert.equal(model.focus.nextAction.questionCount, 2);
  assert.match(model.focus.nextAction.label, nextAction);
  assert.equal(/\d+\s*\/\s*\d+|%/.test(model.focus.evidence), false);
}

const noData = retention.resumeModel(null);
assert.equal(noData.hasFocus, false);
assert.equal(noData.cta, '開始口說');
assert.match(noData.body, /下一次直接從那裡繼續/);
assert.equal(retention.resumeModel({ id: 'resolved', weakness_tag: 'weak_vocab', resolved: true }).hasFocus, false);

const closure = retention.closureModel({
  record_id: 'saved-record',
  weakness_tag: 'lack_detail',
  persisted: true,
}, 3);
assert.equal(closure.answerCount, 3);
assert.equal(closure.today, '今天完成 3 題 Speaking Part 1。');
assert.equal(closure.focus.tag, 'lack_detail');
assert.match(closure.focus.nextAction.label, /具體原因或例子/);

const unpersisted = retention.closureModel({
  record_id: 'not-saved',
  weakness_tag: 'lack_detail',
  persisted: false,
}, 3);
assert.equal(unpersisted.focus, null);
assert.match(unpersisted.neutral, /沒有足夠的已保存資料/);

const props = JSON.parse(JSON.stringify(retention.eventProps('lack_detail', 'resume_card', 3)));
assert.deepEqual(Object.keys(props).sort(), [
  'authenticated', 'session_answer_count', 'source', 'weakness_category',
].sort());
assert.equal(props.authenticated, true);
const serialized = JSON.stringify(props).toLowerCase();
for (const forbidden of ['transcript', 'email', 'audio', 'raw_ip', 'user_content']) {
  assert.equal(serialized.includes(forbidden), false);
}

for (const event of [
  'session_closure_viewed',
  'session_closure_continue_clicked',
  'current_focus_resolved',
  'authenticated_process_success',
]) {
  assert.equal(index.includes(event), true);
}
assert.match(index, /body\.anonymous-review #prescription-card/);
assert.match(index, /body\.anonymous-review #session-closure/);
assert.equal(index.includes('PENDING_RESUME_MAX_AGE_MS'), false);
assert.equal(index.includes('/api/retention/resume'), false);
assert.equal(index.includes('id="retention-resume-card"'), false);
assert.ok(index.indexOf('id="prescription-card"') < index.indexOf('id="practice-hub"'));
assert.match(index, /@media \(max-width: 560px\)[\s\S]*?\.retention-actions > \* \{ width: 100%; \}/);
assert.match(index, /id="end-session-btn"[\s\S]*?hidden>結束這輪練習/);
assert.match(index, /id="session-closure-evidence"[\s\S]*?hidden>這次回答已加入你的進度證據。/);
assert.match(index, /id="session-closure-active-vocab"[\s\S]*?hidden/);
assert.match(index, /currentPrescriptionSession\?\.type === 'active_vocabulary'[\s\S]*?currentActiveVocabObserved/);
assert.match(index, /latestAuthenticatedFeedback\?\.persisted === true/);
assert.match(index, /progressEvidence\.includesRecord\([\s\S]*?latestAuthenticatedFeedback\.record_id/);

console.log('Retention behavior: all assertions passed.');
