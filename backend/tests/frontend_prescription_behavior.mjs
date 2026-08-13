import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const moduleSource = fs.readFileSync(
  new URL('../../frontend/app/retention.js', import.meta.url),
  'utf8',
);
const context = { globalThis: {} };
vm.createContext(context);
vm.runInContext(moduleSource, context);
const retention = context.globalThis.BlabbyRetention;
const index = fs.readFileSync(new URL('../../frontend/app/index.html', import.meta.url), 'utf8');

const expectedTitles = {
  weak_vocab: '把更精確的詞說出口',
  safe_answer: '把回答說得更具體',
  lack_detail: '把理由說完整',
  grammar_minor: '先把意思說完整',
  off_topic: '先直接回答問題',
};

for (const [tag, title] of Object.entries(expectedTitles)) {
  const record = { id: `record-${tag}`, weakness_tag: tag, resolved: false };
  const model = retention.prescriptionModel(record, 8);
  const focus = retention.focusFrom(record);
  assert.equal(model.type, 'speaking_focus');
  assert.equal(model.source, 'unresolved_weakness');
  assert.equal(model.title, title);
  assert.equal(model.action.target, 'speaking');
  assert.equal(model.action.recommendedQuestionCount, 2);
  assert.equal(model.action.weaknessTag, tag);
  assert.equal(model.action.instruction, focus.nextAction.label);
  assert.equal(model.canResolve, true);
}

const resolvedFallsThrough = retention.prescriptionModel(
  { id: 'resolved', weakness_tag: 'lack_detail', resolved: true },
  4,
);
assert.equal(resolvedFallsThrough.type, 'vocabulary_review');
assert.equal(resolvedFallsThrough.action.target, 'vocabulary');
assert.equal(resolvedFallsThrough.cta, '開始複習');
assert.match(resolvedFallsThrough.description, /4 個/);

const cappedDue = retention.prescriptionModel(null, 10);
assert.equal(cappedDue.type, 'vocabulary_review');
assert.equal(/10 個|待辦|任務/.test(cappedDue.description), false);

const baseline = retention.prescriptionModel(null, 0);
assert.equal(baseline.type, 'speaking_baseline');
assert.equal(baseline.title, '先做一輪口說');
assert.equal(baseline.action.recommendedQuestionCount, 3);
assert.equal(baseline.cta, '開始口說');
assert.match(baseline.description, /才能開始記住/);

const unknown = retention.prescriptionModel(
  { id: 'unknown', weakness_tag: 'future_tag', resolved: false },
  0,
);
assert.equal(unknown.type, 'speaking_focus');
assert.equal(unknown.title, '先把一個回答說完整');
assert.equal(unknown.action.target, 'speaking');

const props = JSON.parse(JSON.stringify(retention.prescriptionEventProps(
  retention.prescriptionModel({ id: 'focus', weakness_tag: 'lack_detail', resolved: false }, 0),
)));
assert.deepEqual(Object.keys(props).sort(), [
  'authenticated',
  'prescription_type',
  'recommended_question_count',
  'source',
  'target',
  'weakness_category',
].sort());
assert.equal(props.authenticated, true);
const serialized = JSON.stringify(props).toLowerCase();
for (const forbidden of ['transcript', 'snippet', 'email', 'audio', 'raw_ip', 'raw_answer']) {
  assert.equal(serialized.includes(forbidden), false);
}

for (const event of [
  'prescription_viewed',
  'prescription_clicked',
  'prescription_started',
  'prescription_session_completed',
]) {
  assert.equal(index.includes(event), true);
}

assert.equal(index.includes('id="retention-resume-card"'), false);
assert.ok(index.indexOf('id="prescription-card"') < index.indexOf('id="progress-evidence-card"'));
assert.ok(index.indexOf('id="progress-evidence-card"') < index.indexOf('id="practice-hub"'));
assert.match(index, /body\.anonymous-review #prescription-card/);
assert.match(index, /prescriptionModel = retention\.prescriptionModel\(pendingResume, activeVocabTarget, vocabDueCount\)/);
assert.match(index, /loadPendingResume\(\)[\s\S]*?loadVocabDueCount\(\)/);
assert.equal(index.includes('/api/prescription/'), false);
assert.match(index, /\/api\/vocabulary\/active-use\/current/);
assert.match(index, /window\.location\.href = '\/vocabulary\.html\?source=prescription'/);
assert.match(index, /currentPrescriptionSession = prescriptionModel/);
assert.match(index, /這一輪：\$\{currentPrescriptionSession\.title\} · 建議 \$\{count\} 題/);
assert.match(index, /建議題數不是限制/);
assert.match(index, /authenticatedSessionAnswerCount - prescriptionStartAnswerCount/);
assert.equal(index.includes('recommendedQuestionCount].disabled'), false);
assert.match(index, /current_focus_resolved[\s\S]*?prescription_card/);
assert.match(index, /@media \(max-width: 560px\)[\s\S]*?\.retention-actions > \* \{ width: 100%; \}/);

for (const forbidden of ['streak', 'leaderboard', 'mastery', 'quest', 'mission', 'challenge', 'progress bar', 'countdown']) {
  assert.equal(new RegExp(`\\b${forbidden.replace(' ', '\\s+')}\\b`, 'i').test(moduleSource), false);
}

console.log('Prescription behavior: all assertions passed.');
