import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const moduleSource = fs.readFileSync(
  new URL('../../frontend/app/progress-evidence.js', import.meta.url),
  'utf8',
);
const context = { globalThis: {} };
vm.createContext(context);
vm.runInContext(moduleSource, context);
const evidence = context.globalThis.BlabbyProgressEvidence;
const index = fs.readFileSync(new URL('../../frontend/app/index.html', import.meta.url), 'utf8');
const progress = fs.readFileSync(new URL('../../frontend/app/progress.html', import.meta.url), 'utf8');

const payload = {
  has_evidence: true,
  weakness: { tag: 'lack_detail', label: '缺少具體細節' },
  before: { record_id: 'before', created_at: '2026-08-01', snippet: 'I think it was good.' },
  after: { record_id: 'after', created_at: '2026-08-10', snippet: 'I liked it because the park was quiet.' },
  observation: { status: 'improvement_observed', label: '最近的回答開始把原因說出來。' },
};
const model = evidence.modelFrom(payload);
assert.equal(model.hasEvidence, true);
assert.equal(model.before.id, 'before');
assert.equal(model.after.id, 'after');
assert.equal(model.observation.status, 'improvement_observed');
assert.equal(evidence.includesRecord(model, 'after'), true);
assert.equal(evidence.includesRecord(model, 'before'), false);

assert.deepEqual(
  JSON.parse(JSON.stringify(evidence.modelFrom({ has_evidence: false }))),
  { hasEvidence: false, reason: 'insufficient_evidence' },
);
assert.equal(evidence.modelFrom({ ...payload, before: { snippet: '' } }).hasEvidence, false);

const props = JSON.parse(JSON.stringify(evidence.eventProps(model, 'progress_page')));
assert.deepEqual(Object.keys(props).sort(), [
  'authenticated', 'observation_status', 'source', 'weakness_category',
].sort());
assert.equal(props.authenticated, true);
const serialized = JSON.stringify(props).toLowerCase();
for (const forbidden of ['transcript', 'snippet', 'email', 'audio', 'raw_ip', 'user_content']) {
  assert.equal(serialized.includes(forbidden), false);
}

for (const event of [
  'progress_evidence_viewed',
  'progress_evidence_opened',
  'progress_evidence_insufficient',
]) {
  assert.equal(index.includes(event) || progress.includes(event), true);
}
assert.match(index, /body\.anonymous-review #progress-evidence-card/);
assert.match(index, /if \(!progressEvidenceModel\.hasEvidence\) \{[\s\S]*?progressEvidenceCard\.hidden = true/);
assert.ok(index.indexOf('id="retention-resume-card"') < index.indexOf('id="progress-evidence-card"'));
assert.ok(index.indexOf('id="progress-evidence-card"') < index.indexOf('id="practice-hub"'));
assert.match(index, /@media \(max-width: 560px\)[\s\S]*?\.progress-evidence-pair \{ grid-template-columns: 1fr; \}/);
assert.match(progress, /再累積幾次練習後，Blabby 會用你自己的回答比較前後變化/);
assert.match(progress, /@media \(max-width: 559px\)[\s\S]*?\.evidence-card__pair \{ grid-template-columns: 1fr; \}/);
assert.equal(/mastery|XP|streak|leaderboard|progress percentage/i.test(moduleSource), false);

console.log('Progress evidence behavior: all assertions passed.');
