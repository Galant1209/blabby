import assert from 'node:assert/strict';
import fs from 'node:fs';
import vm from 'node:vm';

const moduleSource = fs.readFileSync(
  new URL('../../frontend/app/retention.js', import.meta.url),
  'utf8',
);
const index = fs.readFileSync(new URL('../../frontend/app/index.html', import.meta.url), 'utf8');
const context = { globalThis: {} };
vm.createContext(context);
vm.runInContext(moduleSource, context);
const retention = context.globalThis.BlabbyRetention;

const rawCandidate = {
  record_id: 'candidate-record',
  weakness_tag: 'lack_detail',
  label: '缺少具體細節',
  reason: 'recent_clean_evidence',
  observation: { status: 'improvement_observed', label: '最近的回答開始把原因說出來。' },
};
const review = retention.prescriptionModel(
  { id: 'candidate-record', weakness_tag: 'lack_detail', resolved: false },
  rawCandidate,
  { id: 'vocab', word: 'restorative', topic: 'place', review_count: 2 },
  7,
);
assert.equal(review.type, 'resolution_review');
assert.equal(review.action.target, 'resolution');
assert.equal(review.cta, '標記為暫時解決');
assert.equal(review.secondaryCta, '再練一次確認');
assert.equal(review.resolutionCandidate.recordId, 'candidate-record');
assert.equal(
  retention.prescriptionModel(null, retention.resolutionCandidate(rawCandidate), null, 0).type,
  'resolution_review',
);

const retry = retention.resolutionPracticeModel(rawCandidate);
assert.equal(retry.type, 'speaking_focus');
assert.equal(retry.source, 'resolution_candidate_deferred');
assert.equal(retry.action.recommendedQuestionCount, 2);

for (const tag of ['safe_answer', 'grammar_minor', 'off_topic']) {
  assert.equal(retention.resolutionCandidate({ ...rawCandidate, weakness_tag: tag }), null);
}

const props = JSON.parse(JSON.stringify(retention.resolutionEventProps(review)));
assert.deepEqual(Object.keys(props).sort(), [
  'authenticated', 'reason', 'source', 'weakness_category',
].sort());
for (const forbidden of ['transcript', 'snippet', 'email', 'audio', 'record_id', 'answer']) {
  assert.equal(JSON.stringify(props).toLowerCase().includes(forbidden), false);
}

for (const event of [
  'resolution_candidate_viewed',
  'resolution_candidate_confirmed',
  'resolution_candidate_deferred',
  'resolved_weakness_recurred',
]) assert.equal(index.includes(event), true);

assert.match(index, /\/api\/practice-records\/resolution-candidate/);
assert.match(index, /body\?\.has_candidate === true/);
const candidateLoader = index.split('async function loadResolutionCandidate()', 2)[1]
  .split('// Loader for the canonical weakness summary block', 1)[0];
assert.match(candidateLoader, /document\.body\.classList\.contains\('anonymous-review'\)/);
const anonymousEntry = index.split('async function showAnonymousMain()', 2)[1]
  .split('// Mirrors the gate in admin.html', 1)[0];
assert.equal(anonymousEntry.includes('loadResolutionCandidate'), false);
assert.match(index, /if \(prescriptionModel\.type === 'resolution_review'\)/);
assert.match(index, /retention\.resolutionPracticeModel/);
assert.equal(index.includes('resolution_candidate_auto'), false);
assert.equal(moduleSource.toLowerCase().includes('mastery'), false);

console.log('Resolution cycle behavior: all assertions passed.');
