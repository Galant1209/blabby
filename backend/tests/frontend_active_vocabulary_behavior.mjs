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
const backend = fs.readFileSync(new URL('../main.py', import.meta.url), 'utf8');

const target = {
  id: '11111111-1111-4111-8111-111111111111',
  word: 'overwhelming',
  topic: 'travel',
  review_count: 2,
  active_use_observed: false,
};

const focusWins = retention.prescriptionModel(
  { id: 'focus', weakness_tag: 'lack_detail', resolved: false },
  target,
  6,
);
assert.equal(focusWins.type, 'speaking_focus');
assert.equal(focusWins.source, 'unresolved_weakness');

const active = retention.prescriptionModel(null, target, 6);
assert.equal(active.type, 'active_vocabulary');
assert.equal(active.source, 'reviewed_vocabulary_without_active_use');
assert.equal(active.action.target, 'speaking');
assert.equal(active.action.recommendedQuestionCount, 1);
assert.equal(active.action.activeVocabularyId, target.id);
assert.equal(active.activeVocabulary.word, 'overwhelming');
assert.match(active.title, /overwhelming/);
assert.match(active.action.instruction, /如果自然，可以試著用/);
assert.equal(/一定|必須|失敗|掌握|master/i.test(active.action.instruction), false);

const observedTargetFallsBack = retention.prescriptionModel(
  null,
  { ...target, active_use_observed: true },
  4,
);
assert.equal(observedTargetFallsBack.type, 'vocabulary_review');
assert.equal(retention.prescriptionModel(null, null, 4).type, 'vocabulary_review');
assert.equal(retention.prescriptionModel(null, null, 0).type, 'speaking_baseline');

const bank = [
  { topic: 'Travel', question: 'Do you like travelling?' },
  { topic: 'Travel', question: 'What kind of places do you visit on holiday?' },
  { topic: 'Technology', question: 'How often do you use your phone?' },
];
const firstPick = JSON.parse(JSON.stringify(retention.activeVocabularyQuestion(target, bank)));
const secondPick = JSON.parse(JSON.stringify(retention.activeVocabularyQuestion(target, [...bank].reverse())));
assert.deepEqual(firstPick, secondPick);
assert.equal(firstPick.topic, 'Travel');
assert.equal(bank.some(item => item.question === firstPick.question), true);

const genericPick = retention.activeVocabularyQuestion({ ...target, topic: 'unknown' }, bank);
assert.ok(genericPick && genericPick.question);
assert.equal(bank.some(item => item.question === genericPick.question), true);

const props = JSON.parse(JSON.stringify(retention.activeVocabularyEventProps(active, true)));
assert.deepEqual(Object.keys(props).sort(), [
  'active_use_observed', 'authenticated', 'category', 'source', 'vocabulary_item_id',
].sort());
assert.equal(props.active_use_observed, true);
assert.equal(props.authenticated, true);
const serializedProps = JSON.stringify(props).toLowerCase();
for (const forbidden of ['transcript', 'audio', 'email', 'translation', 'raw_ip', 'answer', 'overwhelming']) {
  assert.equal(serializedProps.includes(forbidden), false);
}

for (const event of [
  'active_vocab_prescription_viewed',
  'active_vocab_prescription_clicked',
  'active_vocab_practice_started',
  'active_vocab_use_observed',
  'active_vocab_practice_completed',
]) {
  assert.equal(index.includes(event), true);
}

assert.match(index, /loadActiveVocabTarget\(\)/);
assert.match(index, /\/api\/vocabulary\/active-use\/current/);
assert.match(index, /prescriptionModel = retention\.prescriptionModel\([\s\S]*?pendingResume,[\s\S]*?resolutionCandidateState,[\s\S]*?activeVocabTarget,[\s\S]*?vocabDueCount/);
assert.match(index, /activeVocabularyQuestion\([\s\S]*?QUESTION_BANK/);
assert.match(index, /form\.append\('active_vocabulary_id', submittedActiveVocabularyId\)/);
assert.match(index, /data\?\.persisted === true[\s\S]*?active_use_observed === true/);
assert.match(index, /active_vocab_use_observed[\s\S]*?activeVocabObservedTracked/);
assert.match(index, /active_vocab_practice_completed[\s\S]*?currentActiveVocabObserved/);
assert.match(index, /currentPrescriptionSession\?\.type === 'active_vocabulary' \? 1 : 3/);
assert.match(index, /這次真的用到了/);
assert.match(index, /真正用進了回答/);
assert.match(index, /sessionClosureActiveVocab\.hidden = !\([\s\S]*?currentActiveVocabObserved/);
assert.match(index, /body\.anonymous-review #prescription-card/);

const anonymousBlock = index.split('async function showAnonymousMain()')[1].split(
  '// Mirrors the gate',
)[0];
assert.equal(anonymousBlock.includes('loadActiveVocabTarget'), false);

assert.match(backend, /def _contains_lexical_expression/);
assert.match(backend, /_contains_lexical_expression\(after_text, taught\)/);
assert.match(backend, /_select_active_vocab_candidate[\s\S]*?_contains_lexical_expression/);
assert.match(backend, /_active_vocab_process_observation[\s\S]*?_contains_lexical_expression/);
assert.match(backend, /active_vocabulary_observation = _active_vocab_process_observation/);
assert.match(backend, /"active_vocabulary":\s+active_vocabulary_observation/);
assert.equal(backend.includes('active_vocab_mastery'), false);

for (const forbidden of ['streak', 'leaderboard', 'mastery score', 'xp', 'level up', 'badge']) {
  assert.equal(new RegExp(`\\b${forbidden.replace(' ', '\\s+')}\\b`, 'i').test(moduleSource), false);
}

console.log('Active vocabulary behavior: all assertions passed.');
