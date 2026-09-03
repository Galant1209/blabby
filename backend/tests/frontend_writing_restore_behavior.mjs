/**
 * writing.html 的 ?submission= 還原路徑。
 *
 * 批改原本只存在於那一次 POST 的回應裡，重整即消失。交卷後把 submission id
 * 寫進 URL、載入時據此還原，是這條路徑的全部。
 *
 * 最需要釘住的是「URL 是狀態，每條離開還原畫面的路都要清掉它」。
 * leaveRestoredView() 現在掛在 writeAgain() 與 getQuestion()。未來有人加第三
 * 條離開路徑而忘了清，症狀是「按 Write Again 寫新的一篇再重整，跳回上一篇的
 * 批改」—— 幾乎不可能從症狀聯想到根因。
 *
 * 手法與 frontend_billing_behavior.mjs 相同：以大括號配對取出真實函式，在 vm
 * 裡配 DOM stub 執行。驗的是這些函式實際做了什麼，不是它們呼叫了誰。
 *
 * 用法：
 *   node frontend_writing_restore_behavior.mjs                 只跑行為檢查
 *   node frontend_writing_restore_behavior.mjs payloads.json   另跑分數一致性
 *
 * payloads.json 由 pytest 端用真的 build_writing_feedback_view() 產生，內容是
 * {"post": <交卷回應>, "get": <還原回應>}。分數一致性不能在這裡自己造資料 ——
 * 那樣驗的是這支檔案的副本，不是後端真正回什麼。
 */

import fs from 'node:fs';
import vm from 'node:vm';
import assert from 'node:assert/strict';

const SRC = fs.readFileSync(
  new URL('../../frontend/app/writing.html', import.meta.url), 'utf8');

const pass = (msg) => console.log('  ok  ' + msg);

/** 以大括號配對取出函式本體。regex 對付不了巢狀大括號。 */
function grab(signature) {
  const start = SRC.indexOf(signature);
  assert.ok(start >= 0, 'missing function: ' + signature);
  let depth = 0, i = SRC.indexOf('{', start);
  for (; i < SRC.length; i++) {
    if (SRC[i] === '{') depth++;
    else if (SRC[i] === '}') { depth--; if (depth === 0) break; }
  }
  return SRC.slice(start, i + 1);
}

const RESTORE_FNS = [
  'function showElement(id)',
  'function hideElement(id)',
  'function renderFeedback(data)',
  'function renderEssayRecall(text)',
  'function clearSubmissionParam()',
  'function leaveRestoredView()',
  'async function restoreSubmissionFromUrl()',
].map(grab).join('\n');

const PANEL_IDS = [
  'essay-recall', 'essay-recall-text', 'priority-fix-text', 'band-overall',
  'criteria-cards', 'feedback-panel', 'question-panel', 'writing-panel',
  'task1-lock', 'error-display',
];

function makeElement(id) {
  return {
    id, style: { display: '' }, textContent: '', innerHTML: '', className: '',
    _children: [],
    appendChild(child) { this._children.push(child); },
  };
}

function makeContext({ search, fetchImpl, fns = RESTORE_FNS }) {
  const els = {};
  PANEL_IDS.forEach((id) => { els[id] = makeElement(id); });

  const tracked = [];
  const historyUrls = [];
  const ctx = {
    API_BASE: 'https://api.test',
    currentSession: { access_token: 'test-token' },
    document: {
      getElementById: (id) => els[id] || (els[id] = makeElement(id)),
      createElement: () => makeElement('created'),
      createTextNode: (text) => ({ nodeValue: text }),
    },
    window: {
      location: {
        href: 'https://app.test/writing.html' + search,
        pathname: '/writing.html', search, hash: '',
      },
      analytics: { track: (event) => tracked.push(event) },
    },
    history: { replaceState: (_s, _t, url) => historyUrls.push(url) },
    fetch: fetchImpl,
    URL, URLSearchParams, console,
  };
  ctx.window.window = ctx.window;
  vm.createContext(ctx);
  vm.runInContext(fns, ctx);
  return { ctx, els, tracked, historyUrls };
}

const SUBMISSION = {
  submission_id: 'abc-123',
  word_count: 155,
  band_overall: 4.0,
  priority_fix: 'Compare the two lines, do not just list them.',
  essay_text: 'The chart shows a steady rise.\n\nSecond paragraph.',
  criteria: {
    task_achievement:   { band: 4.5, feedback: 'fb ta',  fix: 'fix ta' },
    coherence_cohesion: { band: 4.0, feedback: 'fb cc',  fix: 'fix cc' },
    lexical_resource:   { band: 4.0, feedback: 'fb lr',  fix: 'fix lr' },
    grammatical_range:  { band: 4.0, feedback: 'fb gra', fix: 'fix gra' },
  },
};

// ── 1. 成功還原 ────────────────────────────────────────────────────────────
async function testSuccessfulRestore() {
  const { ctx, els, tracked, historyUrls } = makeContext({
    search: '?submission=abc-123',
    fetchImpl: async (url, opts) => {
      assert.equal(url, 'https://api.test/api/writing/submission/abc-123');
      assert.equal(opts.headers.Authorization, 'Bearer test-token');
      return { ok: true, json: async () => SUBMISSION };
    },
  });
  await ctx.restoreSubmissionFromUrl();

  assert.equal(els['essay-recall'].style.display, 'block', '原文區塊未顯示');
  assert.equal(els['essay-recall-text'].textContent, SUBMISSION.essay_text);
  assert.equal(els['priority-fix-text'].textContent, SUBMISSION.priority_fix);
  assert.equal(String(els['band-overall'].textContent), '4');
  assert.equal(els['criteria-cards']._children.length, 4, '四張評分卡未齊');
  assert.equal(els['feedback-panel'].style.display, 'block');
  assert.equal(els['question-panel'].style.display, 'none', '未跳過題目區');
  assert.equal(els['writing-panel'].style.display, 'none', '未跳過作答區');
  assert.deepEqual(tracked, ['writing_submission_restored']);
  assert.deepEqual(historyUrls, [], '成功時不該動 URL');
  pass('成功還原：原文 + 四張評分卡 + 跳過題目/作答區 + 發事件');
}

// ── 2. 404 靜默 ────────────────────────────────────────────────────────────
async function test404IsSilent() {
  const { ctx, els, tracked, historyUrls } = makeContext({
    search: '?submission=does-not-exist',
    fetchImpl: async () => ({ ok: false, status: 404, json: async () => ({}) }),
  });
  await ctx.restoreSubmissionFromUrl();

  // 連結過期、換帳號、手動改網址、別人的 submission —— 沒有一種是使用者的錯。
  assert.equal(els['error-display'].style.display, '', '不得顯示錯誤訊息');
  assert.equal(els['feedback-panel'].style.display, '');
  assert.equal(els['essay-recall'].style.display, '');
  assert.deepEqual(historyUrls, ['/writing.html'], 'URL 參數未清掉');
  assert.deepEqual(tracked, [], '失敗不得發事件');
  pass('404 靜默：無錯誤訊息、參數清掉、不發事件');
}

// ── 3. 網路例外靜默 ────────────────────────────────────────────────────────
async function testNetworkErrorIsSilent() {
  const { ctx, els, historyUrls } = makeContext({
    search: '?submission=abc-123',
    fetchImpl: async () => { throw new Error('offline'); },
  });
  await ctx.restoreSubmissionFromUrl();

  assert.equal(els['error-display'].style.display, '');
  assert.deepEqual(historyUrls, ['/writing.html']);
  pass('網路例外靜默');
}

// ── 4. 無參數 ──────────────────────────────────────────────────────────────
async function testNoParamDoesNothing() {
  const { ctx, els, tracked, historyUrls } = makeContext({
    search: '',
    fetchImpl: async () => { throw new Error('不該打 API'); },
  });
  await ctx.restoreSubmissionFromUrl();

  assert.equal(els['feedback-panel'].style.display, '');
  assert.equal(els['question-panel'].style.display, '');
  assert.deepEqual(historyUrls, []);
  assert.deepEqual(tracked, []);
  pass('無參數：不打 API、不動任何面板');
}

// ── 5. 離開還原畫面要清狀態 ────────────────────────────────────────────────
async function testLeavingClearsState() {
  const { ctx, els, historyUrls } = makeContext({
    search: '?submission=abc-123',
    fetchImpl: async () => ({ ok: true, json: async () => SUBMISSION }),
  });
  await ctx.restoreSubmissionFromUrl();
  ctx.leaveRestoredView();

  assert.equal(els['essay-recall'].style.display, 'none', '原文區塊未收起');
  assert.deepEqual(historyUrls, ['/writing.html'], 'URL 參數未清掉');
  pass('leaveRestoredView：原文收起 + 參數清掉');
}

// ── 6. 空 essay_text ───────────────────────────────────────────────────────
async function testEmptyEssayShowsNoBox() {
  const { ctx, els } = makeContext({
    search: '?submission=abc-123',
    fetchImpl: async () => ({ ok: true, json: async () => ({ ...SUBMISSION, essay_text: '' }) }),
  });
  await ctx.restoreSubmissionFromUrl();

  assert.equal(els['essay-recall'].style.display, '', '空原文不該顯示空框');
  assert.equal(els['feedback-panel'].style.display, 'block', '批改仍應顯示');
  pass('空 essay_text：不顯示空框，批改照常');
}

// ── 7. 靜態合約 ────────────────────────────────────────────────────────────
function testStaticContracts() {
  assert.ok(
    SRC.includes("history.replaceState(null, '', `?submission=${data.submission_id}`);"),
    '交卷後未把 submission id 寫進 URL');
  // pushState 會在瀏覽器歷史多一格，「上一頁」回到一個已經交出去的作答畫面。
  assert.ok(!SRC.includes('history.pushState'), '不得使用 pushState');

  // 這兩條是本檔存在的主要理由：URL 是狀態，每條離開路徑都要清理它。
  assert.ok(SRC.includes('function writeAgain() {\n            leaveRestoredView();'),
    'writeAgain 未清理還原狀態');
  assert.ok(SRC.includes('async function getQuestion() {\n            leaveRestoredView();'),
    'getQuestion 未清理還原狀態');
  assert.ok(SRC.includes('await restoreSubmissionFromUrl();'),
    'initPage 未呼叫還原');

  // renderFeedback 的職責是畫批改。原文是獨立區塊、獨立函式。
  assert.ok(!grab('function renderFeedback(data)').includes('essay'),
    'renderFeedback 的職責被擴張到原文');
  pass('靜態合約：replaceState 非 pushState、兩條離開路徑都清理、renderFeedback 未擴張');
}

// ── 8. 分數一致性（payload 由 pytest 端的真 builder 供給）──────────────────
function testBandRenderingIsIdentical(payloadPath) {
  const { post, get } = JSON.parse(fs.readFileSync(payloadPath, 'utf8'));

  const renderOnly = grab('function renderFeedback(data)');
  function render(payload) {
    const els = {};
    ['priority-fix-text', 'band-overall', 'criteria-cards']
      .forEach((id) => { els[id] = makeElement(id); });
    const ctx = {
      document: {
        getElementById: (id) => els[id] || (els[id] = makeElement(id)),
        createElement: () => makeElement('created'),
        createTextNode: (text) => ({ nodeValue: text }),
      },
    };
    vm.createContext(ctx);
    vm.runInContext(renderOnly, ctx);
    ctx.renderFeedback(payload);
    return {
      overall: String(els['band-overall'].textContent),
      // 每張卡的 <h3> 第一個子節點是 band badge
      bands: els['criteria-cards']._children.map(
        (card) => String(card._children[0]._children[0].textContent)),
      cards: els['criteria-cards']._children.length,
      priority: els['priority-fix-text'].textContent,
    };
  }

  const before = render(post);
  const after = render(get);
  console.log('      重整前 overall=' + before.overall + ' bands=[' + before.bands.join(', ') + ']');
  console.log('      重整後 overall=' + after.overall  + ' bands=[' + after.bands.join(', ')  + ']');

  // numeric 欄位經 PostgREST 序列化成字串（"4.0"），交卷當下的值來自 LLM 解析
  // （4.0）。後端的 _writing_band() 收斂兩者；沒有它，同一份批改會在交卷當下
  // 顯示 4、重整後顯示 4.0 —— 使用者看到兩個不同的分數。
  assert.equal(before.overall, after.overall, 'band_overall 渲染不一致');
  assert.deepEqual(before.bands, after.bands, 'criteria band 渲染不一致');
  assert.equal(before.priority, after.priority);
  assert.equal(before.cards, 4);
  assert.equal(after.cards, 4);
  pass('分數一致性：重整前後 overall 與四個 criteria band 逐項相同');
}

// ── run ────────────────────────────────────────────────────────────────────
console.log('[writing restore behavior]');
await testSuccessfulRestore();
await test404IsSilent();
await testNetworkErrorIsSilent();
await testNoParamDoesNothing();
await testLeavingClearsState();
await testEmptyEssayShowsNoBox();
testStaticContracts();

const payloadPath = process.argv[2];
if (payloadPath) {
  testBandRenderingIsIdentical(payloadPath);
} else {
  console.log('  --  分數一致性略過（未提供 payload 檔）');
}
