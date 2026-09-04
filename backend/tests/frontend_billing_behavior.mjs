/**
 * TASK 6B-R — focused frontend billing behavior harness.
 *
 * No jsdom/Playwright. Uses Node + minimal DOM stubs to exercise the real
 * checkout / paywall logic extracted from the HTML files.
 *
 * Run via: node backend/tests/frontend_billing_behavior.mjs
 * Or: pytest backend/tests/test_frontend_billing_contracts.py -k behavior
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import assert from "node:assert/strict";
import vm from "node:vm";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const APP = path.resolve(__dirname, "../../frontend/app");

let passed = 0;
function ok(name) {
  passed += 1;
  console.log(`  ok — ${name}`);
}

function read(name) {
  return fs.readFileSync(path.join(APP, name), "utf8");
}

function extractScript(html) {
  const parts = [];
  const re = /<script(?![^>]*\bsrc=)[^>]*>([\s\S]*?)<\/script>/gi;
  let m;
  while ((m = re.exec(html))) {
    // Third-party analytics bootstrap is not the behavior under test.  Skip
    // it so the VM only needs the page's own script and DOM contract.
    if (/posthog\.init\(/.test(m[1])) continue;
    parts.push(m[1]);
  }
  return parts.join("\n");
}

// ── Minimal DOM ─────────────────────────────────────────────────────────────

function el(tag, attrs = {}) {
  const node = {
    tagName: tag.toUpperCase(),
    attrs: { ...attrs },
    children: [],
    style: {},
    hidden: !!attrs.hidden,
    disabled: !!attrs.disabled,
    textContent: "",
    value: attrs.value || "",
    href: attrs.href || "",
    type: attrs.type || "",
    name: attrs.name || "",
    action: attrs.action || "",
    method: attrs.method || "",
    classList: {
      _set: new Set(),
      add(c) { this._set.add(c); },
      remove(c) { this._set.delete(c); },
      contains(c) { return this._set.has(c); },
    },
    listeners: {},
    get innerHTML() {
      return this._innerHTML || "";
    },
    set innerHTML(v) {
      this._innerHTML = String(v);
      if (v === "") this.children = [];
    },
    appendChild(child) {
      this.children.push(child);
      return child;
    },
    replaceChildren() {
      this.children = [];
    },
    addEventListener(type, fn) {
      (this.listeners[type] ||= []).push(fn);
    },
    click() {
      for (const fn of this.listeners.click || []) fn({ preventDefault() {} });
    },
    submit() {
      this._submitted = true;
      for (const fn of this.listeners.submit || []) fn({ preventDefault() {} });
    },
  };
  return node;
}

function makeDocument(idMap) {
  return {
    getElementById(id) {
      return idMap[id] || null;
    },
    createElement(tag) {
      return el(tag);
    },
  };
}

// ── 1–7: upgrade.html checkout behavior ─────────────────────────────────────

async function testUpgradeCheckout() {
  console.log("\n[upgrade.html checkout]");
  const html = read("upgrade.html");

  // Static: price / no old prices / no upgrade_intent
  assert.match(html, /NT\$199/);
  assert.doesNotMatch(html, /NT\$149|NT\$259|NT\$499/);
  assert.equal(html.includes("upgrade_intent"), false);
  ok("no old prices; NT$199 only; no upgrade_intent write");

  // Source XSS: malicious source must not land in hero HTML
  const headlinesMatch = html.match(/const HEADLINES = \{[\s\S]*?\n\};/);
  assert.ok(headlinesMatch, "HEADLINES map present");
  const applyBlock = `
    ${headlinesMatch[0]}
    const source = "<img src=x onerror=alert(1)>";
    const copy = HEADLINES[source] || HEADLINES.direct;
    const h = { innerHTML: "" };
    h.innerHTML = copy.headline;
    globalThis.__hero = h.innerHTML;
    globalThis.__kicker = copy.kicker;
  `;
  const sandbox = { globalThis: {} };
  sandbox.globalThis = sandbox;
  vm.runInNewContext(applyBlock, sandbox);
  assert.equal(sandbox.__kicker, "Blabby Pro");
  assert.equal(sandbox.__hero.includes("<img"), false);
  assert.equal(sandbox.__hero.includes("onerror"), false);
  assert.match(sandbox.__hero, /Thou mayest enter/);
  ok("malicious source query does not inject HTML (falls back to direct)");

  // Behavioral: Bearer JWT + form submit + double-click
  const checkoutBtn = el("button");
  checkoutBtn.disabled = false;
  const checkoutError = el("div");
  const loginNote = el("p");
  loginNote.style.display = "none";
  const ecpayForm = el("form");
  ecpayForm.submit = function () {
    this._submitted = true;
  };

  const idMap = {
    "checkout-btn": checkoutBtn,
    "checkout-error": checkoutError,
    "login-note": loginNote,
    "ecpay-form": ecpayForm,
    "hero-kicker": el("p"),
    "hero-headline": el("h1"),
    "hero-sub": el("p"),
  };

  let fetchCalls = 0;
  const fetchImpl = async (url, opts) => {
    fetchCalls += 1;
    assert.match(url, /\/api\/payment\/create-order$/);
    assert.equal(opts.method, "POST");
    assert.equal(opts.headers.Authorization, "Bearer test-jwt-token");
    return {
      status: 200,
      ok: true,
      async json() {
        return {
          action_url: "https://payment-stage.ecpay.com.tw/Cashier/AioCheckOut/V5",
          params: {
            MerchantID: "3002607",
            MerchantTradeNo: "TESTMTN001",
            CheckMacValue: "ABC",
            TotalAmount: "199",
          },
        };
      },
    };
  };

  // Extract goToCheckout + helpers from the page script and adapt bindings.
  const script = extractScript(html);
  // Isolate the checkout function body by re-declaring deps in a vm context.
  const ctx = {
    console,
    API_BASE: "https://blabby-backend.onrender.com",
    source: "direct",
    checkoutInFlight: false,
    checkoutBtn,
    checkoutError,
    loginNote,
    ecpayForm,
    document: makeDocument(idMap),
    fetch: fetchImpl,
    setError(message) {
      checkoutError.textContent = message;
    },
    async getToken() {
      return "test-jwt-token";
    },
    window: { analytics: { track() {} } },
    Object,
    URLSearchParams,
  };
  ctx.window = ctx;

  // Pull the real goToCheckout function text from the page.
  const fnMatch = script.match(
    /async function goToCheckout\(\) \{[\s\S]*?\n\}\n\ncheckoutBtn\.addEventListener/
  );
  assert.ok(fnMatch, "goToCheckout function extractable");
  const fnSrc = fnMatch[0].replace(/\n\ncheckoutBtn\.addEventListener$/, "");
  vm.runInNewContext(fnSrc + "\n; this.goToCheckout = goToCheckout;", ctx);

  await Promise.all([ctx.goToCheckout(), ctx.goToCheckout(), ctx.goToCheckout()]);
  assert.equal(fetchCalls, 1, "double/triple click must create only one order");
  assert.equal(ecpayForm._submitted, true);
  assert.equal(
    ecpayForm.action,
    "https://payment-stage.ecpay.com.tw/Cashier/AioCheckOut/V5"
  );
  assert.equal(ecpayForm.children.length, 4);
  const names = ecpayForm.children.map((c) => c.name).sort();
  assert.deepEqual(names, [
    "CheckMacValue",
    "MerchantID",
    "MerchantTradeNo",
    "TotalAmount",
  ]);
  ok("Bearer JWT sent; ECPay form built+submitted; double-click → one order");

  // 401 path
  fetchCalls = 0;
  ctx.checkoutInFlight = false;
  checkoutBtn.disabled = false;
  ecpayForm._submitted = false;
  ecpayForm.children = [];
  ctx.fetch = async () => ({ status: 401, ok: false, async json() { return {}; } });
  await ctx.goToCheckout();
  assert.match(checkoutError.textContent, /登入已過期/);
  assert.equal(ecpayForm._submitted, false);
  ok("401 shows recoverable error; no form submit");
}

// ── 8–10: modal source wiring ───────────────────────────────────────────────

function testModalWiring() {
  console.log("\n[quota modal wiring]");
  const writing = read("writing.html");
  const reading = read("reading.html");
  const index = read("index.html");

  // Writing: simulate showProModal
  function runShowProModal(html, source) {
    const overlay = el("div", { hidden: true });
    overlay.hidden = true;
    const kicker = el("p");
    const title = el("h2");
    const sub = el("p");
    const cta = el("a", { href: "/upgrade.html" });
    const idMap = {
      "pro-modal-overlay": overlay,
      "pro-modal-kicker": kicker,
      "pro-modal-title": title,
      "pro-modal-sub": sub,
      "pro-modal-cta": cta,
    };
    const copyMatch = html.match(/const PRO_MODAL_COPY = \{[\s\S]*?\n\s*\};/);
    assert.ok(copyMatch, "PRO_MODAL_COPY present");
    const fnMatch = html.match(/function showProModal\(source\) \{[\s\S]*?\n\s*\}/);
    assert.ok(fnMatch, "showProModal present");
    const ctx = {
      document: makeDocument(idMap),
      encodeURIComponent,
    };
    // reading.html uses $() helper
    ctx.$ = (id) => idMap[id];
    vm.runInNewContext(
      copyMatch[0] + "\n" + fnMatch[0] + "\n; showProModal(" + JSON.stringify(source) + ");",
      ctx
    );
    return { overlay, cta, title, kicker, sub };
  }

  const w = runShowProModal(writing, "writing_quota");
  assert.equal(w.overlay.hidden, false);
  assert.match(w.title.textContent, /three essays/);
  assert.equal(w.cta.href, "/upgrade.html?source=writing_quota");
  ok("Writing daily quota opens modal → upgrade.html?source=writing_quota");

  const w1 = runShowProModal(writing, "writing_task1");
  assert.equal(w1.overlay.hidden, false);
  assert.equal(w1.cta.href, "/upgrade.html?source=writing_task1");
  ok("Writing pro_required opens modal → writing_task1");

  const r = runShowProModal(reading, "reading_quota");
  assert.equal(r.overlay.hidden, false);
  assert.equal(r.cta.href, "/upgrade.html?source=reading_quota");
  ok("Reading quota_exceeded modal → upgrade.html?source=reading_quota");

  for (const [src, needle] of [
    ["feedback_quota", "twenty counsels"],
    ["drill_quota", "twenty trials"],
    ["part2_quota", "ten long turns"],
  ]) {
    const m = runShowProModal(index, src);
    assert.equal(m.overlay.hidden, false);
    assert.match(m.title.textContent, new RegExp(needle));
    assert.equal(m.cta.href, `/upgrade.html?source=${src}`);
  }
  ok("Speaking three quota errors map to distinct modal sources");

  // showP2Modal still exists for mic errors
  assert.match(index, /function showP2Modal\(/);
  assert.match(index, /showP2Modal\('無法錄音'/);
  ok("showP2Modal retained for non-quota uses");
}

// ── 11–12: hub Pro badge + fail-soft ────────────────────────────────────────

function testHubProResolution() {
  console.log("\n[hub.html Pro resolution]");
  const html = read("hub.html");
  const resolveMatch = html.match(
    /function resolveIsPro\([\s\S]*?\n    \}\n\n    function renderPro/
  );
  assert.ok(resolveMatch, "resolveIsPro present");
  const renderMatch = html.match(/function renderPro\([\s\S]*?\n    \}\n\n    function renderQuestion/);
  assert.ok(renderMatch, "renderPro present");

  const badge = el("span", { hidden: true });
  badge.hidden = true;
  const upgradeBadge = el("a", { hidden: true });
  upgradeBadge.hidden = true;
  const expiryEl = el("span", { hidden: true });
  expiryEl.hidden = true;
  const idMap = {
    "pro-badge": badge,
    "upgrade-badge": upgradeBadge,
    "pro-expiry": expiryEl,
  };
  const ctx = {
    document: makeDocument(idMap),
    Date,
    Number,
  };
  vm.runInNewContext(resolveMatch[0].replace(/\n\n    function renderPro$/, "") + "\n" + renderMatch[0].replace(/\n\n    function renderQuestion$/, "") + "\n; this.resolveIsPro=resolveIsPro; this.renderPro=renderPro;", ctx);

  // Pro via quota
  let isPro = ctx.resolveIsPro(
    { status: "fulfilled", value: { is_pro: true } },
    { status: "rejected", reason: new Error("nope") }
  );
  assert.equal(isPro, true);
  ctx.renderPro(isPro, { expires_at: "2026-08-29T00:00:00Z" });
  assert.equal(badge.hidden, false);
  assert.equal(upgradeBadge.hidden, true);
  ok("Pro user hides upgrade badge");

  // Free via quota
  isPro = ctx.resolveIsPro(
    { status: "fulfilled", value: { is_pro: false } },
    { status: "fulfilled", value: { subscription: null } }
  );
  assert.equal(isPro, false);
  ctx.renderPro(isPro, null);
  assert.equal(badge.hidden, true);
  assert.equal(upgradeBadge.hidden, false);
  ok("Free user shows upgrade badge");

  // Both APIs fail → unknown → do NOT claim free (upgrade stays hidden)
  isPro = ctx.resolveIsPro(
    { status: "rejected", reason: new Error("quota down") },
    { status: "rejected", reason: new Error("sub down") }
  );
  assert.equal(isPro, null);
  ctx.renderPro(isPro, null);
  assert.equal(badge.hidden, true);
  assert.equal(upgradeBadge.hidden, true);
  ok("subscription/quota failure does not false-demote to Free / crash Hub");

  // Quota failed but active subscription → still Pro
  const future = new Date(Date.now() + 86400000).toISOString();
  isPro = ctx.resolveIsPro(
    { status: "rejected", reason: new Error("quota down") },
    { status: "fulfilled", value: { subscription: { status: "active", expires_at: future } } }
  );
  assert.equal(isPro, true);
  ok("subscription API fallback keeps Pro when quota probe fails");
}

// ── Hub boot fail-soft (Promise.allSettled present) ─────────────────────────

function testHubDoesNotBlockOnSubscription() {
  const html = read("hub.html");
  assert.match(html, /Promise\.allSettled/);
  assert.match(html, /\/api\/user\/subscription/);
  ok("Hub fans out with allSettled — subscription failure cannot block page");
}

// ── success.html activation truthfulness (TASK 6B-S) ────────────────────────

function loadSuccessPage(opts = {}) {
  const html = read("success.html");
  // Static HTML must not claim Pro before JS confirms.
  assert.equal(html.includes("你現在是 Pro"), false);
  assert.match(html, /付款結果確認中/);

  const title = el("div");
  title.textContent = "The ledger is being examined.";
  const subtitle = el("div");
  subtitle.textContent = "付款結果確認中，請稍候。";
  const spinner = el("div");
  spinner.hidden = false;
  const expiry = el("div");
  expiry.hidden = true;
  const order = el("div");
  order.hidden = true;
  const ctaHub = el("a", { href: "/hub.html", hidden: true });
  ctaHub.hidden = true;
  const btnRetry = el("button", { hidden: true });
  btnRetry.hidden = true;
  const ctaLogin = el("a", { href: "/index.html", hidden: true });
  ctaLogin.hidden = true;

  const idMap = {
    title,
    subtitle,
    spinner,
    expiry,
    "order-id": order,
    "cta-hub": ctaHub,
    "btn-retry": btnRetry,
    "cta-login": ctaLogin,
  };

  const timers = [];
  let now = Date.now();
  const fetchImpl =
    opts.fetch ||
    (async () => ({
      status: 200,
      ok: true,
      async json() {
        return { subscription: null };
      },
    }));

  const sessionToken = opts.token === undefined ? "test-jwt" : opts.token;
  const tracked = [];
  const storage = new Map();
  const ctx = {
    console,
    setTimeout(fn, ms) {
      const id = { fn, ms, cleared: false };
      timers.push(id);
      return id;
    },
    clearTimeout(id) {
      if (id) id.cleared = true;
    },
    Date: class extends Date {
      constructor(...a) {
        if (a.length === 0) return new Date(now);
        // eslint-disable-next-line prefer-rest-params
        super(...a);
      }
      static now() {
        return now;
      }
    },
    fetch: fetchImpl,
    URLSearchParams,
    encodeURIComponent,
    document: makeDocument(idMap),
    window: {
      BLABBY_CONFIG: { supabaseUrl: "https://example.supabase.co", supabaseAnonKey: "anon" },
      __BLABBY_SUCCESS_NO_BOOT__: true,
      location: { search: opts.search || "" },
      analytics: {
        track(event, props) { tracked.push({ event, props }); },
        identify() {},
      },
      sessionStorage: {
        getItem(key) { return storage.has(key) ? storage.get(key) : null; },
        setItem(key, value) { storage.set(key, String(value)); },
      },
      addEventListener() {},
      supabase: {
        createClient() {
          return {
            auth: {
              async getSession() {
                if (!sessionToken) return { data: { session: null } };
                return { data: { session: { access_token: sessionToken } } };
              },
            },
          };
        },
      },
    },
  };
  // Make Date.now work for isActiveSubscription comparisons using real Date for ISO parse
  ctx.Date = Date;
  ctx.window.document = ctx.document;
  ctx.window.fetch = fetchImpl;
  ctx.window.setTimeout = ctx.setTimeout;
  ctx.window.clearTimeout = ctx.clearTimeout;
  ctx.window.URLSearchParams = URLSearchParams;

  const script = extractScript(html);
  // Only the inline page script (last one without src) — extractScript already skips src.
  // The page has one inline block after the CDN script tag (CDN has src, skipped).
  vm.runInNewContext(script, ctx, { timeout: 5000 });

  assert.ok(ctx.window.__BLABBY_SUCCESS__, "__BLABBY_SUCCESS__ exported");

  async function flushTimers() {
    // Run due timers once (poll schedules one at a time).
    const due = timers.filter((t) => !t.cleared);
    timers.length = 0;
    for (const t of due) {
      await t.fn();
    }
  }

  return {
    api: ctx.window.__BLABBY_SUCCESS__,
    els: { title, subtitle, spinner, expiry, order, ctaHub, btnRetry, ctaLogin },
    timers,
    flushTimers,
    tracked,
    setFetch(fn) {
      ctx.fetch = fn;
      ctx.window.fetch = fn;
    },
  };
}

async function testSuccessPageTruthfulness() {
  console.log("\n[success.html activation truthfulness]");

  // 1. Initial DOM / checking is not "你現在是 Pro"
  {
    const html = read("success.html");
    assert.equal(html.includes("你現在是 Pro"), false);
    assert.equal(html.includes("Thy payment has been received."), false);
    assert.match(html, /付款結果確認中，請稍候/);
    const { api, els } = loadSuccessPage({ token: null });
    api.renderState("checking");
    assert.match(els.subtitle.textContent, /付款結果確認中/);
    assert.equal(els.subtitle.textContent.includes("Pro 已開通"), false);
    ok("initial / checking is not Activated Pro claim");
  }

  // 2. Activated only when active + unexpired
  {
    const future = new Date(Date.now() + 86400000).toISOString();
    const { api, els } = loadSuccessPage();
    assert.equal(
      api.isActiveSubscription({ status: "active", expires_at: future }),
      true
    );
    api.renderState("activated", { expiresAt: future });
    assert.match(els.subtitle.textContent, /Pro 已開通/);
    assert.equal(els.ctaHub.hidden, false);
    assert.equal(els.expiry.hidden, false);
    assert.match(els.expiry.textContent, /Pro 有效至/);
    ok("is_pro/active subscription → Activated + safe expiry");
  }

  // 3. pending does not show Activated
  {
    const { api, els } = loadSuccessPage();
    api.renderState("pending");
    assert.equal(els.subtitle.textContent.includes("Pro 已開通"), false);
    assert.match(els.subtitle.textContent, /仍在確認中/);
    assert.equal(els.btnRetry.hidden, false);
    ok("pending does not show Activated");
  }

  // 4. inactive does not show Activated
  {
    const { api, els } = loadSuccessPage();
    assert.equal(
      api.isActiveSubscription({ status: "cancelled", expires_at: "2099-01-01" }),
      false
    );
    api.renderState("inactive");
    assert.equal(els.subtitle.textContent.includes("Pro 已開通"), false);
    assert.match(els.subtitle.textContent, /尚未開通/);
    ok("inactive does not show Activated");
  }

  // 5. network error does not show Activated
  {
    const { api, els } = loadSuccessPage();
    api.renderState("error");
    assert.equal(els.subtitle.textContent.includes("Pro 已開通"), false);
    assert.match(els.subtitle.textContent, /無法確認/);
    ok("network/error does not show Activated");
  }

  // 6. 401 → login / authExpired
  {
    let calls = 0;
    const { api, els, flushTimers } = loadSuccessPage({
      token: "expired-jwt",
      fetch: async () => {
        calls += 1;
        return { status: 401, ok: false, async json() { return {}; } };
      },
    });
    api.startPolling("expired-jwt");
    await flushTimers();
    // first cycle is sync until first await — need microtask flush
    await Promise.resolve();
    await flushTimers();
    // runPollCycle is async; wait for it
    await new Promise((r) => setTimeout(r, 20));
    assert.equal(calls, 1);
    assert.equal(els.ctaLogin.hidden, false);
    assert.match(els.subtitle.textContent, /登入已過期|請先登入/);
    assert.equal(els.subtitle.textContent.includes("Pro 已開通"), false);
    ok("401 prompts re-login; not Activated");
  }

  // 7–8. retry does not create duplicate polling; max attempts bounded
  {
    let calls = 0;
    const { api, flushTimers } = loadSuccessPage({
      fetch: async (url, opts) => {
        calls += 1;
        assert.match(String(url), /\/api\/user\/subscription$/);
        assert.equal(opts.headers.Authorization, "Bearer test-jwt");
        return {
          status: 200,
          ok: true,
          async json() {
            return { subscription: null };
          },
        };
      },
    });
    assert.equal(api.MAX_POLLS, 5);
    const started = api.startPolling("test-jwt");
    assert.equal(started, true);
    assert.equal(api.startPolling("test-jwt"), false); // single-flight
    // Drain up to MAX_POLLS cycles
    for (let i = 0; i < 12; i++) {
      await new Promise((r) => setTimeout(r, 5));
      await flushTimers();
      if (!api.poller.active) break;
    }
    assert.ok(calls <= api.MAX_POLLS, `calls=${calls} <= MAX_POLLS`);
    assert.equal(api.poller.active, false);
    assert.equal(api.poller.attempts, api.MAX_POLLS);
    ok("retry/single-flight + polling capped at MAX_POLLS; Bearer JWT sent");
  }

  // 9. query string cannot inject HTML
  {
    const { api, els } = loadSuccessPage({
      search: "?order=<img src=x onerror=alert(1)>",
    });
    const label = api.applyOrderFromQuery("?order=<img src=x onerror=alert(1)>");
    assert.equal(label, null);
    assert.equal(els.order.hidden, true);
    assert.equal(els.order.textContent.includes("<img"), false);
    // Valid order uses textContent
    api.applyOrderFromQuery("?order=20260730TB5J238FCAUO");
    assert.equal(els.order.textContent, "Order 20260730TB5J238FCAUO");
    assert.equal(els.order.hidden, false);
    ok("query string cannot inject HTML; safe order via textContent");
  }

  // 10. expiry safe render (no HTML)
  {
    const { api, els } = loadSuccessPage();
    api.renderState("activated", {
      expiresAt: "2026-08-29T08:26:24.000Z<script>alert(1)</script>",
    });
    // Invalid date → hidden; no script echo
    assert.equal(els.expiry.textContent.includes("<script>"), false);
    api.renderState("activated", { expiresAt: "2026-08-29T08:26:24.000Z" });
    assert.match(els.expiry.textContent, /^Pro 有效至 \d{4}\/\d{2}\/\d{2}$/);
    ok("expiry rendered safely via textContent");
  }

  // 11. arrival is observational and one-shot per order in one tab
  {
    const page = loadSuccessPage({ search: "?order=20260904ABC" });
    assert.equal(page.api.trackSuccessView("?order=20260904ABC"), true);
    assert.equal(page.api.trackSuccessView("?order=20260904ABC"), false);
    assert.equal(
      JSON.stringify(page.tracked),
      JSON.stringify([
        { event: "checkout_success_viewed", props: { order_id: "20260904ABC" } },
      ]),
    );
    ok("success arrival emits once per order without claiming payment truth");
  }

  // 12–13. no session → login, no authorized API call
  {
    let calls = 0;
    const { api, els } = loadSuccessPage({
      token: null,
      fetch: async () => {
        calls += 1;
        throw new Error("should not fetch");
      },
    });
    api.renderState("login");
    assert.equal(els.ctaLogin.hidden, false);
    assert.match(els.subtitle.textContent, /請先登入/);
    assert.equal(calls, 0);
    // startPolling without going through boot still needs token from caller;
    // getToken returns null — onRetryClick path
    await api.onRetryClick();
    await new Promise((r) => setTimeout(r, 10));
    assert.equal(calls, 0);
    assert.equal(els.ctaLogin.hidden, false);
    ok("no session → login prompt; no authorized API call");
  }
}

async function main() {
  console.log("frontend_billing_behavior.mjs");
  await testUpgradeCheckout();
  testModalWiring();
  testHubProResolution();
  testHubDoesNotBlockOnSubscription();
  await testSuccessPageTruthfulness();
  console.log(`\nAll ${passed} behavior assertions passed.`);
}

main().catch((err) => {
  console.error("\nFAILED:", err);
  process.exit(1);
});
