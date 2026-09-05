# Blabby Revenue Funnel — Round M

**POSTHOG_QUERY = BLOCKED**

**NO REAL METRICS — QUERY ACCESS BLOCKED**

The canonical funnel and read-only query plan are defined. Production frontend capture configuration is verified in served assets; actual ingestion, backend capture configuration, project identity and production funnel metrics remain unverified. No largest-drop-off claim is supportable.

## 1. Preflight and scope

Fresh commands in `/Users/yichengchiu/dev/Blabby/blabby` returned:

| Item | Observed |
|---|---|
| HEAD | `093fb7e5a02df575c069eb6737c599cab4ac2542` |
| origin/main | `88eada001f89597eb7721ea425c6fd4af23edde3` |
| `git rev-list --left-right --count origin/main...HEAD` | `0 6` — ahead/behind 6/0 |
| Working tree | clean |

This investigation reread current code and tests rather than treating earlier documents as implementation evidence. No payment event, identity, pricing, callback, activation or application behavior was changed. Blabby Pro remains **NT$199 for 30 days**, with the existing `plan="monthly"` value; no new plan or taxonomy is introduced.

Read-only production evidence was collected on **2026-09-05, approximately 04:34 UTC**. Browser JavaScript was not executed: fetching the public HTML did not generate synthetic production events.

## 2. Current telemetry architecture and event truth

Frontend pages initialize PostHog directly with a shared public capture token and the US ingestion host. `window.analytics.track` catches capture exceptions. Upgrade uses `person_profiles: 'identified_only'`; success does not explicitly set that option. The hosted SDK is loaded from the ingestion host without a pinned SDK version, so transport/default-property behavior must be checked in actual ingestion rather than guessed.

`backend/telemetry.py` makes an optional `POST /capture/` with a 1.5-second timeout. `POSTHOG_CAPTURE_API_KEY` is required to attempt delivery; absence is a no-op. HTTP errors/exceptions return false; there is no durable queue or retry. Only allowlisted events and bounded scalar properties are forwarded. A personal query key is intentionally never used as a capture fallback.

| Event | Emitted from | Exact trigger | Client/server | Truth level |
|---|---|---|---|---|
| `paywall_viewed` | `frontend/app/upgrade.html:417` | Once when the main inline script evaluates, before asynchronous session lookup/identify | Client | Observational page exposure; not necessarily authenticated |
| `paywall_cta_clicked` | `upgrade.html:451–466` | Accepted checkout attempt after single-flight guard **and successful token lookup**, before create-order fetch | Client | Authenticated intent; not every physical CTA click |
| `checkout_order_created` | `backend/main.py:3744–3756` | JWT accepted, pending subscription insert returns data, signed ECPay checkout form successfully constructed | Server | Authoritative order creation; not payment |
| `subscription_activated` | `main.py:3991–4015` | Signed callback passes configuration/merchant/amount/owner/simulation checks and `accept_ecpay_payment` returns exactly `activated` | Server | Authoritative accepted payment/activation boundary |
| `checkout_success_viewed` | `frontend/app/success.html:286–300,500–503` | Success page boot, before subscription polling or auth resolution; tab/order storage guard applied | Client | Browser arrival only, even if later UI is pending/login/error |

`payment_return` (`main.py:4019–4045`) only redirects the browser, optionally carrying `MerchantTradeNo` as the `order` query parameter. It grants nothing. `subscription_activated` is not emitted by success.html. The backend telemetry allowlist contains `checkout_success_viewed`, but no backend call site emits it; an allowlist is not an emitter.

Coverage boundary: the legacy `/api/webhooks/lemonsqueezy` handler at `main.py:3235–3369` can update paid profile state and mirror a subscription without emitting these PostHog truth events. Its production usage is unknown. Manual grants/administrative extensions also must not be counted as paid conversions. This funnel covers the instrumented **ECPay checkout**, not every route to effective PRO.

The five-event instrumentation was introduced by repository commit `69ffdcb`. The relevant code exists in both current HEAD and origin/main. A commit timestamp is **not** an instrumentation deployment timestamp or first-ingestion timestamp.

## 3. Production capture and configuration evidence

### Fresh production frontend readback

| Public asset | HTTP | SHA-256 | Comparison |
|---|---:|---|---|
| `https://blabby.vercel.app/upgrade.html` | 200 | `81c55d34954a324e6fb38ea97db439acfabf8910b45e6c8cf6c459da7437932b` | Byte-identical to HEAD and origin/main |
| `https://blabby.vercel.app/success.html` | 200 | `e7b0ce2ea5926d662305e8569b08b78070bd23a5e3f8ff0dbbea70773ab03009` | Byte-identical to HEAD and origin/main |
| `https://blabby.vercel.app/config.js` | 200 | `4f05f5eabbe46427c67163d0965978daa269eb27221a3e3fd921b363e310a0f9` | Read for configuration context only |

Both relevant HTML files contain a capture token with the **same SHA-256 prefix `432393a1df46`**, and `api_host=https://us.i.posthog.com`. Token values are omitted. Upgrade contains `paywall_viewed` and `paywall_cta_clicked`; success contains `checkout_success_viewed`.

**Frontend = CAPTURE_CONFIGURED (served configuration only).** This proves production code has a capture route, not that SDK delivery succeeds or that PostHog has stored any events. No capture test was sent and no browser success-page visit was manufactured.

### Credential and deployment configuration audit

| Surface inspected | Result | Classification / limit |
|---|---|---|
| Current process, all POSTHOG-named variables | None | Query/capture variables absent locally |
| Canonical ignored env candidates | None | No local ignored PostHog configuration found |
| Existing legacy checkout `backend/.env`, read only | No personal key, project ID, host or capture key | No PostHog credential; unrelated values neither used nor recorded |
| `.zshenv`, `.zprofile`, `.zshrc` | No POSTHOG references | No shell-managed query configuration discovered |
| Existing Codex connector config | No PostHog/Render reference | No configured query connector or Render metadata client discovered |
| Render CLI/config and process/legacy env API credential candidates | No callable CLI/config/API credential discovered | **Backend production capture = BLOCKED**; not proof that Render lacks a key |
| Existing Vercel CLI authentication | Token file present; authenticated projects request returned HTTP 403 | Production env list unavailable through this credential |
| Existing Vercel connector | Team readable; project list empty; deployment lookup for production alias returned 404 | Does not prove the project/deployment is absent; public assets are separately verified |
| PostHog project settings / numerical project ID | No authenticated query/settings access | **BLOCKED**; never derive a project ID from a capture token |

Overall configuration is **PARTIAL**: frontend capture is configured; backend production capture and query access are blocked. No environment variable was changed. Backend capture key equality with frontend is **UNKNOWN**, not assumed. Backend code defaults to `https://us.i.posthog.com`; its deployed override and deployed SHA are unverified.

### Capture credentials versus query credentials

| Purpose | Configuration | Current truth |
|---|---|---|
| Browser ingestion | Hardcoded public capture token + `api_host` | Verified in production HTML |
| Server ingestion | `POSTHOG_CAPTURE_API_KEY`; optional `POSTHOG_HOST` defaults to US ingest | Local variable absent; production value unknown |
| Private query | `POSTHOG_PERSONAL_API_KEY`, `POSTHOG_PROJECT_ID`, query-process `POSTHOG_HOST` | Missing / unverified |

US public ingestion and private querying use different hosts: `https://us.i.posthog.com` and `https://us.posthog.com`. Capture tokens do not authorize private analytics reads. [Official API overview](https://posthog.com/docs/api)

**Do not overwrite the backend's ingestion `POSTHOG_HOST` with the query host.** The query recipe below runs in a separate operator process; the existing variable name has different context there.

## 4. Identity and attribution

### A. Anonymous paywall → authenticated checkout

On a fresh browser, PostHog supplies the anonymous identity. `paywall_viewed` fires before upgrade's boot `getToken()` calls `identify(session.user.id)`. Subsequent authenticated CTA attempts call the same identify path before capture. The signup/login flow in `index.html` also identifies using the Supabase user UUID, including `completeAnonymousAuthAttribution` at lines 3775–3787.

PostHog's JavaScript `identify` can link earlier anonymous activity to the identified user. That linkage still requires the same project, retained browser identity and successful SDK ingestion. We did **not** verify those person merges in production. The app's anonymous-trial visitor ID/marker is not proven to equal PostHog's anonymous distinct ID. A new alias call is not justified by static inspection. [Official identity behavior](https://posthog.com/docs/product-analytics/identify)

### B. Login/logout and session gaps

- Backend order and activation use `distinct_id=str(user_id)` from authenticated/validated ownership; frontend identify uses the same intended UUID.
- `success.html` defines an identify wrapper but never calls it. It relies on previously persisted browser identity. Fresh browser, lost storage or cross-device arrival can remain anonymous.
- `index.html:5270` and `account.html:517` sign out of Supabase without a PostHog reset call. Shared-device or account-switch attribution can be contaminated. This is a documented existing gap, not repaired by emailing/aliasing identities. Reset on logout is the documented PostHog pattern. [Official logout guidance](https://posthog.com/docs/product-analytics/identify#reset-on-logout)
- No explicit cross-server/browser session ID is included in payment events. SDK session properties, if present, cannot be assumed on server captures.
- Raw UUID shape alone cannot prove an ID is authenticated: anonymous SDK IDs may also look like UUIDs.

### C. Order → activation

Both server events carry the same technical `order_id`, whose value is the generated `merchant_trade_no`. Both use the validated owner's UUID as distinct_id. This supports an order join without email. The technical identifier is pseudonymous and must still be treated as restricted analytics data; no real identifiers are printed here.

Success arrival optionally carries a regex-validated `order_id`. The browser query string is untrusted and can be fabricated or absent; syntax validation does not prove an order exists or was paid. Match against captured orders/activations before classifying it.

### D. Person funnel and source attribution

A person funnel is structurally possible after same-project identity merging is proven. It is **not production-verified**. A standard person funnel can also join different orders belonging to one person; the authoritative order-to-activation KPI must additionally match `order_id`.

`source` appears only on view and CTA, not on order/activation. Order creation receives no source field. Therefore:

- View→CTA source breakdown can use the initiating view's normalized source.
- End-to-end source attribution is provisional, requiring a verified person/time chain to the chosen order; it is not a native property on activation.
- Cross-device loss, multiple paywall sources, retries and multiple orders prevent an unconditional last-touch/source claim.
- No paid source breakdown or new-vs-returning cohort is published until the join is validated.

## 5. Property audit

### Properties explicitly emitted today

| Event | Explicit properties |
|---|---|
| paywall_viewed | `source` |
| paywall_cta_clicked | `source` |
| checkout_order_created | `order_id`, `plan="monthly"`, `amount_twd=199`, `currency="TWD"`, `payment_provider="ecpay"`, `subscription_status="pending"` |
| subscription_activated | `order_id`, `amount_twd` from stored order, `currency="TWD"`, `payment_provider="ecpay"`, `subscription_status="active"` |
| checkout_success_viewed | Valid `order_id`, otherwise empty properties |

These are code-level properties, not a claim about actual ingested schema. Backend SAFE_PROPERTY_KEYS additionally permits `entitlement_active`, `failure_stage`, `http_status`, `source`, `stage`; current payment call sites do not emit them. `SAFE_EVENT_NAMES` does not include a payment-failure event. Default SDK metadata and person traits require query inspection.

| Property | Classification | Interpretation / gap |
|---|---|---|
| Event name, timestamp, canonical distinct_id | REQUIRED | Event storage fields; server payload does not explicitly send timestamp, so latency/order must be checked |
| Stable order_id on order/activation | REQUIRED | Present in code; analytical dedupe and authoritative join key |
| source | REQUIRED for source analysis; MISSING server-side | Client regex bounds shape/length but accepts unknown labels; not a strict enumeration |
| plan | USEFUL; MISSING on activation | Keep existing `monthly`; join from order if required, do not introduce `pro_monthly` as another live value |
| amount_twd / currency | REQUIRED for purchase-value analysis | Present server-side; don't rename to generic `amount` or infer client amounts; captured value is not net revenue/refund reconciliation |
| subscription_status | REQUIRED for truth checks | Server `pending`/`active`; browser arrival does not imply either |
| payment_provider | USEFUL | Present on server events; no need to invent browser provider properties |
| is_authenticated | USEFUL; MISSING explicitly | CTA semantically token-gated; don't infer all views are logged in |
| route/page | USEFUL | Page location known from emitter, but explicit property absent; SDK `$pathname` must be verified, not assumed |
| environment | REQUIRED for reliable production-only server analysis; MISSING | Shared hardcoded browser key can also receive local/staging traffic. Client hostname filters cannot filter backend events lacking hostname/environment. Stage simulated activations can call the same capture seam if its capture key is configured. |
| Provider rejection / checkout failure details | MISSING | No taxonomy additions now; order→activation drop-off alone cannot distinguish card failure, abandonment, unavailable backend or telemetry loss |
| Email, JWT, access/refresh tokens, ECPay secrets, raw callback/card data | REMOVE / prohibited | Absent from explicit payment-event properties; backend allowlist rejects unsupported keys. Do not query/export these values. |

Adjacent privacy findings: `account.html:400` currently identifies with an email trait; this is outside the five payment call sites but prevents claiming project-wide absence of email. Default URL/referrer/autocapture metadata was not inspected in PostHog and may carry unwanted query-string context. No such values are collected by the prepared aggregate queries. No general auth/analytics refactor is included here.

Existing source labels found at call sites/headline definitions include `direct`, `vocab_limit`, `writing_quota`, `writing_task1`, `reading_quota`, `feedback_quota`, `drill_quota`, `part2_quota`, `speaking_memory`, `history`, `account`. Group these into explicit known buckets plus `other`; never label every unknown input as organic/direct acquisition. For dashboard presentation only, writing_quota/writing_task1 may be grouped under Writing and reading_quota under Reading.

## 6. Duplicate and delivery semantics

These classifications describe **application code**, not a proven PostHog delivery guarantee.

| Event | Classification | Boundary / caveat |
|---|---|---|
| paywall_viewed | POSSIBLY_DUPLICATED | Each page/script load captures again; refresh/new tab/back-navigation can add exposures. No exposure ID. |
| paywall_cta_clicked | POSSIBLY_DUPLICATED | Single-flight prevents immediate double-clicks; failures re-enable retries. One authenticated attempt may be followed by more legitimate attempts. Anonymous physical clicks produce no CTA event. |
| checkout_order_created | AT-MOST-ONCE capture attempt per minted order | Each create-order retry intentionally mints a different trade number and pending order. Deduping by user would erase real attempts. Delivery can be lost. |
| subscription_activated | IDEMPOTENT_BY_KEY at activation boundary | Current RPC row lock, pending→active gate and ledger uniqueness return duplicate for repeat acceptance; duplicate callback exits before capture. At most one capture attempt for accepted order; no exactly-once analytics guarantee. |
| checkout_success_viewed | POSSIBLY_DUPLICATED globally | At-most-once per order per tab session when sessionStorage works. New tabs/sessions or blocked storage can repeat. Missing/invalid IDs share one `no-order` storage key. |

The activation RPC implementation at `supabase/migrations/20260730_ecpay_backend.sql:584–642` and duplicate-callback test were inspected read-only. This is not production schema readback or migration authorization.

**The larger known delivery risk is loss, not demonstrated callback inflation:** capture is after the activation transaction. If capture fails or the process exits after activation, a callback retry returns duplicate and does not emit again. There is no durable outbox/reconciliation replay. Likewise success-page storage is marked before capture; an unavailable SDK can suppress a later retry in that tab. Do not change payment acceptance/acknowledgement to fix analytics.

Analytical dedupe strategy:

- Core conversion counts unique resolved persons completing ordered prefixes, not raw view/click counts.
- Orders and activations use unique `(payment_provider, order_id)` after property/ownership consistency checks; repeated captured rows do not create extra payments.
- Success mismatch uses unique validated order IDs, retaining missing-ID/orphan arrivals as separate quality counts.
- Keep raw-event counts alongside deduped counts to detect inflation.
- No new `$insert_id`, event name, retry, alias or payment-flow change was made. A future telemetry replay would need explicit order/event dedupe and separate authorization, never re-running a callback or marking an unsigned browser visit paid.

## 7. Canonical funnel and calculation contract

**Core:** `paywall_viewed → paywall_cta_clicked → checkout_order_created → subscription_activated`

`checkout_success_viewed` is an optional observational companion, never step 5 of paid conversion.

Default report: last 30 days as of a recorded UTC cutoff T; display daily charts in Asia/Taipei. Start later only if a verified instrumentation deployment timestamp warrants it. First_seen in a bounded query is only first observed **within that window**. No deployment start is inferred from a Git commit or successful HTML fetch.

Configuration: ordered steps, unique resolved persons, conversion window **7 days from paywall view**; intervening events allowed. Use a coherent view→CTA→order→activation chain, with the same order ID for the last two steps. A person's earliest qualifying view that has a valid chain is the attribution anchor. Use its normalized source for a breakdown. A plain PostHog person-only funnel is provisional until order pairing and production identity checks pass.

Let N1…N4 be unique people completing each ordered prefix within that window, not independent event uniques:

| KPI | Definition |
|---|---|
| Paywall→CTA CTR | N2 / N1; label **authenticated checkout intent rate**, because logged-out button clicks do not emit this event |
| CTA→Order conversion | N3 / N2 |
| Order→Activated conversion, person funnel | N4 / N3 |
| Paywall→Activated overall | N4 / N1 |
| Order→Activated conversion, order diagnostic | Unique created orders with a matching activation / unique created orders in the same cohort; don't mix this denominator into person KPIs |
| Largest drop-off | Compare Ni − Ni+1 and 1 − Ni+1/Ni only on the same mature cohort; report both absolute loss and rate |

Zero denominators display `N/A`, not 0%. Distinguish still-open windows from drop-off. For decisions at cutoff T, use matured entries before T−7 days; keep the last seven days visible as provisional/pending. For small cohorts, show counts and **INSUFFICIENT SAMPLE**; no ranked conversion insight with 1–2 users. Proposed display rule: suppress comparative drop-off conclusions when a relevant denominator is below 30 or any capture/identity gate remains unresolved. This is a reporting guard, not a statistical significance claim.

Mismatch: compare distinct order sets, including both/activated-only/arrival-only. Arrival can precede callback; allow the same seven-day order window. `success_viewed > activated` may reflect unauthenticated/fabricated arrival, unfinished processing, duplicate delivery or missing activation telemetry. `activated > success_viewed` may reflect a successful callback without browser return, storage loss or blocked client analytics. Neither difference alone proves a payment failure or success. Isolate unknown/missing order IDs and mismatched owners.

## 8. Dashboard specification — Blabby Revenue Funnel

Specification only: no PostHog dashboard/insight/project was created or modified, and no dashboard was added to Blabby Admin.

| Panel | Definition | Gate / breakdown |
|---|---|---|
| A. Core Funnel | Four ordered primary events; unique persons; 7-day window | Same-project person merges + order pairing; production traffic isolation required |
| B. Conversion KPIs | N2/N1, N3/N2, N4/N3, N4/N1 with numerator/denominator and pending cohort | No independent event-count ratios presented as conversion |
| C. Daily Activations | Distinct provider/order activations per Taipei day, first observed activation per order | Optional deduped amount_twd sum labelled captured activation value, not MRR/net revenue |
| D. Arrival / Activation Mismatch | Both, activated-only, arrival-only; missing/unmatched order IDs separately | Unique order sets; browser arrival never substitutes for activation |
| E. Source Breakdown | Direct, Vocabulary limit, Writing, Reading, other observed known sources, other | View→CTA initially; paid attribution withheld until person/time/order chain verified |
| F. Data Quality | Raw vs deduped event count, property presence, identity overlap, cross-owner order anomalies, ingest first/last seen | Aggregate values only, no user/order lists or sensitive property values |
| G. New vs Returning (optional) | Person's first observed relevant activity before/inside cohort start | **DISABLED** pending adequate historical coverage, resolved identity and production isolation; no invented cohort |

## 9. Real metrics and query status

| Metric / step | Count | Conversion |
|---|---|---|
| paywall_viewed | UNKNOWN | N/A |
| paywall_cta_clicked | UNKNOWN | UNKNOWN |
| checkout_order_created | UNKNOWN | UNKNOWN |
| subscription_activated | UNKNOWN | UNKNOWN |
| checkout_success_viewed, observational | UNKNOWN | Not a paid conversion step |
| Largest drop-off | UNKNOWN | No authenticated readback |

**NO REAL METRICS — QUERY ACCESS BLOCKED.** Sample size is unknown, not zero. No PostHog query was submitted with the capture token or an invented project ID. No manual PostHog browser scraping was attempted.

## 10. Exact query credential requirement

**POSTHOG QUERY ACCESS REQUIRED**

Provide through an approved local secret/configuration mechanism, **not in chat, frontend or Git**:

- `POSTHOG_PERSONAL_API_KEY`: existing authorized personal key scoped to the Blabby project.
- `POSTHOG_PROJECT_ID`: confirmed numerical Blabby project ID; verify it is the project receiving the production frontend capture token fingerprint above.
- `POSTHOG_HOST=https://us.posthog.com`: private query host in an isolated operator process. If project ownership establishes a different region/host, stop and reconcile; do not guess from an ID.

Minimum permission for the prepared query endpoint is **`query:read`**, restricted to the intended project. No admin/write/insight-write/dashboard-write scope is requested. [Official query endpoint and scope](https://posthog.com/docs/api/query)

Separate prerequisite for a true production claim: authenticated Render configuration/source evidence showing a capture key is present, which project it targets, the effective ingest host, and whether staging shares that key. Vercel configuration access alone cannot establish backend capture.

## 11. Prepared read-only query sequence

**NOT EXECUTED; no claim of live HogQL validation.** These queries return aggregates only. First validate project/timeframe and schema; reconcile errors without requesting broader privileges or dumping event/property values. If production traffic cannot be isolated, label results mixed/unknown-environment and withhold production revenue KPIs.

### Q1 — Existence, counts, distinct IDs/persons, bounded first/last seen

```sql
SELECT event,
       count() AS event_count,
       count(DISTINCT distinct_id) AS distinct_ids,
       count(DISTINCT person_id) AS resolved_person_ids,
       min(timestamp) AS first_seen_in_window,
       max(timestamp) AS last_seen_in_window
FROM events
WHERE timestamp >= now() - INTERVAL 30 DAY
  AND timestamp < now()
  AND event IN ('paywall_viewed', 'paywall_cta_clicked',
                'checkout_order_created', 'subscription_activated',
                'checkout_success_viewed')
GROUP BY event
ORDER BY event
```

Do not equate resolved person count with validated authenticated user count. An absent event becomes zero only after a successful query in a verified complete window, not while access is blocked.

### Query transport recipe

Run in a separate trusted shell after secure environment configuration; no credentials are printed or placed in shell arguments. This recipe uses Q1, and can submit each SELECT below by replacing only `query`.

```python
import json, os, requests

key = os.environ.get('POSTHOG_PERSONAL_API_KEY', '').strip()
project = os.environ.get('POSTHOG_PROJECT_ID', '').strip()
host = os.environ.get('POSTHOG_HOST', '').rstrip('/')
if not key or key.startswith('phc_') or not project.isdigit():
    raise SystemExit('POSTHOG QUERY ACCESS REQUIRED')
if host != 'https://us.posthog.com':
    raise SystemExit('Confirm the private project host before querying')
query = """
SELECT event, count() AS event_count,
       count(DISTINCT distinct_id) AS distinct_ids,
       count(DISTINCT person_id) AS resolved_person_ids,
       min(timestamp) AS first_seen_in_window,
       max(timestamp) AS last_seen_in_window
FROM events
WHERE timestamp >= now() - INTERVAL 30 DAY AND timestamp < now()
  AND event IN ('paywall_viewed','paywall_cta_clicked',
                'checkout_order_created','subscription_activated',
                'checkout_success_viewed')
GROUP BY event ORDER BY event
"""
r = requests.post(
    f'{host}/api/projects/{project}/query/',
    headers={'Authorization': 'Bearer ' + key},
    json={'query': {'kind': 'HogQLQuery', 'query': query}},
    timeout=30, allow_redirects=False,
)
if r.status_code != 200:
    raise SystemExit(f'Query blocked/failed: HTTP {r.status_code}')
body = r.json()
if 'results' not in body:
    raise SystemExit('Query incomplete; inspect sanitized query status')
print(json.dumps({'columns': body.get('columns'), 'results': body['results']}))
```

Before publishing, freeze the same UTC start/end cutoff for every query. SQL query execution uses a POST transport but performs a data read; do not call capture, person modification, insight creation or dashboard mutation endpoints. [Official HogQL query transport](https://posthog.com/docs/api/query)

### Q2 — Ingested property key schema, no property values

```sql
SELECT event, arrayJoin(JSONExtractKeys(properties)) AS property_key,
       count() AS rows_with_key
FROM events
WHERE timestamp >= now() - INTERVAL 30 DAY AND timestamp < now()
  AND event IN ('paywall_viewed','paywall_cta_clicked',
                'checkout_order_created','subscription_activated',
                'checkout_success_viewed')
GROUP BY event, property_key
ORDER BY event, rows_with_key DESC
LIMIT 250
```

This is a bounded schema sample, not proof of absent properties if the limit is reached. Inspect presence of order/source/environment/path/SDK identity fields without returning email, URLs, tokens, callback data or person traits.

### Q3 — Backend/client distinct-ID and person overlap

```sql
SELECT count() AS ids_seen,
       countIf(has_client > 0) AS ids_with_client_events,
       countIf(has_server > 0) AS ids_with_server_events,
       countIf(has_client > 0 AND has_server > 0) AS shared_ids,
       countIf(person_count > 1) AS ids_with_multiple_person_ids
FROM (
    SELECT distinct_id,
           countIf(event IN ('paywall_viewed','paywall_cta_clicked',
                             'checkout_success_viewed')) AS has_client,
           countIf(event IN ('checkout_order_created',
                             'subscription_activated')) AS has_server,
           count(DISTINCT person_id) AS person_count
    FROM events
    WHERE timestamp >= now() - INTERVAL 30 DAY AND timestamp < now()
      AND event IN ('paywall_viewed','paywall_cta_clicked',
                    'checkout_order_created','subscription_activated',
                    'checkout_success_viewed')
    GROUP BY distinct_id
)
```

Also run the same client/server overlap grouped internally by person_id: merged anonymous/client IDs may differ while belonging to one person. Return only the resulting aggregate counts. Neither UUID shape nor some shared IDs proves all merges correct; cross-device and missing-reset cases remain relevant.

### Q4 — Order dedupe, owner mismatch and browser-arrival quality

```sql
SELECT count() AS observed_order_keys,
       countIf(created > 0) AS created_orders,
       countIf(activated > 0) AS activated_order_keys,
       countIf(created > 0 AND activated > 0) AS matched_created_activated,
       countIf(activated > 0 AND arrived > 0) AS activated_and_arrived,
       countIf(activated > 0 AND arrived = 0) AS activated_without_arrival,
       countIf(arrived > 0 AND activated = 0) AS arrival_without_activation,
       countIf(created > 1) AS repeated_creation_rows,
       countIf(activated > 1) AS repeated_activation_rows,
       countIf(server_owners > 1) AS cross_owner_order_keys
FROM (
    SELECT properties.order_id AS order_key,
           countIf(event = 'checkout_order_created') AS created,
           countIf(event = 'subscription_activated') AS activated,
           countIf(event = 'checkout_success_viewed') AS arrived,
           uniqIf(distinct_id, event IN
                  ('checkout_order_created','subscription_activated')) AS server_owners
    FROM events
    WHERE timestamp >= now() - INTERVAL 30 DAY AND timestamp < now()
      AND event IN ('checkout_order_created','subscription_activated',
                    'checkout_success_viewed')
      AND match(toString(properties.order_id), '^[A-Za-z0-9_-]{1,64}$')
    GROUP BY order_key
)
```

Q4 is a quality diagnostic, **not yet a cohort conversion rate**: window-edge orders and arrivals can have counterpart events outside the window. Before publishing panel D, anchor on created orders, look up counterparts through the seven-day window, verify one server owner/provider per order and separately count missing/invalid IDs. Exclude anomalies from canonical conversion rather than guessing a paid result. The provider on the currently instrumented server events is ECPay; the legacy LemonSqueezy path is not instrumented by these events. Partition by provider if future observed schema requires it without changing event taxonomy.

### Q5 — Daily deduped activation trend

```sql
SELECT toDate(toTimeZone(first_activation, 'Asia/Taipei')) AS day,
       count() AS activated_orders
FROM (
    SELECT properties.payment_provider AS provider,
           properties.order_id AS order_key,
           min(timestamp) AS first_activation
    FROM events
    WHERE timestamp >= now() - INTERVAL 30 DAY AND timestamp < now()
      AND event = 'subscription_activated'
      AND properties.payment_provider = 'ecpay'
      AND match(toString(properties.order_id), '^[A-Za-z0-9_-]{1,64}$')
    GROUP BY provider, order_key
)
GROUP BY day ORDER BY day
```

### Q6 — Source breakdown, bounded to known labels

```sql
SELECT event,
       if(properties.source IN
          ('direct','vocab_limit','writing_quota','writing_task1',
           'reading_quota','feedback_quota','drill_quota','part2_quota',
           'speaking_memory','history','account'),
          toString(properties.source), 'other') AS source_bucket,
       count() AS event_count,
       count(DISTINCT person_id) AS observed_person_ids
FROM events
WHERE timestamp >= now() - INTERVAL 30 DAY AND timestamp < now()
  AND event IN ('paywall_viewed','paywall_cta_clicked')
GROUP BY event, source_bucket ORDER BY event, event_count DESC
```

Raw counts in Q6 are not CTR. For panels A/B/E, apply the ordered-chain definitions in section 7 after the identity, order and environment gates pass. Keep event-existence diagnostics and actual sequential conversion calculations separate.

## 12. Gaps and next evidence required

1. **Query access:** missing personal read key + confirmed project ID. Actual event existence/counts/schema/person merges cannot be answered.
2. **Backend production capture:** no authenticated Render configuration/deployed-source evidence. Capture absence locally does not prove absence in production.
3. **Environment contamination:** no explicit environment on server truth events. Stage/local capture using the same project would prevent production-only revenue claims.
4. **Identity:** success lacks identify; logout lacks reset; browser persistence/cross-device linkage is unverified. No email-based workaround.
5. **Attribution:** source stays on frontend; backend order/activation lack it. Logged-out CTA clicks are not counted as intent.
6. **Delivery loss:** backend no-op/failure has no durable replay; duplicate callback correctly avoids reactivation and cannot recover missing analytics by itself.
7. **Property/privacy:** activation lacks plan; explicit auth/page/environment properties absent; existing account email trait and default SDK URL metadata need a separately scoped privacy review.
8. **Coverage and financial truth:** the legacy LemonSqueezy activation path is outside this instrumentation. Captures are not a payment ledger completeness proof. No refund/settlement/net-revenue inference; no direct production DB reconciliation in this round.

No telemetry fix was made merely to produce a code commit. Existing capture triggers and callback dedupe are covered by tests; production configuration/ingestion gaps require evidence first. No taxonomy, payment logic, ECPay validation or activation transaction was modified.

## 13. Validation and production safety

- Re-read telemetry module, frontend emitters/auth paths, create-order, callback/RPC, return navigation, pricing constant and existing telemetry/billing tests.
- Public frontend artifact comparison and sanitized credential-presence inspection completed.
- Existing focused telemetry/ECPay/billing pytest suite: **189 passed, 5 existing deprecation warnings**, 2.39 seconds, using Python 3.11 and Node 20.20.2. The billing pytest wrapper executes its Node behavior harness. No live provider integration ran.

  Command: `python -m pytest backend/tests/test_payment_funnel_telemetry.py backend/tests/test_ecpay.py backend/tests/test_frontend_billing_contracts.py -q`
- No full pytest required for a documentation-only change.
- Prepared HogQL queries were **not executed or certified against the live project**.
- No PostHog project/insight/dashboard/person updates, capture probes, production checkout, callback replay or manual analytics scraping.

PRODUCTION DB WRITE: 0

PRODUCTION MIGRATION APPLY: 0

PRODUCTION DEPLOYMENT: 0

PUSH: NO

`20260905040134_atomic_vocabulary_save_quota.sql` remains **PENDING_BLABBY / LOCAL ONLY**. Current backend still depends on it. Do not deploy current backend independently, apply pending migrations, implement vocabulary OPTION D or change the protected production rollout documents.
