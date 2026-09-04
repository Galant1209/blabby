# Blabby Full-Site Production Safety Audit — 2026-07-14

Audit timestamp: 2026-07-14 10:56–11:10 Asia/Taipei  
Method: read-only code/migration review, bounded local tests, safe production GET/OPTIONS/auth-negative probes  
Production mutations: one unauthenticated empty POST to `/api/debug/rec-log` during verification may have created one all-null telemetry row; no user data or secret was sent. No other production mutation was performed.

## 1. Executive Summary

- Production frontend and backend responded successfully; Vercel reported the production deployment `Ready`.
- Verdict: **NOT SAFE FOR PUBLIC USE** until the P1 items below are contained.
- P0: 0. P1: 4. P2: 8. P3: 3.
- Largest security risk: production DOM XSS through unescaped transcript/AI fields and unsanitized AI-generated SVG.
- Largest data risk: migrations omit RLS for `subscriptions` and `diagnosis_cache`; live exposure requires immediate Supabase verification.
- Largest speaking-flow risk: Part 2 reads an unbounded upload into memory and trusts its filename extension.
- Largest product-integrity risk: `reading_questions` deliberately disables RLS while storing correct answers and explanations.
- CORS correctly rejects localhost, malicious, and null origins; sampled protected APIs reject missing JWTs.
- Full tests are not currently reproducible because three modules fail during collection on an invalid dummy Supabase key.
- No physical-device, valid-user, cross-user, live-schema, or destructive test was performed.

## 2. Environment Baseline

```text
Current branch: main (tracking origin/main)
HEAD: 071695ba19a4f946200f7e2f2816a6bc2cbff87f
Working tree before audit: modified README.md; untracked frontend/app/hub.html
Working tree created by audit: this report only
Remote: origin = https://github.com/Galant1209/blabby.git
Frontend deployment: https://blabby.vercel.app — Ready; deployment dpl_GkD1AfHQMuXdw7q3n7JkU6ZhdC6n; created 2026-07-07 13:54 +08
Backend deployment: https://blabby-backend.onrender.com — /health returned 200
Database: Supabase project ref mkwywkwruyqzdhuzwnoa; live settings/schema not verified
Audit timestamp: 2026-07-14 (Asia/Taipei)
```

Runtime observed locally:

- Python 3.10.0
- Node 25.9.0
- Vercel CLI 52.2.1
- Backend: FastAPI 0.135.2, Starlette 1.0.0, Uvicorn 0.42.0
- Python packages are mostly pinned, but `anthropic>=0.25.0` and `apscheduler>=3.10.0` are not reproducibly pinned.
- No JavaScript package/lock file exists; frontend loads Supabase JS from jsDelivr using floating `@2`.
- `backend/.env` exists locally and is ignored. Only variable names were inspected; values were not printed. It is not tracked at HEAD.

Deployment evidence:

- Vercel production alias points to `blabby-7obi3q50z-galants-projects.vercel.app` and status was `Ready`.
- Production `index.html` SHA-256 exactly matched local `frontend/app/index.html`: `d1f49705...7341cb5`.
- Frontend `/` returned 200; `robots.txt`, `manifest.json`, and a random missing path returned 404.
- Backend `/health` returned only status and timestamp. It does not validate Supabase or AI providers.
- Render dashboard commit, build log, worker count, memory, start command, and environment scope: **NOT VERIFIED**.

## 3. Test Coverage

### Actually verified

- Production frontend `/`, missing asset behavior, response headers, and exact deployed HTML match.
- Production backend `/health` response.
- CORS preflight for production, localhost, malicious, and null origins.
- Missing-auth behavior for `/process`, `/api/history`, `/api/admin/subscriptions`, `/admin/recent`, and `/reading/history`.
- Public unauthenticated behavior of `/api/debug/rec-log`.
- Static route/auth/rate-limit/service-role ownership review.
- Static Supabase migrations/RLS/SECURITY DEFINER review.
- Part 1/Part 2 MediaRecorder, upload, temp-file, timeout, rendering, and persistence paths.
- Reading validators: 22 passed.

### Confirmed from code only

- JWT validation delegates to `supabase_admin.auth.get_user(token)` and returns its user ID.
- Admin role is checked server-side against `ADMIN_EMAILS` after token validation.
- Most user-owned service-role queries constrain `user_id` to the JWT-derived subject.
- Part 1 has a 25 MiB size bound and temp-file cleanup; Part 2 does not have equivalent bounds.
- AI correction shape validation exists for Part 1; Part 2 accepts a generic JSON object without field/type/range validation.

### Not verified

- Live Supabase Auth settings, JWT expiry/audience configuration, OAuth URLs, RLS state, grants, storage, constraints, indexes, orphan/duplicate records, and migration drift.
- Valid, expired, disabled, banned, or deleted user tokens.
- Cross-user IDOR using two controlled test accounts.
- Render deployment/dashboard/logs and Vercel build warnings/env separation/old preview behavior.
- Provider timeouts/401/403/429/500 and network disconnects.
- Python CVEs: `pip-audit` failed/was bounded and is **NOT VERIFIED**.
- Real Part 1/Part 2 provider E2E and production persistence.

### Physical devices

```text
Not verified on physical device
```

Chrome/Safari/Firefox desktop, iPhone Safari, Android Chrome, Bluetooth microphone, calls/app switching, screen lock, and real noisy/silent audio were not physically tested.

## 4. Findings

### BLABBY-001

```text
ID: BLABBY-001
Severity: P1
Area: Frontend / XSS
Status: Confirmed
Evidence: frontend/app/index.html:4832-4899 inserts transcript and multiple AI fields into innerHTML without escaping. Production contains the same lines and exactly matches the local file hash.
Reproduction: With a controlled mocked response, return an inert marker containing HTML in transcript, criteria name/description, strengths, or improvements and observe that it is parsed as DOM rather than displayed as text.
Expected: Transcript and AI strings are rendered as text or escaped before template insertion.
Actual: Untrusted values are interpolated directly into HTML.
Impact: Script-capable markup could execute in an authenticated user's origin, exposing session-accessible data and issuing authenticated API calls.
Root cause: Mixed safe/unsafe template rendering; only notes and notes_analysis use escapeHtml.
Recommended fix: Escape every dynamic string or build nodes with textContent; add a strict CSP as defense in depth.
Regression test: Feed all Part 2 response fields `<img src=x onerror=...>` and assert no element/event handler is created.
Affected files: frontend/app/index.html
```

### BLABBY-002

```text
ID: BLABBY-002
Severity: P1
Area: Frontend / AI-generated SVG
Status: Confirmed
Evidence: frontend/app/writing.html:534 assigns data.chart_svg directly to innerHTML. Backend prompt requests raw SVG but performs no sanitizer pass.
Reproduction: Mock chart_svg with an inert disallowed SVG element/attribute and inspect the resulting DOM.
Expected: SVG passes a strict allowlist sanitizer or is rendered as a non-scriptable image.
Actual: Arbitrary provider output becomes same-origin DOM.
Impact: A compromised/misbehaving model response can become XSS.
Root cause: Prompt constraints are treated as a security boundary.
Recommended fix: Sanitize server-side and client-side with an SVG-specific allowlist; reject scripts, event handlers, foreignObject, external URLs, animation, and unsafe CSS.
Regression test: Corpus of malicious SVG payloads must be rejected or rendered inert.
Affected files: backend/main.py, frontend/app/writing.html
```

### BLABBY-003

```text
ID: BLABBY-003
Severity: P1
Area: Part 2 upload / availability and cost abuse
Status: Confirmed
Evidence: backend/main.py:5966 reads the complete file before any size check; 5968-5971 derives decoder suffix from the attacker-controlled filename. No magic-byte, MIME, duration, or maximum-size validation exists.
Reproduction: Code inspection; no large production upload was sent.
Expected: Request/body limit, bounded streaming read, format allowlist plus content sniffing, duration limit, provider timeout, and concurrency cap.
Actual: Authenticated users can submit arbitrary-size/non-audio content and consume memory/provider resources.
Impact: Worker exhaustion, cost abuse, slow requests, and reduced availability.
Root cause: Part 2 did not inherit Part 1's 25 MiB guard and has no content validation.
Recommended fix: Enforce a smaller product-appropriate body limit before buffering; validate container and duration; reject unknown extensions/types; add per-user and concurrency controls.
Regression test: 0-byte, truncated, HTML/ZIP-as-webm, over-limit, long-duration, and parallel uploads fail before provider calls.
Affected files: backend/main.py
```

### BLABBY-004

```text
ID: BLABBY-004
Severity: P1
Area: Supabase RLS / Reading integrity
Status: Probable (subscriptions/diagnosis exposure); Confirmed by migration design (Reading answer exposure)
Evidence: 20260508_subscriptions.sql and 20260508_diagnosis_cache.sql create user-owned tables without enabling RLS or revoking client roles. 20260518_reading_module.sql explicitly disables RLS on reading_questions, which stores correct_answer, explanation, and evidence_quote.
Reproduction: Do not probe live user data. In a staging clone, query these tables with anon and authenticated clients before completing an attempt.
Expected: All user-owned tables have FORCE/ENABLE RLS with owner policies; pre-submit clients cannot read answer keys.
Actual: Migration source has no RLS for two user-owned tables and intentionally exposes the full Reading question rows.
Impact: Possible billing/diagnosis privacy breach and trivial Reading answer-key retrieval.
Root cause: Service-role-only application usage was assumed to replace database authorization; shared content and secret answer data share one table/grant.
Recommended fix: Immediately verify live grants; enable RLS on subscriptions/diagnosis_cache. Split public question prompts/options from protected answers or expose answers only through a post-submit RPC.
Regression test: anon and user A cannot read billing/diagnosis rows; pre-submit user cannot select answers; submitted owner can retrieve reveal only through scoped API.
Affected files: supabase/migrations/20260508_subscriptions.sql, 20260508_diagnosis_cache.sql, 20260518_reading_module.sql
```

### BLABBY-005

```text
ID: BLABBY-005
Severity: P2
Area: Rate limiting / quota races
Status: Confirmed
Evidence: backend/main.py:89 uses get_remote_address globally. Monthly/drill/Reading quota patterns count then later insert without an atomic database reservation.
Reproduction: Static review; no high-volume production test performed.
Expected: Per-user high-cost limits, per-IP fallback, atomic quota reservation, concurrency gate, and idempotency key.
Actual: NAT users share limits, one user can rotate IP, and parallel requests can pass the same pre-insert count.
Impact: Cost abuse, accidental duplicate charges/records, and denial of service to shared-IP users.
Root cause: SlowAPI IP limiter plus non-transactional application quota checks.
Recommended fix: JWT-subject rate key where authenticated, IP fallback, database atomic consume RPC, and in-flight lock.
Regression test: N parallel requests at the final remaining quota produce exactly one provider call/write.
Affected files: backend/main.py, new migration likely required
```

### BLABBY-006

```text
ID: BLABBY-006
Severity: P2
Area: Telemetry / unauthenticated write
Status: Confirmed in production
Evidence: POST /api/debug/rec-log returned 200 without Authorization and writes payload fields through the service-role client at backend/main.py:5911-5943.
Reproduction: An empty unauthenticated POST returned {"ok":true}.
Expected: Removed from production or authenticated, tightly validated, sampled, and privacy-minimized.
Actual: Any site/client can create telemetry rows and submit arbitrary UA/error strings up to logger truncation behavior.
Impact: Database/log pollution, storage abuse, misleading diagnostics, and potential collection of attacker-controlled personal content.
Root cause: Temporary debug endpoint remained public.
Recommended fix: Disable in production or require a user token and strict schema/length limits.
Regression test: Missing token returns 401 and unexpected/oversized fields return 422.
Affected files: backend/main.py, frontend/app/index.html
```

### BLABBY-007

```text
ID: BLABBY-007
Severity: P2
Area: Data integrity / IDOR reference
Status: Confirmed from code
Evidence: backend/main.py:1876-1886 validates retry_of as UUID only; it is inserted as a foreign key at 2408 without checking that the referenced practice record belongs to the JWT subject.
Reproduction: In staging with users A/B, submit as A with retry_of set to B's known record UUID.
Expected: Referenced record lookup includes id and user_id=current subject, otherwise 404.
Actual: Any existing UUID satisfying the FK may be linked cross-user.
Impact: Cross-user graph corruption and misleading progression/analytics; direct data disclosure was not found in this path.
Root cause: FK existence was treated as authorization.
Recommended fix: Owner-scoped lookup before insert and preferably a composite ownership constraint/RPC.
Regression test: A cannot link to B's record; unknown and foreign IDs both return 404 without provider usage.
Affected files: backend/main.py
```

### BLABBY-008

```text
ID: BLABBY-008
Severity: P2
Area: AI feedback contract / Part 2
Status: Confirmed
Evidence: run_claude checks only that JSON parses to a dict. /part2/evaluate spreads result into the response and persists it without validating required fields, bands, arrays, language, size, or extra fields.
Reproduction: Mock run_claude to return {}, wrong types, huge strings, invalid bands, or HTML.
Expected: Strict schema with bounded fields, enums, 0.5 band increments, criterion names/count, and safe fallback.
Actual: Structurally invalid but parseable JSON is accepted and stored/rendered.
Impact: UI breakage, unsafe rendering amplification, misleading scores, and malformed permanent records.
Root cause: Prompt-only contract.
Recommended fix: Pydantic strict response model and bounded repair/fallback before persistence.
Regression test: non-JSON, fences, missing/null/extra/wrong-type/oversized fields all fail safely without duplicate usage/write.
Affected files: backend/main.py, frontend/app/index.html
```

### BLABBY-009

```text
ID: BLABBY-009
Severity: P2
Area: Security headers
Status: Confirmed in production
Evidence: Frontend response had HSTS but no CSP, X-Content-Type-Options, Referrer-Policy, Permissions-Policy, frame-ancestors, or X-Frame-Options. Access-Control-Allow-Origin was * on static HTML.
Reproduction: curl -D - https://blabby.vercel.app/
Expected: Enforced CSP compatible with the app, nosniff, restrictive referrer/permissions, and frame protection.
Actual: Defense-in-depth headers are absent.
Impact: Increases exploitability of XSS/clickjacking and unnecessary browser capability/referrer exposure.
Root cause: No Vercel header configuration was found.
Recommended fix: Add tested headers; inline JS/CSS requires nonce/hash migration or a carefully staged CSP.
Regression test: Automated header assertions against preview and production.
Affected files: new vercel.json or deployment configuration
```

### BLABBY-010

```text
ID: BLABBY-010
Severity: P2
Area: Tests / release safety
Status: Confirmed
Evidence: Full pytest collection failed in test_process_question_guard.py, test_progress.py, and test_rec_log.py because create_client rejects the tests' invalid Supabase key. Pure validators passed 22/22.
Reproduction: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/pytest tests -v -p anyio
Expected: Unit suite collects/runs offline with clients injected or correctly mocked.
Actual: 3 collection errors prevent the full suite.
Impact: Auth, ownership, recording, and regression failures can ship undetected.
Root cause: Service clients are created at module import and tests depend on dummy key parsing behavior.
Recommended fix: Dependency-inject/lazily create clients or monkeypatch create_client before import; add focused security tests.
Regression test: The complete documented command passes without network credentials.
Affected files: backend/main.py, backend/tests/test_process_question_guard.py, backend/tests/test_progress.py, backend/tests/test_rec_log.py
```

### BLABBY-011

```text
ID: BLABBY-011
Severity: P2
Area: Error handling / logging
Status: Confirmed from code
Evidence: Groq failure logs include filename, upstream body, exception text, and user ID; /api/diagnosis/me returns `Diagnosis failed: {str(exc)}`. Validation handler returns internal Pydantic error details.
Reproduction: Mock upstream errors containing internal/provider details.
Expected: Stable public error code/request ID; detailed but redacted server log.
Actual: Some internal/upstream strings can reach logs or client responses.
Impact: Sensitive metadata/provider payload exposure and inconsistent client behavior.
Root cause: Raw exception interpolation and provider-body logging.
Recommended fix: Central exception mapping, request IDs, structured redaction, and no upstream body/token/transcript logging.
Regression test: Secret canaries in exceptions never appear in response or captured logs.
Affected files: backend/main.py
```

### BLABBY-012

```text
ID: BLABBY-012
Severity: P2
Area: Reliability / persistence and idempotency
Status: Confirmed from code
Evidence: /part2/evaluate persists after provider calls but has no idempotency key; _persist_part2 swallows DB exceptions and the endpoint still returns success. Retry can create duplicates and repeat provider cost.
Reproduction: Mock DB insert failure/response disconnect, then retry the same blob.
Expected: Client request ID with unique constraint; same request returns prior result; persistence failure is explicitly represented/retriable without rescoring.
Actual: Success can be shown without storage and retry can duplicate work/records.
Impact: Lost history, duplicate records/cost, and user/backend state divergence.
Root cause: Best-effort persistence with no idempotent transaction boundary.
Recommended fix: Required idempotency key, unique `(user_id,key)`, stored processing state, and deterministic retry response.
Regression test: repeated/concurrent identical key produces one provider execution and one record.
Affected files: backend/main.py, frontend/app/index.html, new migration
```

### BLABBY-013

```text
ID: BLABBY-013
Severity: P3
Area: Supply chain
Status: Confirmed
Evidence: Multiple pages load https://cdn.jsdelivr.net/npm/@supabase/supabase-js@2 without an exact version or SRI.
Reproduction: Inspect script tags.
Expected: Exact reviewed version with SRI/crossorigin or a self-hosted pinned asset.
Actual: Floating major-version CDN dependency executes with page privileges.
Impact: Non-reproducible frontend and increased CDN/supply-chain blast radius.
Root cause: Convenience CDN tag.
Recommended fix: Pin exact version and integrity hash, then regression-test auth flows.
Regression test: CI asserts exact URL and integrity attribute.
Affected files: frontend/app/*.html
```

### BLABBY-014

```text
ID: BLABBY-014
Severity: P3
Area: Health/observability
Status: Confirmed
Evidence: /health returns status/timestamp only and does not test required dependencies; no request ID was observed in API error contract.
Reproduction: Review /health and response.
Expected: Separate liveness and bounded readiness checks; correlation ID on errors/logs.
Actual: Healthy response can coexist with unusable Auth/DB/providers.
Impact: False-positive monitoring and slower incident diagnosis.
Root cause: Minimal liveness endpoint used as general health signal.
Recommended fix: Keep liveness minimal; add authenticated/internal readiness with bounded dependency checks and request IDs.
Regression test: Dependency failure leaves liveness up but readiness down without revealing configuration.
Affected files: backend/main.py, Render health-check configuration
```

### BLABBY-015

```text
ID: BLABBY-015
Severity: P3
Area: Static assets / discoverability
Status: Confirmed in production
Evidence: /robots.txt and /manifest.json return Vercel 404; random 404 is platform plain text.
Reproduction: GET the three paths.
Expected: Explicit robots policy; manifest only if PWA is intended; branded safe 404.
Actual: Assets are absent.
Impact: Low-risk SEO/installability/UX gap.
Root cause: Static deployment has no corresponding files/routes.
Recommended fix: Add only the assets that product requirements need.
Regression test: Asset status/content assertions.
Affected files: frontend/app (new static assets if desired)
```

## 5. Endpoint Matrix

Legend: `U` valid user JWT; `A` admin JWT/email allowlist; `—` public. `IP` means only IP rate limiting. Input/error entries summarize code review, not full fuzzing.

| Method/path | Auth/role | Limit | Input validation | Success / error notes |
|---|---:|---:|---|---|
| POST `/process` | U | 5/min IP | Partial: 25 MiB, UUID, mode/tag; no magic/duration | Structured feedback; 4xx/5xx, temp cleanup |
| POST `/api/webhooks/lemonsqueezy` | signature | none | HMAC path reviewed | Billing mutation; live webhook not tested |
| GET/POST `/api/covenant/status`, `/sign` | U | 60/10 | Partial | Owner-scoped profile/RPC |
| POST `/api/payment/create-order` | U | 5 | Partial | Creates pending subscription |
| POST `/api/payment/callback` | provider fields | none | Needs stronger verification review | Billing callback; live not tested |
| GET `/api/payment/return` | — | none | Query-driven redirect | No sensitive response expected |
| GET `/api/user/subscription` | U | 20 | N/A | Owner-scoped |
| `/api/admin/subscriptions*` | A | 20–30 | Partial IDs/body | Server-side admin check |
| GET `/api/practice-records/last-unresolved` | U | 10 | N/A | Owner-scoped |
| GET `/api/history` | U | 30 | bounded limit in code | Owner-scoped |
| GET `/api/progress` | U | 30 | N/A | Owner-scoped |
| GET `/api/diagnosis/timeline` | U | 20 | N/A | Owner-scoped API; cache RLS concern |
| GET `/api/practice-records/weakness-summary` | U | 10 | N/A | Owner-scoped |
| GET `/api/questions/bank` | — | 30 | filter/limit partial | Public shared content |
| GET `/api/questions/next` | U | 10 | Partial | Owner history influences result |
| GET `/api/drill/check_quota` | U | 20 | N/A | Non-atomic count |
| POST `/api/track/upgrade_*` | — | 30–60 | Partial | Public analytics writes |
| PATCH `/api/practice-records/{id}/resolve` | U | 60 | UUID + owner lookup | 204 / safe ownership |
| GET `/api/vocabulary/items` | — | 30 | Query bounds partial | Public catalog |
| `/api/vocabulary/my*`, `/review*` | U | 30–60 | Owner constraints mostly present | Service-role queries scoped to subject |
| POST `/api/vocabulary/generate` | U | 3 | Partial prompt/input bounds | Paid AI; IP-only rate limit |
| `/admin/*`, `/api/admin/*` | A | 3–30 | Varies; UUID checks common | Server-side allowlist; sampled missing token 401 |
| POST `/api/diagnosis/me` | U | 10 | No body | Raw exception leak path |
| GET `/part2/topics` | — | none | category allow behavior via data match | Public random topic |
| POST `/api/debug/rec-log` | — | 20 | Arbitrary JSON/truncation only | Public DB/log write; confirmed 200 |
| POST `/part2/evaluate` | U | 5 | **Insufficient upload/schema validation** | Provider + best-effort DB; P1/P2 findings |
| GET `/reading/quota` | U | 30 | N/A | Non-atomic quota snapshot |
| POST `/reading/attempt/abandon` | U | 30 | owner lookup | Owner-scoped |
| POST `/vocab/lookup`, `/translate_zh` | U | 30 | Some word/text bounds | Paid AI; IP-only limit |
| POST `/api/vocabulary/save_word` | U | 30 | Partial | Owner association |
| POST `/reading/passage/generate*` | U | 6 | Strict output validators, quota race | Paid AI; shared persisted passage |
| POST `/reading/questions/generate` | U | 6 | Output validator | Answer table directly readable by design |
| POST `/reading/attempt/start` | U | 12 | passage ID; idempotent same in-progress | Quota count-then-insert race |
| POST `/reading/attempt/submit` | U | 12 | owner attempt + question checks | Answer insert then attempt update; partial transaction risk |
| GET `/reading/attempt/{id}` | U | 30 | owner filter | Owner-scoped reveal API |
| GET `/reading/history` | U | 30 | bounded result | Owner-scoped |
| GET `/debug/sse_test` | — | none | N/A | Production debug surface |
| GET `/api/writing/question` | U | 20 | enums/query constraints | May return unsafe SVG sink downstream |
| POST `/api/writing/submit` | U | 10 | word count/question checks partial | Paid AI + persistence |
| GET `/api/writing/history`, `/submission/{id}` | U | 20 | owner filter; bounded limit | Cross-owner detail returns 404 |
| POST `/api/admin/writing/pregen` | A | 5 | no body | Starts background thread |

Auth negative results:

- `/process`: 401 without Authorization.
- `/api/history`, `/api/admin/subscriptions`, `/admin/recent`, `/reading/history`: 401 without Authorization.
- `/part2/evaluate`: missing multipart fields returns 422 before endpoint JWT code runs; a well-formed request still calls `verify_token` before provider work.
- Empty/random/expired/disabled/valid/cross-user token matrix: **NOT VERIFIED** without controlled accounts.

## 6. Speaking Flow Matrix

| Stage | Browser state | Backend/provider/database | Failure handling / audit result |
|---|---|---|---|
| Permission | `getUserMedia` | none | Modal/error exists; physical Safari not verified |
| Record P1 | recorder, timer, chunks | none | WebKit chooses MP4; others prefer WebM; support matrix not physical-tested |
| Stop P1 | stop/requestData, blob | none | duplicate-state protection is boolean/UI-based, not explicit state machine |
| Upload P1 | loading; 60s AbortController | JWT, 1 KiB–25 MiB check, temp file | Temp unlink in finally; type/magic/duration unverified |
| Transcribe P1 | waiting | Groq Whisper | Provider failure mapped but verbose upstream logging |
| Analyse P1 | waiting | Claude/Groq; correction validator + retries | Better than P2; quota/idempotency race remains |
| Save P1 | result transition | service-role insert | Normal insert failure returns error; retry_of ownership gap |
| Record P2 | prep/record timer, chunks | none | Apple forces MP4; no physical device verification |
| Upload P2 | score loading; 90s abort | JWT then unbounded `audio.read()` | P1 resource risk; retry retains blob |
| Transcribe P2 | waiting | Groq Whisper, temp unlink | Empty transcript 422; no audio validation/duration |
| Analyse P2 | waiting | Claude generic JSON parse | No strict contract; malformed dict accepted |
| Save P2 | score page | best-effort insert | DB failure swallowed; retry not idempotent |
| Render P2 | completed | none | Unescaped transcript/AI fields create XSS path |

The code has operational states, but not a single enforced state machine matching `idle → requesting_permission → recording → stopping → uploading → transcribing → analysing → saving → completed/error`. Generation counters exist for some question updates; there is no end-to-end request identity preventing a stale provider response or duplicate retry from winning.

Part 1/Part 2 content philosophy (one main issue, actionable task, Traditional Chinese, evidence accuracy) was reviewed in prompts but not empirically scored against real audio, so it is **PARTIAL**, not PASS.

## 7. Security Checklist

| Control | Result | Evidence |
|---|---|---|
| Frontend contains service-role key | PASS | Only anon JWT in frontend; service key read from backend env |
| Paid AI endpoints require JWT | PASS/PARTIAL | Main paid routes call verify_token; debug telemetry is public; full valid-token matrix absent |
| Backend derives user from JWT | PASS | verify_token returns Supabase user.id; most service queries use it |
| Admin enforced server-side | PASS | verify_admin calls Auth admin lookup and ADMIN_EMAILS allowlist |
| User-owned tables all use RLS | FAIL | subscriptions/diagnosis_cache migrations omit RLS |
| RLS live state | NOT VERIFIED | No live schema/admin access used |
| Reading answer secrecy | FAIL | reading_questions RLS disabled with answer fields |
| CORS allowlist | PASS | prod allowed; localhost/evil/null rejected; credentials false |
| Per-user rate limit | FAIL | global key is remote IP |
| Atomic quota/concurrency | FAIL | count-then-insert patterns |
| Upload type/size/duration | FAIL | P2 unbounded and filename-trusting; P1 only size |
| Temp-file cleanup | PASS | P1/P2 unlink in finally after temp creation |
| Strict AI schema | PARTIAL | P1 validator exists; P2/Writing insufficient |
| XSS-safe rendering | FAIL | P2 and Writing sinks confirmed |
| Safe error contract/request ID | FAIL | raw strings/log details; no request ID contract |
| Secrets committed | PASS for current tree | `.env` ignored/not tracked; full Git-history secret scan NOT VERIFIED |
| Security headers | FAIL | only HSTS among requested browser protections |
| Dependency vulnerabilities | NOT VERIFIED | pip-audit failed within bounded workflow |
| Physical browser/device matrix | NOT VERIFIED | no physical devices tested |

## 8. Prioritized Remediation Plan

### Within 24 hours — P1 containment

1. **Disable or safely escape Part 2 rich rendering.** Owner: frontend. Files: `frontend/app/index.html`. Acceptance: every transcript/provider field renders as text; malicious fixture creates no DOM element/handler. Test: focused DOM/XSS test. Migration: no. Production config: redeploy frontend.
2. **Disable Writing SVG charts or sanitize with a strict allowlist.** Owner: frontend/backend. Files: `frontend/app/writing.html`, `backend/main.py`. Acceptance: malicious SVG corpus is rejected/inert. Migration: no. Production config: deploy both sides if server sanitizer added.
3. **Contain Part 2 uploads.** Owner: backend. File: `backend/main.py`. Acceptance: bounded body/read, type+magic+duration validation, provider never called for invalid/over-limit files. Migration: no. Production config: possibly Render proxy/body and concurrency settings.
4. **Audit live Supabase grants immediately.** Owner: database/security. Files: new forward-only migrations. Acceptance: anon/authenticated cannot read `subscriptions`/`diagnosis_cache`; Reading answer keys unavailable before authorized reveal. Migration: yes, non-destructive policy/grant migration. Production config: Supabase.

### Within 3 days — core fixes and regression tests

1. Add strict Part 2 Pydantic response schema and safe fallback. Owner: backend. Test malformed/null/type/size/language/band cases. Migration: no.
2. Add user-owned `retry_of` lookup. Owner: backend/database. Test A→B returns 404. Migration: optional composite ownership constraint.
3. Add authenticated per-user rate key, atomic quota consume, in-flight concurrency, and idempotency keys. Owner: backend/database. Test parallel boundary requests. Migration: yes.
4. Remove/authenticate `/api/debug/rec-log` and `/debug/sse_test`. Owner: backend. Test missing JWT 401/route absent. Migration: optional cleanup/retention only; do not destructively clean without approval.
5. Repair offline test collection. Owner: backend. Acceptance: documented full pytest command collects/runs without real credentials.

### Within 7 days — reliability/browser/data consistency

1. Add CSP rollout, nosniff, referrer, permissions, and frame protection. Owner: frontend/platform. Test headers and core auth/recording flows in preview.
2. Test Chrome/Safari/Firefox plus physical iPhone Safari/Android Chrome for MIME, interruptions, backgrounding, and permissions.
3. Implement Part 2 request state/idempotency and explicit save failure semantics.
4. Add provider timeout/error/429/disconnect contract tests and redacted structured logs/request IDs.
5. Verify every live table/view/function/storage bucket against migration source; inventory orphans/duplicates/indexes with read-only aggregate queries.

### Within 30 days — architecture and supply-chain governance

1. Split public Reading content from protected answer material and move reveal to an owner-scoped RPC/API.
2. Pin all Python transitive dependencies with hashes; pin/self-host frontend JS with SRI; make CVE audit a bounded CI job.
3. Add liveness/readiness separation, metrics for provider latency/error/quota/idempotency, and alerting.
4. Add staging security E2E with controlled users A/B/admin and disposable synthetic audio.
5. Define data-retention/privacy policy for transcripts, recordings, provider payloads, rec_log, and admin audit events.

## 9. Final Verdict

```text
NOT SAFE FOR PUBLIC USE
```

Reason: production contains confirmed same-origin injection paths, Part 2 lacks a backend upload safety boundary, and the database migration posture exposes or plausibly exposes sensitive/answer-bearing tables to frontend roles. These are containment issues, not cosmetic hardening. Keep public access restricted until BLABBY-001 through BLABBY-004 are fixed and verified in production.

## Reproducible Command Log

```text
Command: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/pytest tests -v -p anyio
Exit code: 2
Passed: collection did not complete
Failed: 3 collection errors
Skipped: not reached
Duration: 50.90s reported by pytest

Command: PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 ./venv/bin/pytest tests/test_reading_validators.py -v
Exit code: 0
Passed: 22
Failed: 0
Skipped: 0
Duration: 0.02s reported by pytest

Command: ./venv/bin/pip check
Exit code: 0
Result: No broken requirements found

Command: uvx pip-audit -r requirements.txt
Exit code: 1
Result: audit environment creation failed (ensurepip SIGABRT); CVE result NOT VERIFIED

Command: uvx pip-audit --path venv/lib/python3.10/site-packages
Result: timed out/no usable result; terminated/not left running; CVE result NOT VERIFIED
```

Production probes used only GET/HEAD/OPTIONS or missing-auth requests, except the disclosed single empty `/api/debug/rec-log` POST. No load test, real user data, real recording, token, API key, JWT, full email, or provider secret is present in this report.

## Final Handoff

```text
Audit verdict: NOT SAFE FOR PUBLIC USE
P0 count: 0
P1 count: 4
P2 count: 8
P3 count: 3

Files inspected: backend/main.py; requirements files; frontend/app HTML/config; all Supabase migrations; tests; deployment responses
Tests executed: full pytest attempt; reading validator suite; pip check; bounded/failed pip-audit; safe production HTTP probes
Tests passed: 22 focused tests; pip check; CORS/auth-negative/header/deployment probes as documented
Tests failed: full suite collection (3 errors); pip-audit unavailable
Physical devices tested: 0 — Not verified on physical device
Production mutations performed: one possible all-null rec_log telemetry insert; no user data

Top 3 risks:
1. Part 2 and Writing same-origin XSS sinks.
2. Missing/unsafe RLS and exposed Reading answer data.
3. Unbounded, weakly validated Part 2 audio upload plus non-atomic abuse controls.

Recommended next action: restrict public access, implement the four 24-hour P1 containment items, then verify them in a staging clone and production with controlled accounts.
```
