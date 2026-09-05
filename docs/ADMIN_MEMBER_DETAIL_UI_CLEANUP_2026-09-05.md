# Admin member management — Round L

Local implementation and browser review, 2026-09-05. No production verification or rollout is claimed.

## Preflight

Fresh commands in `/Users/yichengchiu/dev/Blabby/blabby` returned:

- HEAD: `510ccc2e3c0b6f9d3d5d4bec90f4d5a86bc61a6c`
- origin/main: `88eada001f89597eb7721ea425c6fd4af23edde3`
- `git rev-list --left-right --count origin/main...HEAD`: `0 5` (ahead/behind 5/0)
- Working tree: clean.

The Desktop checkout was not used for edits. No reset, clean, stash manipulation or broad staging was used.

## Audit and old pain points

Read the complete original 3,171-line `frontend/app/admin.html`, its inline JS/CSS, shared Task 1 chart helper, related backend handlers and existing admin/renderer/frontend tests before editing styles.

The member overview mixed three large count cards, weakness tags and a cross-skill activity feed. The header also contained report buttons and a grant/revoke shortcut. Selecting a member fetched their whole speaking response, then eagerly fetched writing and reading for the overview. Opening those tabs fetched them again. Late requests could render the previous member's data. Quality abbreviations in the sidebar required interpretation.

Two separate subscription presentations existed: an active three-column grant editor and an older subscription-table renderer whose host no longer existed. The redesign unifies member entitlement actions in the member's subscription tab; the top-level subscription button is a shortcut to that tab. The shared member sidebar supplies its search/filter/sort.

### Current data map and disposition

| Area | Existing source and payload | Former location | Reuse / gap / change |
|---|---|---|---|
| Member list | `GET /admin/users` → `get_admin_users_full` RPC; email, user_id, practice_count, last_practice, created_at, paid/grant/effective flags, grant audit, quality_counts | Left sidebar; copied again in subscription sidebar | Reused single list. Email/name search handles `display_name`/`name` if supplied. Current RPC does **not** supply a display-name field, so canonical data displays/searches email; no name is invented and no new identity API was added. |
| Summary | Same RPC row | Header + overview cards | Compact identity/email, effective PRO/FREE, joined and **recent speaking activity**. `last_practice` is not presented as cross-skill last activity. |
| Speaking | `GET /admin/user/{id}` → practice records with transcript/coaching/quality/mode | Full record list + overview counts/tags/feed | Now newest-first 10-row page; exact total, stable created_at/id ordering, offset and has_more. Preserves sequence field. Weakness counts use paged lightweight `weakness_tag` reads, not full transcripts. Metadata failure returns null, displayed as unavailable rather than zero. |
| Writing | `GET /admin/writing/submissions?user_id=...`; task type, prompt/subtype, band, essay, per-criterion feedback/fixes, priority fix, retry, submitted_at | Member tab + duplicate overview fetch; global writing queue | Existing list/detail helper retained. Lazy once per selected member; first 10 visible, more button. Existing service cap 200 explicitly stated; a full cap displays 200+. No new chart renderer: submission payload lacks chart_data/chart_svg. Global Task 1 human review continues to call the unchanged shared renderer. |
| Reading | `GET /admin/reading/attempts?user_id=...`; passage title/difficulty, dates, status, score/total/band, answer_count | Member tab + duplicate overview fetch; global monitor | Same lazy/compact behavior. Completed count means submitted attempts, not unique passages. Mean score/total uses only submitted, scored attempts with positive total; scope is loaded records, max 200. No cross-skill events. |
| Entitlement | `/admin/users` paid/grant/effective + audit fields | Header shortcut + separate grant editor | Reused expiry-aware grant editor in subscription tab. Shows subscription/grant/both/none from existing flags, expired grants remain revocable. Paid-only members have no misleading revoke-grant button. |
| Subscription rows | `GET /api/admin/subscriptions`; plan, status, amount, order_id, start/expiry | Older table renderer (unreachable host) | Added optional validated user_id filter **before** latest-100 limit. No filter retains global API behavior. Member rows expose only supplied fields; provider/renewal details are absent and not guessed. |
| Student brief | `GET /api/admin/student_brief/{id}`; `brief` text | Header button/output | Preserved separate action and copy. Despite its docstring saying last 30, actual query orders ascending and limits 30: UI truthfully says **earliest 30 valid speaking records**. No report algorithm change in this round. No server generation timestamp exists for this action. |
| AI diagnosis | `POST /admin/user/{id}/diagnosis`; structured summary/weakness evidence/next_step, raw fallback, generated_at | Header button/output | Separate explicit button; existing structured renderer retained, server timestamp shown. Existing service reads speaking history only and remains subject to its database response limit. Writing/reading are never advertised as report inputs. |

The example “100 speaking / 4 writing / 23 reading” is demonstrated with synthetic fixture records, not measured production counts. Speaking total is now a server count; writing/reading counts remain scoped to their existing capped response. Weaknesses use the repo's human-readable mapping (lack_detail → 缺少具體細節, etc.). The former mixed activity feed was removed as a presentation; individual histories and dates remain accessible in their skill tabs.

## New information hierarchy

Member sidebar → compact member header → exactly five primary tabs:

1. **口說** (default): total, common weaknesses, recent 10 records, quality/Part filters, newer/older pages, transcript and coaching detail; existing reclassification action retained.
2. **寫作**: submission count, compact history, expandable essay and AI feedback/fixes.
3. **閱讀**: attempt count, completed/average metrics with scope, compact recent attempts, expandable dates/status/score details.
4. **訂閱管理**: effective plan and source, manual grant audit/expiry, grant/revoke, subscription/order rows, extend/cancel, member-filtered session action log.
5. **AI 生成報告**: speaking data scope, student brief + copy, separate AI diagnosis + timestamp. No generation occurs on selection or tab opening.

Sidebar preserves All/PRO/FREE/practiced and recent/count/A–Z sorting. Native row buttons show readable speaking counts, email title and a strong green selection stripe. Header contains no actions, oversized cards, weakness block or progress bar.

The member stylesheet is scoped to this workspace. Warm off-white, green/gold, light row dividers and sans-serif data labels retain Blabby's identity. Existing dashboard, changelog, cross-user writing/reading monitors and Task 1 review remain accessible.

## Loading, isolation and errors

- Initial selection uses the already-loaded member row as summary; only the 10-record speaking page is requested.
- Writing, reading and subscriptions each fetch on first opening; rendered DOM and a simple selected-member cache preserve reopening state. Failed requests clear their cache flag and offer retry.
- Selecting any member clears all skill/report content and cache, resets speaking/quality/Part state and increments an epoch. Epoch checks prevent A → B → A response races, not just mismatched IDs.
- Report handlers retain their own DOM references across awaits, so a previous member's report cannot appear in a new member's panel. They are manual only and separately retryable.
- Empty/loading/error messages are scoped to their panel. Reading failure leaves speaking/subscription functional. A failed weakness aggregation does not erase recent speaking records.
- Grant/revoke/extend/cancel capture target identity before requests. Entitlement refresh updates list badges without automatically switching to the first member. Session grant/revoke logs are filtered to the selected member.
- ArrowLeft/ArrowRight/Home/End navigate the tablist with roving tabindex and aria-selected. Speaking and expandable history rows support Enter/Space.

## Authorization and action truth

Backend `verify_admin` remains unchanged: verify JWT, resolve server-side auth email, check environment-configured `ADMIN_EMAILS`. Every changed read handler still invokes it before DB reads. This is the only access authority.

Removed the hardcoded allowlist and redirect decision from **admin.html**. Its initial `/admin/users` request must succeed; 401/403 replaces the admin surface with an access-denied message. A backend-approved email outside the former frontend array works. The separate `index.html` cosmetic navigation affordance still has its pre-existing allowlist; it was inspected, documented and left outside this page change. It grants no API access.

Mutation handlers are reused unchanged on the backend:

- Manual grant requires a reason; permanent / 7-day / 30-day / custom future expiry form retained.
- Revoke confirms **manual grant only**, leaving paid subscription untouched.
- Cancel confirms preserved paid-window semantics; backend updates status only and leaves expiry/profile flags unchanged.
- Extend explicitly confirms 30 more days, active status and the existing matching-expiry manual grant side effect.
- All member management controls live within the subscription panel. No controls were exercised against production.

## Preserved capabilities and intentional removals

Preserved: member search/filter/sort/selection; full accessible speaking history via paging; Part/quality filters and reclassification; transcript/coaching/better expression detail; writing essay/criterion feedback/retry data; reading attempt detail; timed/permanent grant and revoke; subscription extend/cancel; student brief and copy; structured diagnosis; global dashboard/monitors/changelog; shared Task 1 renderer and review/serving separation.

Removed presentations: cryptic v/p/i sidebar counts, redundant overview cards/mixed feed/header progress, header action duplication, duplicate subscription sidebar and unreachable older grant/table UI functions. These are replaced by the five-tab workflow; report or entitlement capabilities were not silently merged/deleted. No print/download system was added.

## Tests

The new Node 20 harness executes the **actual complete inline admin script** in jsdom with synthetic fetch/auth responses. It is not a source-string-only layout test. `backend/tests/package.json` pins jsdom, its lockfile is checked in, CI installs it with `npm ci --prefix backend/tests --ignore-scripts`, and pytest runs the harness. No browser or external service is required in CI.

- Node 20.20.2: 25 behavioral checks pass: identity, exact tabs/default, request count/page size, paging, speaking detail, writing feedback, reading metrics, caching, action placement, no AI autoload, explicit diagnosis, pending report isolation, empty states, partial failure/retry, grant reason/expiry/target, cancelled revoke, stale speaking, name search, FREE filter, keyboard navigation and server denial.
- Focused pytest: **31 passed** (admin member endpoints + frontend + existing Task 1 review and CI ownership contracts).
- Endpoint tests cover scoped reads, latest-first exact counts, >1,000-row lightweight summary pagination, input bounds, partial summary failure, subscription user scope and backend admin denial before DB access.
- Full pytest: **697 passed, 10 skipped, 5 existing deprecation warnings**, 66.31 seconds. Credential-gated integration tests remained skipped.

Commands:

```sh
npm ci --prefix backend/tests --ignore-scripts --no-audit --no-fund
node backend/tests/frontend_admin_member_behavior.mjs
python -m pytest backend/tests/test_admin_member_contracts.py backend/tests/test_frontend_task1_review_contracts.py backend/tests/test_task1_review.py backend/tests/test_frontend_harness_ci_contract.py -q
python -m pytest backend/tests -q
```

Local execution used Python 3.11 and Node 20. Provider keys were synthetic and all READING_E2E targets/tokens explicitly empty for full pytest.

## Actual browser review

Used agent-browser against the actual local admin page served by `python3 scripts/preview_admin_member.py`. This loopback-only fixture removes remote config/CDN dependencies, injects synthetic auth/API responses, and sets `connect-src 'none'`. No production session, credentials or backend was loaded. Resource readback: **no external resources**; page error readback: **empty**. AI generation request count after visiting all five tabs: **0**.

Reviewed screenshots:

| View | Evidence | Observation |
|---|---|---|
| Speaking, 1440×1000 | [Speaking](evidence/admin-member-2026-09-05/speaking-1440.png) | Compact identity and weakness summary, selected sidebar, 10-row page, normal vertical scrolling. |
| Writing, 1440×1000 | [Writing](evidence/admin-member-2026-09-05/writing-1440.png) | Four compact rows, readable prompt/band; remaining space reflects only four submissions, not empty cards. |
| Reading, 1440×1000 | [Reading](evidence/admin-member-2026-09-05/reading-1440.png) | Scoped metrics, visible in-progress state, compact rows. |
| Subscription, 1440×1000 | [Subscription](evidence/admin-member-2026-09-05/subscription-1440.png) | Grant and subscription controls separated, source and audit available. |
| AI, 1440×1000 | [AI reports](evidence/admin-member-2026-09-05/ai-1440.png) | Two explicit generation buttons, scope and initial empty state. |
| Speaking, 1280×900 | [1280](evidence/admin-member-2026-09-05/speaking-1280.png) | Two columns, all tabs readable, no horizontal overflow. |
| Speaking, 1024×900 | [1024](evidence/admin-member-2026-09-05/speaking-1024.png) | Sidebar narrows, long email safely truncates with title, controls wrap. |
| Subscription, 1024×900 | [Subscription 1024](evidence/admin-member-2026-09-05/subscription-1024.png) | Action area fits; document width 1024, detail client/scroll widths both 784. |
| Writing detail, 1280×900 | [Essay and feedback](evidence/admin-member-2026-09-05/writing-detail-1280.png) | Browser interaction expands original essay/feedback detail. |

Below 760px the member list stacks above detail. Perfect mobile administration is outside scope. Operator timing targets (5/10/30 seconds) are design targets, not a measured user study.

## Production safety and rollout boundary

PRODUCTION DB WRITE: 0
PRODUCTION MIGRATION APPLY: 0
PRODUCTION DEPLOYMENT: 0
PUSH: NO

No pending migrations, Task 1 renderer architecture, quality index, physical waitlist drop, vocabulary OPTION D, student feature, or PostHog work was changed.

`20260905040134_atomic_vocabulary_save_quota.sql` remains **PENDING_BLABBY / LOCAL ONLY**. Round K's local contract remains closed and production rollout pending. Current backend still depends on that atomic RPC: **do not deploy it independently**. Round L's new speaking pagination and scoped subscription UI also require its matching backend when a future coordinated release is authorized.
