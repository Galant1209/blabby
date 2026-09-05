# Vocabulary quota contract — Round K

**ATOMIC QUOTA MIGRATION: LOCAL READY / PRODUCTION BLOCKED**

The two active backend save routes now use one database transaction for owner serialization, idempotency, effective entitlement, quota, and insertion. This is a local/replayed contract; no production schema, data, policy, or deployment was changed.

## Preflight and previous race

Command-derived starting state in `/Users/yichengchiu/dev/Blabby/blabby`:

```text
HEAD: 3de9b30ae5c0c6cb60039af2fbd77847036bc36b
origin/main: 88eada001f89597eb7721ea425c6fd4af23edde3
origin/main...HEAD left/right: 0 4
ahead/behind: 4/0
working tree: clean
```

Round J closed the missing application quota check in Reading and preserved the Vocabulary/Speaking/Reading paywall contracts. Its count and insert were still separate requests. The recorded two-worker model allowed both callers to see 29 and finish at 31. Round K independently reproduced **31 with two real PostgreSQL connections** by removing the serialization mechanism from the new function in a disposable database. This distinguishes the route-level closure from the remaining database race.

## Existing data model

Source: `supabase/migrations/20260507_vocabulary.sql`; catalog readback in the fresh PostgreSQL 17.11 replay confirmed the following. These are **repo/local catalog facts, not current production catalog verification**.

| column | type / constraint / default |
|---|---|
| id | uuid primary key, default `gen_random_uuid()` |
| user_id | uuid NOT NULL, FK `auth.users(id)` ON DELETE CASCADE |
| vocabulary_item_id | uuid NOT NULL, FK `public.vocabulary_items(id)` ON DELETE CASCADE |
| status | text, default `new` |
| srs_level | integer, default 0 |
| review_count | integer, default 0 |
| correct_count | integer, default 0 |
| wrong_count | integer, default 0 |
| last_reviewed_at | timestamptz, nullable |
| next_review_at | timestamptz, default `now()` |
| source | text, nullable |
| source_practice_record_id | uuid, nullable; no FK declared |
| created_at | timestamptz, default `now()` |

The existing `user_vocabulary_user_id_vocabulary_item_id_key` is **UNIQUE(user_id, vocabulary_item_id)**. Its index already supports owner-prefix counting and pair lookup; no new constraint or index was needed. The migration fails if that expected uniqueness is absent. RLS remains enabled with `Users manage own vocabulary`, using and checking `auth.uid() = user_id`. Review logs are separate. No persistent application trigger on `user_vocabulary` was found or added; test-only triggers are removed before the harness finishes.

Before K, the local Supabase shim gave service_role default table privileges; the vocabulary migration itself had no explicit anon/authenticated table grants. After K, direct INSERT is denied to PUBLIC, anon, authenticated, and service_role, including column-level INSERT privileges. Other privileges/policies are preserved; local readback confirmed service_role SELECT and UPDATE remain allowed. Corpus `vocabulary_items` privileges and raw anonymous exposure are unchanged.

## Strategies considered

| option | decision |
|---|---|
| A — Advisory transaction lock by owner | **Selected.** One transaction, one owner key, no quota table, independent of worker count or profile existence |
| B — `profiles FOR UPDATE` | Not selected: `user_vocabulary.user_id` references auth.users, not profiles. Profile existence is not guaranteed by this schema |
| C — SERIALIZABLE plus retry | Correct with an explicit retry protocol, but larger than necessary for two save routes and one owner dimension |
| D — Dedicated counter / trigger architecture | Not needed; would add synchronization state and update/delete bookkeeping |
| E — Existing anonymous-quota row-lock pattern | Repository precedent for DB serialization, but its durable counter row is unnecessary for vocabulary |

Selected key:

```sql
pg_advisory_xact_lock(
  hashtextextended('blabby:vocabulary-save:' || p_user_id::text, 0)
)
```

Both route variants use the same key inside PostgreSQL. Locks last until transaction completion and apply across connections/workers. A hash collision can serialize unrelated owners, but cannot let two saves for the same owner escape serialization. See PostgreSQL 17's [explicit locking documentation](https://www.postgresql.org/docs/17/explicit-locking.html).

The function is explicitly VOLATILE and requires **READ COMMITTED** so its queries after waiting obtain fresh committed snapshots. It fails closed at other isolation levels rather than accepting a stale count. This condition matters when using explicit locks for application consistency; see [PostgreSQL's consistency guidance](https://www.postgresql.org/docs/17/applevel-consistency.html). It does not depend on a single Render worker or a Python lock.

## Entitlement truth

The function calls existing **`public.is_user_pro(p_user_id)`**. This is the same RPC used by backend `get_user_pro_status`, whose current docstring mentions historical paid mechanisms but whose effective truth is the database function.

The repository's reconciled body in `20260730_0_is_user_pro_reconciliation.sql` returns true for either:

- `profiles.is_pro_grant` with no expiry or a future `pro_grant_expires_at`;
- a subscription with `status='active'` and future `expires_at` for that owner.

Bare `profiles.is_pro` does not independently grant access in this reconciled contract. K does not modify the entitlement function, subscriptions, price, or grant rules. No `is_pro` argument exists in the new RPC. A SQL entitlement error rolls back the operation and becomes a generic API 503, with no Free/Pro fallback that could insert a row. Production must verify the actual deployed entitlement helper before any future release; local reuse does not certify its production body.

## RPC transaction and security

Function signature:

```text
public.save_vocabulary_atomic(
  p_user_id uuid,
  p_vocabulary_item_id uuid DEFAULT NULL,
  p_word text DEFAULT NULL,
  p_zh_meaning text DEFAULT '',
  p_source text DEFAULT 'manual_added',
  p_source_practice_record_id uuid DEFAULT NULL
) RETURNS jsonb
```

Supply exactly one of item ID or normalized word. The operation validates inputs/isolation, acquires the owner transaction lock, returns an existing owned link before entitlement/quota, otherwise checks effective Pro and Free count, then inserts the owned link. Free limit is fixed at **30** inside SQL, not a caller parameter. Pro has no saved-word cap.

For Reading, the same transaction resolves the corpus row and creates it only after quota allows the save. A failed owned insert rolls back the corpus insert. Same-owner same-word concurrent requests cannot create two rows through this RPC. If historical corpus spellings are duplicated, word lookup prefers an already-owned matching row for idempotency. No corpus-wide spelling constraint was introduced.

Security mode is **SECURITY DEFINER**, fixed `search_path = pg_catalog, pg_temp`, with schema-qualified application tables and entitlement function. The replay owner is postgres; migration ownership must remain a trusted deployment role. EXECUTE is revoked from PUBLIC, anon, and authenticated and granted only to service_role (apart from the function owner/superuser's inherent privileges). Browser roles cannot invoke an arbitrary-owner definer operation.

Both backend routes derive `p_user_id` from unchanged verified Bearer-token auth; JSON `user_id`, `is_pro`, or `p_is_pro` fields cannot select an owner or bypass entitlement. Direct owned-table INSERT is revoked, including column grants, from the three API roles and PUBLIC; unexpected inherited INSERT privileges make the migration fail and roll back. service_role is still trusted server infrastructure, not a browser credential. Privileged database owners/maintenance that change ownership, disable protections, or alter grants remain outside this RPC contract; there is no claim to constrain superusers or silently audit every external administrative writer.

## API integration

| path | old authoritative operation | new authoritative operation | preserved response |
|---|---|---|---|
| POST `/api/vocabulary/my` | Python count then INSERT | `_save_vocabulary_atomic` → RPC | Existing joined owned row with explicit corpus projection; refetch also filters verified user_id |
| POST `/api/vocabulary/save_word` | Python count, optional corpus INSERT, owned INSERT | Same helper/RPC with normalized word | `status=added` or `exists`, vocabulary_item_id, word |

RPC status `quota_reached` with limit 30 maps to HTTP 403:

```json
{"detail":{"error":"vocab_limit_reached","limit":30,"message":"Free users may save up to 30 words. Upgrade to Pro for unlimited vocabulary."}}
```

`not_found` maps to 404; malformed/unavailable/unknown RPC responses or unexpected limits fail closed with 503. Backend validates successful IDs/status/word. SQL exception text is not sent to the client. A missing function or an old database never triggers fallback Python inserts. `FREE_VOCABULARY_LIMIT` remains only for canonical response formatting/validation; SQL owns the quota decision.

Vocabulary's existing modal, Speaking's nested-error parsing, Reading's vocabulary modal, and `/upgrade.html?source=vocab_limit` are unchanged. No frontend file, checkout code, payment taxonomy, or price changed; Blabby Pro remains NT$199.

## Real concurrency and mutation evidence

`supabase/replay/test_atomic_vocabulary_quota.py` uses stdlib Python and psql against disposable **PostgreSQL 17.11**. Each concurrent operation has its own connection/transaction. A separate gate connection and test-only insertion trigger make overlap observable in `pg_stat_activity`: both workers must be waiting on advisory locks before release. This is not a mocked count or scheduling assumption.

| real DB case | result | final unique saved count |
|---|---|---|
| Free 29, two different item IDs | inserted + quota_reached | 30 |
| Free 29, ID save plus new Reading word | inserted + quota_reached; no losing-request corpus row | 30 |
| Free 30, existing plus new concurrently | existing + quota_reached | 30 |
| Free 29, same item twice | inserted + existing, identical saved ID, no conflict error | 30 |
| Grant Pro 35, two new items | inserted + inserted | 37 |
| Active-subscription Pro 35, two new items | inserted + inserted | 37 |
| Free 29 without profile, same new Reading word twice | inserted + existing; exactly one corpus row | 30 |
| Free 30, absent Reading word | quota_reached; no corpus row created | 30 |
| Injected owned INSERT failure after Reading corpus creation | whole operation rolled back, no corpus row | 29 |

Further DB assertions cover expired grant/subscription and bare is_pro remaining Free, actual anon/authenticated function permission errors, service-role direct INSERT denial despite BYPASSRLS, effective column privileges, preserved review UPDATE, and failure at REPEATABLE READ isolation.

**MUTATION PROBE: expected failure reproduced.** The harness obtains the live function definition and removes only the owner-lock statement in the disposable DB. Both callers then count 29 and wait at the test-only insert gate. On release both insert: the same expected-30 assertion fails with **`(['inserted', 'inserted'], 31)`**. The original function is restored in `finally`; the identical concurrency assertion then passes at 30. No repository migration is mutated by this probe.

## Replay, rerun, and application validation

New migration: `supabase/migrations/20260905040134_atomic_vocabulary_save_quota.sql`, UTC timestamp naming consistent with recent forward migrations. Historical migrations were not edited. It uses BEGIN/COMMIT, checks dependencies/uniqueness, creates/replaces the function, and sets explicit privileges. No new table/index/constraint/production rollback action was introduced.

The existing replay runner already discovers sorted forward SQL and excludes rollback files, so no execution allowlist change is needed. The repository's separate `schema_migration_truth_manifest.json` is updated with the new local function, inventory/reconciliation/drift entry, and expected-pending item. Its historical production snapshots/classifications are preserved. The inventory contract now expects 38 SQL files: 31 forward migrations plus 2 baselines and 5 rollback files.

`replay.sh` now runs the real DB contract harness before `REPLAY OK`, making failure visible in the existing migration-replay CI job. The harness needs only Python 3 and psql. It reapplies the migration after deliberately granting table and column INSERT privileges, verifies the restrictions are restored, and confirms the function definition matches. A fresh separate database completed the entire forward replay, existing migration proofs, K concurrency/mutation tests, and **REPLAY OK**. The shared/production project was not contacted.

Application tests in `backend/tests/test_vocabulary_save_quota.py` cover both routes, boundary/re-add/Pro behavior, verified ownership, client entitlement spoofing, missing/invalid RPC responses, safe errors, and absence of a Python count/insert fallback. Their RPC double tests HTTP integration; actual SQL semantics are tested separately by the replay harness.

Validation results:

- Vocabulary API: **41 passed**.
- Focused vocabulary/auth/frontend suite: **95 passed**, 5 existing deprecation warnings, Python 3.11 / Node 20.20.2.
- Real PostgreSQL concurrency, ACL, entitlement, rollback, mutation, restore, and migration rerun: **PASS**.
- Fresh full migration replay: **REPLAY OK**.
- Node 20 vocabulary paywall, Reading/Speaking save-path behavior, and active-vocabulary harnesses: **PASS**.
- Manifest/inventory contracts: **8 passed**; the new migration is explicitly local pending and historical snapshot fields are unchanged.
- Full `pytest tests -q`: **685 passed, 10 skipped, 0 failed**, 5 existing deprecation warnings, **47.53s**. Credential-gated integration endpoints/tokens were explicitly cleared and provider keys set to placeholders.

The local Homebrew PostgreSQL runtime required missing share/library links to be supplied temporarily. The database used a fresh private data/socket directory with TCP listening disabled. Both temporary runtime links and the running disposable server were removed/stopped after verification; no existing cluster was reused or changed.

## Remaining risks and production rollout

**LOCAL PENDING MIGRATION — NOT APPLIED TO PRODUCTION.** H/H-R still lacks a production READ ONLY SQL session, Render deployed SHA, and verified recoverable backup. Its allowlist remains NONE. K does not resolve any of these gates.

The function and backend must be released together under a controlled save pause after separate authorization. Deploying the backend before the RPC exists produces safe 503s; revoking direct INSERT while the old backend is still serving also breaks saves. No independent production deployment/apply is authorized, and no unsafe fallback was added to hide this compatibility boundary.

Before a future release, verify actual schema/uniqueness, effective grants (including unknown inherited roles/external writers), migration ownership, the deployed entitlement helper, and READ COMMITTED execution. This local migration is not a production drift repair. Timeouts/deadlocks/errors roll back the operation; ordinary client retry can use existing-item idempotency, but no general retry framework was introduced.

Historical over-limit collections or Pro-to-Free downgrades are not deleted/truncated; further new Free saves are denied and re-adds remain successful. Concurrent different owners can still create duplicate corpus spellings because there is no corpus word uniqueness contract; this does not permit an owner to exceed the atomic save bound. A privileged maintenance actor can change data/privileges outside the RPC; that is not an untrusted browser entitlement path.

OPTION D raw anonymous exposure, ADMIN_EMAILS, Task 1, pro_waitlist physical drop, and `practice_records_quality_grade_idx` are unchanged and outside K.

```text
PRODUCTION DB WRITE: 0
PRODUCTION POLICY CHANGE: 0
PRODUCTION MIGRATION APPLY: 0
PRODUCTION DEPLOYMENT: 0
GIT PUSH: NO
ATOMIC QUOTA MIGRATION: LOCAL READY / PRODUCTION BLOCKED
```
