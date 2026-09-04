# Blabby Schema Migration Truth

Status: Round E-R Phase 2 reconciliation record, 2026-09-04.

This is an engineering contract, not a handoff. The machine-readable companion is
[`schema_migration_truth_manifest.json`](schema_migration_truth_manifest.json),
with a pytest guard in
[`backend/tests/test_schema_migration_truth_contract.py`](../backend/tests/test_schema_migration_truth_contract.py).

## Scope

Blabby runs in a shared Supabase project. The Supabase migration ledger is therefore
not a Blabby-only ledger. `gmail_*`, `omg_*`, `npc_*`, and any other object not used by
Blabby are explicitly outside this reconciliation. In particular,
`create_npc_relations` and `harden_npc_relations` are shared, other-project ledger
noise; they must never be copied into this repository or counted as Blabby drift.

The allowlist is derived from the backend/frontend database call sites and the repo
migrations. It covers the core identity, speaking/practice, Reading, Writing,
Vocabulary, anonymous quota, billing, legacy, and admin objects recorded in the
manifest. Migration filename counts are inventory metadata only; they do not prove
that a schema change was applied.

Current repository inventory is 36 SQL files: 31 forward-replay files (including two
baselines) and 5 rollback files. The forward runner intentionally excludes rollback
files. The current production ledger readback contains 12 entries; all 12 are
classified below, including the two shared/other-project entries.

## Canonical reconciliation method

For an available production inspection, compare the allowlisted tables, columns,
types/defaults/nullability, constraints, RLS, policies, grants, functions and
signatures/definitions, views, and triggers with a clean forward replay. Inspect the
ledger separately and classify each entry as Blabby, shared/other-project, or
unknown. The production transaction must be `BEGIN; SET TRANSACTION READ ONLY;`
with `SHOW transaction_read_only` returning `on`, and must end with `ROLLBACK`.

The Phase 2 gate completed on the approved production connection: `transaction_read_only`
was `on`, `current_database()` and `current_user` were `postgres`, and the session
ended with `ROLLBACK`. The catalog-only allowlist snapshot contains 887 normalized
records and has SHA-256
`4ce44086b553a2a2340dce358a0fb55a25711e0b392827c5f11270907bf13ab`.
The database reports no in-band Supabase project-ref setting; project-ref validation
therefore remains a connection-target check, not a claim derived from catalog data.

## Drift matrix

| Item | Repo | Ledger | Production schema | Classification | Action |
|---|---|---|---|---|---|
| `20260803111937_reading_pool_columns_and_backfill` | yes | current yes | current yes | `BLABBY_RECONSTRUCTED_SOURCE` | Exact ledger statement restored; clean replay verifies the three columns, index, and backfill. Do not apply it to production from this contract. |
| `p1_rls_and_reading_answers_idempotency_recheck` | no named file | current yes | current snapshot yes | `BLABBY_RECONCILED_NO_UNIQUE_SCHEMA_DELTA` | Existing `20260714_p1_rls_and_reading_answers.sql` plus the documented idempotent rerun is the source truth; do not create a duplicate. |
| `create_npc_relations` | no | historical yes | out-of-scope shared | `SHARED_OTHER_PROJECT_EXCLUDED` | Ignore for Blabby; never copy, drop, or count it as Blabby drift. |
| `harden_npc_relations` | no | historical yes | out-of-scope shared | `SHARED_OTHER_PROJECT_EXCLUDED` | Ignore for Blabby; never copy, drop, or count it as Blabby drift. |
| `20260726_billing_identity_containment` | yes | current not observed | current snapshot partial parity | `REPO_ONLY_LEDGER_ABSENCE_CONFIRMED` | Do not add a ledger row or rerun; inspected current objects do not authorize ledger repair. |
| `20260813_anonymous_process_quota` | yes | current not observed | current yes | `REPO_NO_LEDGER_SCHEMA_PRESENT_CONFIRMED` | Do not rerun or repair the ledger; current table and quota functions were inspected read-only. |
| `20260904_retire_obsolete_waitlist_exposure` | yes | expected absent before deploy | current old state | `EXPECTED_PENDING_LOCAL_NOT_DEPLOYED` | Keep pending locally; production still has the three legacy INSERT policies and anon INSERT grant. |
| `20260508_quality_grade_index` | yes | not observed | current missing | `REPO_SCHEMA_DELTA_NOT_IN_PRODUCTION` | Record explicit repo-only index drift; no production apply without separate authorization. |

The complete per-file inventory and the remaining early/repo-only grouping are in the
manifest. The key distinction is `REPO + NO LEDGER + SCHEMA PRESENT`: it means a
direct/manual application may have happened and must not be replayed or repaired by
writing the ledger.

## Known exceptions and missing source

The P1 idempotency entry is a recheck of the existing P1 source, not an independent
schema result. The repo already contains the authoritative forward SQL and the
rollout/baseline documents record the re-run; adding a duplicate would make replay
less truthful.

The Reading pool source was recovered from the current production ledger as one
statement (`statement_md5=8127e822b00fbebfa246ecf9ace93c95`) and restored verbatim as
`supabase/migrations/20260803111937_reading_pool_columns_and_backfill.sql`. It contains
the three `NOT NULL` defaults, `idx_reading_passages_pool`, and the
`reading_questions`-existence backfill predicate. No SQL was invented or applied to
production.

`anonymous_process_usage` is a repo-only/ledger-silent historical exception. The
current snapshot confirms the table and both quota functions; the correct action
remains documentation, not a rerun or ledger mutation.

## Security contracts retained

- Reading answer protection remains the P1 column grant and the `42501` permission
  contract; the answer columns are absent from the authenticated grant.
- `user_pro_status` is recreated with `security_invoker`; `user_lookup` is not
  recreated by the containment migration.
- `anonymous_process_usage` enables RLS, revokes public/anon/authenticated access,
  and exposes only the service-role quota RPCs.
- `upgrade_intent` remains a legacy table. The Round D migration is local pending:
  it removes obsolete INSERT policies and revokes anon INSERT while preserving the
  table and historical data. The current production snapshot confirms the old
  policies and grant remain; no production change was made.
- `vocabulary_items` remains unchanged; its historical anon-readable policy is
  Backlog H and is outside Round E.
- No repo migration in this inventory touches the excluded shared domains.

## Rules

Do not rerun a migration merely because the ledger is silent. Do not copy shared
`npc_*` migrations. Do not edit historical migration files. Do not apply production
SQL from this contract. Production inspection must be read-only. Unknown ledger-only
Blabby entries or repo-only schema drift must fail the contract rather than becoming
permanent silent ignores.

## Phase 2 replay comparison

The clean local PostgreSQL 17 replay completed with `REPLAY OK` after source recovery.
The full allowlist projection before source recovery had 104 production-only records;
the recovered replay then matched the four Reading records exactly (three columns and
one index), leaving 100 production-only, one replay-only, and four same-object payload
differences. No unexplained differences remain. The classified differences are: 97
Supabase default-role table grants not modeled by the local shim, the three pending
`upgrade_intent` policies plus anon INSERT grant, the repo-only
`practice_records_quality_grade_idx`, two catalog-only subscription ordinal shifts
caused by the replay's intentional add/drop fixture, and non-executable function
text/ACL differences (one explanatory comment and local default function ACLs).
