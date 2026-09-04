# Blabby Schema Migration Truth

Status: Round E reconciliation record, 2026-09-04.

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

Current repository inventory is 35 SQL files: 30 forward-replay files (including two
baselines) and 5 rollback files. The forward runner intentionally excludes rollback
files. The previously observed production ledger count was 12, but no current ledger
readback was available in this run.

## Canonical reconciliation method

For an available production inspection, compare the allowlisted tables, columns,
types/defaults/nullability, constraints, RLS, policies, grants, functions and
signatures/definitions, views, and triggers with a clean forward replay. Inspect the
ledger separately and classify each entry as Blabby, shared/other-project, or
unknown. The production transaction must be `BEGIN; SET TRANSACTION READ ONLY;`
with `SHOW transaction_read_only` returning `on`, and must end with `ROLLBACK`.

That gate was unavailable here: no production DB URL, read-only helper, Supabase CLI
config, or approved connection variables were present. Consequently this document
does not claim current production parity. Historical notes are labelled as such and
must be refreshed before any production schema decision.

## Drift matrix

| Item | Repo | Ledger | Production schema | Classification | Action |
|---|---|---|---|---|---|
| `20260803111937_reading_pool_columns_and_backfill` | no | historical yes | current unavailable | `BLABBY_SOURCE_UNRECOVERED_BLOCKER` | Recover the original statement or obtain a read-only catalog snapshot; do not invent/apply SQL. |
| `p1_rls_and_reading_answers_idempotency_recheck` | no named file | historical yes | current unavailable | `BLABBY_RECONCILED_NO_UNIQUE_SCHEMA_DELTA` | Existing `20260714_p1_rls_and_reading_answers.sql` plus the documented idempotent rerun is the source truth; do not create a duplicate. |
| `create_npc_relations` | no | historical yes | out-of-scope shared | `SHARED_OTHER_PROJECT_EXCLUDED` | Ignore for Blabby; never copy, drop, or count it as Blabby drift. |
| `harden_npc_relations` | no | historical yes | out-of-scope shared | `SHARED_OTHER_PROJECT_EXCLUDED` | Ignore for Blabby; never copy, drop, or count it as Blabby drift. |
| `20260726_billing_identity_containment` | yes | historical not observed | current unavailable | `REPO_ONLY_LEDGER_ABSENCE_UNVERIFIED` | Do not add a ledger row or rerun; verify later in an approved read-only session. |
| `20260813_anonymous_process_quota` | yes | historical not observed | historical table present; current unavailable | `REPO_NO_LEDGER_SCHEMA_PRESENT_HISTORICAL` | Do not rerun or repair the ledger; verify current function/table parity later. |
| `20260904_retire_obsolete_waitlist_exposure` | yes | expected absent before deploy | current unavailable | `EXPECTED_PENDING_LOCAL_NOT_DEPLOYED` | Keep pending locally; it is not historical drift. |

The complete per-file inventory and the remaining early/repo-only grouping are in the
manifest. The key distinction is `REPO + NO LEDGER + SCHEMA PRESENT`: it means a
direct/manual application may have happened and must not be replayed or repaired by
writing the ledger.

## Known exceptions and missing source

The P1 idempotency entry is a recheck of the existing P1 source, not an independent
schema result. The repo already contains the authoritative forward SQL and the
rollout/baseline documents record the re-run; adding a duplicate would make replay
less truthful.

The Reading pool entry remains a real source-restoration blocker. The checkout,
Git history, dangling objects, prior pasted handoffs, and repo docs did not contain
the original SQL. Application/tests prove that the pool depends on
`reading_passages.is_pregenerated`, `reading_passages.questions_ready`,
`reading_passages.used_count`, and `idx_reading_passages_pool`, but they do not prove
the historical default, backfill predicate, or exact index definition. No migration
file was fabricated. A future read-only snapshot or original ledger statement is
required before adding a historical source artifact.

`anonymous_process_usage` is a repo-only/ledger-silent historical exception. The
repo migration and prior evidence record that the table existed; the correct action
is documentation and later parity verification, not a rerun or ledger mutation.

## Security contracts retained

- Reading answer protection remains the P1 column grant and the `42501` permission
  contract; the answer columns are absent from the authenticated grant.
- `user_pro_status` is recreated with `security_invoker`; `user_lookup` is not
  recreated by the containment migration.
- `anonymous_process_usage` enables RLS, revokes public/anon/authenticated access,
  and exposes only the service-role quota RPCs.
- `upgrade_intent` remains a legacy table. The Round D migration is local pending:
  it removes obsolete INSERT policies and revokes anon INSERT while preserving the
  table and historical data. Production status is not inferred here.
- `vocabulary_items` remains unchanged; its historical anon-readable policy is
  Backlog H and is outside Round E.
- No repo migration in this inventory touches the excluded shared domains.

## Rules

Do not rerun a migration merely because the ledger is silent. Do not copy shared
`npc_*` migrations. Do not edit historical migration files. Do not apply production
SQL from this contract. Production inspection must be read-only. Unknown ledger-only
Blabby entries or repo-only schema drift must fail the contract rather than becoming
permanent silent ignores.
