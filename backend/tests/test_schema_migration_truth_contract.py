"""Machine checks for the Round E schema reconciliation contract.

This test is deliberately repository-only. It never connects to Supabase and does
not pretend to prove current production parity.
"""

from __future__ import annotations

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "docs" / "schema_migration_truth_manifest.json"
MIGRATIONS_DIR = ROOT / "supabase" / "migrations"


def _manifest() -> dict:
    return json.loads(MANIFEST_PATH.read_text())


def test_manifest_covers_every_repo_migration_with_the_right_kind():
    manifest = _manifest()
    inventory = manifest["repo_inventory"]
    repo_files = sorted(path.name for path in MIGRATIONS_DIR.glob("*.sql"))
    listed_files = sorted(item["name"] for item in inventory)

    assert listed_files == repo_files
    assert len(inventory) == 39
    assert sum(item["kind"] == "forward" for item in inventory) == 32
    assert sum(item["kind"] == "rollback" for item in inventory) == 5
    assert sum(item["kind"] == "baseline" for item in inventory) == 2
    assert sum(item["kind"] in {"forward", "baseline"} for item in inventory) == 34

    forward_names = {
        item["name"] for item in inventory if item["kind"] in {"forward", "baseline"}
    }
    reconciled_names = {
        name
        for group in manifest["repo_only_reconciliation"]
        for name in group["files"]
    }
    assert reconciled_names == forward_names - {
        "20260617120832_create_rec_log_table.sql",
        "20260618040300_create_writing_module_tables.sql",
        "20260618045448_writing_questions_add_svg_pregen.sql",
        "20260630033125_add_retry_of_to_practice_records.sql",
        "20260714_p1_rls_and_reading_answers.sql",
        "20260803111937_reading_pool_columns_and_backfill.sql",
    }

    allowed_domains = {
        "core", "identity", "speaking", "reading", "writing", "vocabulary",
        "anonymous", "billing", "admin", "security", "content", "legacy",
        "diagnostics",
    }
    for item in inventory:
        name = item["name"]
        expected_kind = (
            "baseline" if name.startswith("000")
            else "rollback" if name.endswith("_rollback.sql")
            else "forward"
        )
        assert item["kind"] == expected_kind
        assert item["domains"]
        assert set(item["domains"]) <= allowed_domains
        assert item["touched_objects"]


def test_atomic_vocabulary_source_is_local_pending_not_production_verified():
    manifest = _manifest()
    name = "20260905040134_atomic_vocabulary_save_quota"
    entry = next(item for item in manifest["drift_matrix"] if item["item"] == name)
    assert entry["classification"] == "EXPECTED_PENDING_LOCAL_NOT_DEPLOYED"
    assert entry["ledger"] == "NOT_VERIFIED_IN_ROUND_K"
    assert entry["production_schema"] == "NOT_VERIFIED_IN_ROUND_K"
    assert "save_vocabulary_atomic" in manifest["scope"]["functions"]


def test_allowlist_and_inventory_exclude_shared_other_project_domains():
    manifest = _manifest()
    excluded = tuple(manifest["scope"]["excluded_shared_patterns"])
    assert excluded == ("gmail_*", "omg_*", "npc_*")

    touched_objects = {
        obj.lower()
        for item in manifest["repo_inventory"]
        for obj in item["touched_objects"]
    }
    for prefix in ("gmail_", "omg_", "npc_"):
        assert not any(obj.startswith(prefix) for obj in touched_objects)

    assert {
        item["name"] for item in manifest["reconciliation_contract"]["shared_ledger_exceptions"]
    } == {"create_npc_relations", "harden_npc_relations"}


def test_known_drift_is_classified_and_unknown_drift_is_not_silently_ignored():
    manifest = _manifest()
    matrix = {item["item"]: item for item in manifest["drift_matrix"]}
    required = {
        "20260803111937_reading_pool_columns_and_backfill",
        "p1_rls_and_reading_answers_idempotency_recheck",
        "create_npc_relations",
        "harden_npc_relations",
        "20260726_billing_identity_containment",
        "20260813_anonymous_process_quota",
        "20260904_retire_obsolete_waitlist_exposure",
        "20260508_quality_grade_index",
    }
    assert required <= matrix.keys()
    assert manifest["reconciliation_contract"]["unknown_drift_causes_failure"] is True
    assert manifest["historical_ledger"]["unclassified_entries"] == []
    allowed_ledger_classes = {
        "BLABBY_RECONSTRUCTED_SOURCE",
        "BLABBY_RECONCILED_SOURCE",
        "BLABBY_RECONCILED_NO_UNIQUE_SCHEMA_DELTA",
        "BLABBY_SOURCE_UNRECOVERED_BLOCKER",
        "SHARED_OTHER_PROJECT_EXCLUDED",
    }
    assert {
        entry["classification"] for entry in manifest["historical_ledger"]["known_entries"]
    } <= allowed_ledger_classes

    assert matrix["20260803111937_reading_pool_columns_and_backfill"]["classification"] == (
        "BLABBY_RECONSTRUCTED_SOURCE"
    )
    assert matrix["p1_rls_and_reading_answers_idempotency_recheck"]["classification"] == (
        "BLABBY_RECONCILED_NO_UNIQUE_SCHEMA_DELTA"
    )
    assert matrix["create_npc_relations"]["classification"] == "SHARED_OTHER_PROJECT_EXCLUDED"
    assert matrix["harden_npc_relations"]["classification"] == "SHARED_OTHER_PROJECT_EXCLUDED"
    assert matrix["20260904_retire_obsolete_waitlist_exposure"]["classification"] == (
        "EXPECTED_PENDING_LOCAL_NOT_DEPLOYED"
    )
    assert matrix["20260508_quality_grade_index"]["classification"] == (
        "REPO_SCHEMA_DELTA_NOT_IN_PRODUCTION"
    )


def test_recovered_reading_pool_source_is_recorded_and_complete():
    manifest = _manifest()
    source = (MIGRATIONS_DIR / "20260803111937_reading_pool_columns_and_backfill.sql").read_text()
    recovered = {
        item["name"]: item for item in manifest["ledger_reconstructed_sources"]
    }["20260803111937_reading_pool_columns_and_backfill"]

    assert recovered["version"] == "20260803111937"
    assert recovered["statement_count"] == 1
    assert "ADD COLUMN is_pregenerated boolean NOT NULL DEFAULT false" in source
    assert "ADD COLUMN questions_ready boolean NOT NULL DEFAULT false" in source
    assert "ADD COLUMN used_count integer NOT NULL DEFAULT 0" in source
    assert "CREATE INDEX idx_reading_passages_pool" in source
    assert "WHERE EXISTS" in source and "reading_questions q" in source


def test_anonymous_quota_billing_and_reading_security_sources_remain_present():
    anonymous = (MIGRATIONS_DIR / "20260813_anonymous_process_quota.sql").read_text()
    assert "CREATE TABLE IF NOT EXISTS public.anonymous_process_usage" in anonymous
    assert "ALTER TABLE public.anonymous_process_usage ENABLE ROW LEVEL SECURITY" in anonymous
    assert "CREATE OR REPLACE FUNCTION public.get_anonymous_process_quota" in anonymous
    assert "CREATE OR REPLACE FUNCTION public.consume_anonymous_process_quota" in anonymous

    billing = (MIGRATIONS_DIR / "20260726_billing_identity_containment.sql").read_text()
    assert "CREATE TABLE IF NOT EXISTS public.payment_events" in billing
    assert "CREATE OR REPLACE FUNCTION public.is_user_pro" in billing
    assert "CREATE VIEW public.user_pro_status" in billing

    p1 = (MIGRATIONS_DIR / "20260714_p1_rls_and_reading_answers.sql").read_text()
    assert "reading_questions_prompt_select" in p1
    reading_grant = re.search(
        r"grant\s+select\s*\(([^)]*)\)\s+on\s+table\s+public\.reading_questions",
        p1,
        re.IGNORECASE | re.DOTALL,
    )
    assert reading_grant is not None
    assert "correct_answer" not in reading_grant.group(1)

    round_d = (MIGRATIONS_DIR / "20260904_retire_obsolete_waitlist_exposure.sql").read_text()
    assert "DROP POLICY IF EXISTS" in round_d
    assert "REVOKE INSERT ON TABLE public.upgrade_intent FROM anon" in round_d


def test_replay_and_production_status_are_explicitly_separate():
    manifest = _manifest()
    assert manifest["replay"]["forward_migration_replay"] == "PASS"
    assert manifest["replay"]["rollback_files_in_forward_replay"] is False
    assert manifest["replay"]["production_parity"] == "READ_ONLY_SNAPSHOT_WITH_CLASSIFIED_DIFFS"
    assert manifest["replay"]["production_parity_must_not_be_inferred"] is True
    assert manifest["production_inspection"]["available"] is True
    assert manifest["production_inspection"]["transaction_read_only"] == "on"
    assert manifest["production_inspection"]["catalog_snapshot"]["record_count"] == 887
    assert manifest["replay"]["classified_diff_summary"]["unexplained_differences"] == 0


def test_task1_review_migration_is_safe_and_serving_separate():
    source = (MIGRATIONS_DIR / "20260904123000_task1_chart_human_review.sql").read_text()
    assert "ADD COLUMN IF NOT EXISTS review_status text NOT NULL DEFAULT 'pending'" in source
    assert "ADD COLUMN IF NOT EXISTS review_issue text" in source
    assert "ADD COLUMN IF NOT EXISTS review_note text" in source
    assert "ADD COLUMN IF NOT EXISTS reviewed_at timestamptz" in source
    assert "ADD COLUMN IF NOT EXISTS reviewed_by uuid" in source
    assert "review_status IN ('pending', 'approved', 'needs_fix', 'retired')" in source
    assert "renderer_unsupported" in source
    assert "idx_writing_questions_review_status" in source
    assert "UPDATE writing_questions" not in source
    assert "SET is_pregenerated" not in source


def test_public_vocabulary_hardening_is_local_pending():
    manifest = _manifest()
    name = "20260905120000_vocabulary_public_access_lockdown"
    entry = next(item for item in manifest["drift_matrix"] if item["item"] == name)
    assert entry["ledger"] == entry["production_schema"] == "NOT_VERIFIED_IN_ROUND_N"
    assert entry["classification"] == "EXPECTED_PENDING_LOCAL_NOT_DEPLOYED"
    assert name in manifest["reconciliation_contract"]["expected_pending"]
    runner = (ROOT / "supabase/replay/replay.sh").read_text()
    assert 'python3 "$HERE/test_public_vocabulary_access.py" --disposable' in runner
