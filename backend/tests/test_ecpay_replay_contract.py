"""Structural contract for the PostgreSQL replay gate.

These tests do not pretend to execute SQL.  They make sure the CI entrypoint
still contains each runtime scenario as a complete block; GitHub Actions'
PostgreSQL 17 service is the behavioral authority.
"""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
REPLAY = (ROOT / "supabase/replay/replay.sh").read_text(encoding="utf-8")
MIGRATION = (
    ROOT / "supabase/migrations/20260730_ecpay_backend.sql"
).read_text(encoding="utf-8")


def _section(start: str, end: str) -> str:
    match = re.search(
        rf'echo "{re.escape(start)}.*?\n(?P<body>.*?)\necho "{re.escape(end)}',
        REPLAY,
        re.DOTALL,
    )
    assert match, f"missing replay section {start!r} → {end!r}"
    return match.group("body")


def test_replay_is_fail_fast_clean_db_ci_without_production_credentials():
    workflow = (ROOT / ".github/workflows/test.yml").read_text(encoding="utf-8")
    assert "set -euo pipefail" in REPLAY
    assert REPLAY.count("ON_ERROR_STOP=1") >= 5
    assert "postgres:17" in workflow
    assert "services:" in workflow and "migration-replay:" in workflow
    assert "PGURI: postgresql://postgres:postgres@localhost:5432/postgres" in workflow
    assert "supabase.co" not in REPLAY
    assert "SUPABASE_SERVICE_KEY" not in REPLAY


def test_atomic_section_covers_activation_duplicates_rejections_and_rollback():
    block = _section(
        "── assert ECPay acceptance is atomic and idempotent",
        "── assert RPC security and precise conflict handling",
    )
    assert block.count("public.accept_ecpay_payment(") >= 6
    for outcome in (
        "expected activated",
        "exact duplicate returned",
        "duplicate returned",
        "wrong amount was not rejected without side effects",
        "wrong user was not rejected without side effects",
        "ledger insert survived a rolled-back activation",
        "failed activation set an expiry",
    ):
        assert outcome in block
    assert "payment_events" in block
    assert "status FROM public.subscriptions" in block
    assert "expires_at FROM public.subscriptions" in block


def test_security_section_checks_exact_signature_acl_and_unrelated_unique_error():
    block = _section(
        "── assert RPC security and precise conflict handling",
        "── assert concurrent duplicate acceptance",
    )
    assert "to_regprocedure(" in block
    assert "prosecdef" in block
    assert "search_path=public" in block
    assert "aclexplode" in block
    assert "anon" in block and "authenticated" in block
    assert "service_role" in block
    assert "WHEN unique_violation" in block
    assert "replay_unrelated_payment_event_unique" in block
    assert "unrelated unique violation was swallowed as duplicate" in block


def test_concurrency_section_runs_two_sessions_and_checks_one_extension():
    block = _section(
        "── assert concurrent duplicate acceptance",
        "── assert the eight previously-unbacked tables exist",
    )
    assert block.count('psql "$PGURI"') >= 4
    assert '>"$CONCURRENCY_DIR/one" &' in block
    assert '>"$CONCURRENCY_DIR/two" &' in block
    assert 'wait "$pid1"' in block and 'wait "$pid2"' in block
    assert '"activated duplicate "' in block
    assert "count(*) FROM public.payment_events" in block
    assert "IS DISTINCT FROM '2099-02-01 00:00:00+00'" in block


def test_rpc_uses_only_the_idempotency_unique_identity_as_conflict_target():
    assert (
        "ON CONFLICT (merchant_trade_no, total_success_times, source) DO NOTHING"
        in MIGRATION
    )
    assert re.search(
        r"SELECT \*\s+INTO target\s+FROM public\.subscriptions.*?FOR UPDATE;",
        MIGRATION,
        re.DOTALL,
    )
    assert "target.user_id IS DISTINCT FROM p_expected_user_id" in MIGRATION
    assert "target.amount IS DISTINCT FROM p_expected_amount" in MIGRATION
    assert "ON CONFLICT DO NOTHING" not in MIGRATION
