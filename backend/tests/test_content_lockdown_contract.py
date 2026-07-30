"""Structural contract for the content-access lockdown.

Like test_ecpay_replay_contract.py, this does not execute SQL. It asserts the
migration and the replay harness still contain each load-bearing piece, so a
future edit cannot quietly remove one. PostgreSQL 17 in CI is the behavioural
authority.

The reading_questions assertions are the reason this file exists. On 2026-07-30
an audit misread a working column-level GRANT as an answer leak and drafted a
"fix" that would have deleted it. These tests make that specific regression
loud: if reading_questions ever appears in a mutation of the lockdown
migration, this file fails.
"""

from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[2]
MIGRATION = (
    ROOT / "supabase/migrations/20260731_content_access_lockdown.sql"
).read_text(encoding="utf-8")
ROLLBACK = (
    ROOT / "supabase/migrations/20260731_content_access_lockdown_rollback.sql"
).read_text(encoding="utf-8")
REPLAY = (ROOT / "supabase/replay/replay.sh").read_text(encoding="utf-8")
SHIM = (ROOT / "supabase/replay/00_local_shim.sql").read_text(encoding="utf-8")
AUDIT = (ROOT / "scripts/rls_exposure_audit.sql").read_text(encoding="utf-8")
CLAUDE_MD = (ROOT / "CLAUDE.md").read_text(encoding="utf-8")

THREE_TABLES = ("questions", "reading_passages", "writing_questions")


def _code_only(sql: str) -> str:
    """Strip `--` comments so assertions test statements, not prose.

    These files deliberately discuss the wrong approaches at length in their
    headers, so a naive substring check against the whole file matches the
    warning rather than any executable statement.
    """
    return "\n".join(
        line.split("--", 1)[0] for line in sql.splitlines()
    )


def _mutation_section() -> str:
    """Everything after the preflight — i.e. the statements that change state."""
    return MIGRATION[MIGRATION.index("$preflight$;") + len("$preflight$;"):]


# ── transaction and gate ordering ────────────────────────────────────────
def test_migration_owns_its_transaction_and_gates_before_mutation():
    assert re.search(r"(?m)^BEGIN;$", MIGRATION)
    assert re.search(r"(?m)^COMMIT;$", MIGRATION)
    assert MIGRATION.index("BEGIN;") < MIGRATION.index("DO $preflight$")
    assert MIGRATION.index("DO $preflight$") < MIGRATION.index("DROP POLICY IF EXISTS")
    assert MIGRATION.index("DROP POLICY IF EXISTS") < MIGRATION.rindex("COMMIT;")
    # A swallowed exception would turn an abort into a silent partial apply.
    assert "WHEN OTHERS" not in MIGRATION


# ── both layers, not one ─────────────────────────────────────────────────
def test_both_rls_and_grant_layers_are_addressed():
    mutations = _mutation_section()
    for table in THREE_TABLES:
        assert f"ALTER TABLE public.{table}          ENABLE ROW LEVEL SECURITY" in mutations \
            or f"ALTER TABLE public.{table}   ENABLE ROW LEVEL SECURITY" in mutations \
            or f"ALTER TABLE public.{table}  ENABLE ROW LEVEL SECURITY" in mutations, table
        assert re.search(rf"REVOKE ALL ON TABLE public\.{table}\s+FROM anon, authenticated;",
                         mutations), table


def test_zero_policies_are_created():
    """RLS on plus a permissive policy is not a lockdown."""
    mutations = _mutation_section()
    assert "CREATE POLICY" not in mutations.upper()


def test_the_20260726_policies_are_dropped_by_name():
    """Replay applies 20260726 §4, so the lockdown must remove its policies."""
    mutations = _mutation_section()
    for policy in ("questions_read_authenticated",
                   "writing_questions_read_authenticated",
                   "reading_passages_read_own"):
        assert f"DROP POLICY IF EXISTS {policy}" in mutations, policy


# ── reading_questions must never be mutated here ─────────────────────────
def test_reading_questions_is_never_mutated_by_the_lockdown():
    """The 2026-07-30 near-miss, encoded.

    reading_questions may be *inspected* by the preflight, but no statement
    after the preflight may touch it.
    """
    mutations = _code_only(_mutation_section())
    for forbidden in ("ALTER TABLE public.reading_questions",
                      "DROP POLICY IF EXISTS reading_questions_prompt_select",
                      "REVOKE ALL ON TABLE public.reading_questions",
                      "REVOKE SELECT (", "GRANT SELECT ("):
        assert forbidden not in mutations, forbidden
    # The rollback must not touch it either — comments about it are fine and
    # in fact required, so only executable statements are checked.
    assert "reading_questions" not in _code_only(ROLLBACK)


def test_preflight_proves_the_reading_questions_grants_are_intact():
    preflight = MIGRATION[
        MIGRATION.index("DO $preflight$"):MIGRATION.index("$preflight$;")
    ]
    # 7 safe columns granted, 3 answer columns not, anon nothing, policy present.
    assert "answers_granted <> 0" in preflight
    assert "safe_granted <> 7" in preflight
    assert "anon_rq_cols <> 0" in preflight
    assert "reading_questions_prompt_select is missing" in preflight
    for col in ("correct_answer", "explanation", "evidence_quote"):
        assert col in preflight, col


def test_preflight_accepts_all_three_real_states_and_names_strays():
    preflight = MIGRATION[
        MIGRATION.index("DO $preflight$"):MIGRATION.index("$preflight$;")
    ]
    # (A) production, (B) 20260726 §4 applied, (C) already locked.
    assert preflight.count("rls = false") >= 1
    assert preflight.count("rls = true") >= 2
    assert "unrecognised state" in preflight
    assert "unrecognised policy/policies" in preflight
    assert "known_policies" in preflight


# ── rollback honesty ─────────────────────────────────────────────────────
def test_rollback_warns_loudly_before_anything_else():
    head = ROLLBACK[:2000]
    assert head.lstrip().startswith("-- #")
    assert "WARNING" in head
    assert "re-opens" in head.lower() or "readable by the anon role again" in head
    for table in THREE_TABLES:
        assert table in head, table
    # The banner must precede the first statement that re-opens anything.
    assert ROLLBACK.index("WARNING") < ROLLBACK.index("GRANT SELECT")
    assert ROLLBACK.index("WARNING") < ROLLBACK.index("DISABLE ROW LEVEL SECURITY")


# ── the replay proves it by role, not by catalog ─────────────────────────
def test_replay_verifies_by_role_and_covers_service_role_and_answers():
    section = REPLAY[REPLAY.index("assert content lockdown by ROLE"):]
    assert "SET LOCAL ROLE anon" in section
    assert "SET LOCAL ROLE authenticated" in section
    assert "SET LOCAL ROLE service_role" in section
    assert "insufficient_privilege" in section
    # service_role must keep working or FastAPI is dead.
    assert "FastAPI would break" in section
    # answers stay denied; the answering path stays open.
    assert "ANSWER LEAK" in section
    assert "broke the answering path" in section
    # rerun and rollback are both exercised.
    assert "complete rerun" in section
    assert "rollback verified, re-locked" in section


def test_shim_makes_service_role_faithful_to_supabase():
    """Without BYPASSRLS the replay cannot tell a good lockdown from a broken one."""
    assert "ALTER ROLE service_role BYPASSRLS;" in SHIM
    assert "GRANT ALL ON TABLES TO service_role" in SHIM
    # anon/authenticated must NOT be widened, or the replay stops being evidence.
    assert "GRANT ALL ON TABLES TO anon" not in SHIM


# ── the audit script must not regress to table-level probing ─────────────
def test_audit_script_is_column_level_and_gathers_evidence():
    code = _code_only(AUDIT)
    assert "has_column_privilege" in code
    assert "has_table_privilege" not in code, (
        "table-level probing is what caused the 2026-07-30 misreading"
    )
    # It must still *explain* the pitfall, or the next author repeats it.
    assert "has_table_privilege" in AUDIT
    assert "SET LOCAL ROLE" in code
    assert "vocabulary_items" in AUDIT
    assert "ROLLBACK;" in AUDIT


# ── the rule is written down where the next person will look ─────────────
def test_claude_md_records_the_zero_policy_rule_and_the_asymmetry():
    assert "零 policy" in CLAUDE_MD
    assert "reading_questions" in CLAUDE_MD
    assert "42501" in CLAUDE_MD
    assert "has_table_privilege" in CLAUDE_MD
