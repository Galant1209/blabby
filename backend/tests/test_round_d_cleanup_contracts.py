"""Static contracts for Round D's dead waitlist/admin/track cleanup."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[2]
APP_DIR = ROOT / "frontend" / "app"
BACKEND = ROOT / "backend" / "main.py"
SCRIPT = ROOT / "scripts" / "verify_sprint2b.py"
MIGRATION = ROOT / "supabase" / "migrations" / "20260904_retire_obsolete_waitlist_exposure.sql"


def test_frontend_has_no_obsolete_waitlist_surfaces():
    html = "\n".join(p.read_text(encoding="utf-8") for p in APP_DIR.glob("*.html"))
    assert "pro_waitlist" not in html
    assert "upgrade_intent" not in html
    assert "upgrade_page_view" not in html
    assert "upgrade_interest" not in html
    assert "waitlist-content" not in html


def test_removed_track_and_admin_routes_are_absent_from_runtime_source():
    source = BACKEND.read_text(encoding="utf-8")
    for route in (
        "/api/track/upgrade_page_view",
        "/api/track/upgrade_interest",
        "/admin/pro_breakdown",
        "/admin/waitlist",
        "/admin/dashboard",
        "/admin/activity",
    ):
        assert route not in source
    assert "_resolve_optional_user_id" not in source
    assert "[UPGRADE_INTEREST]" not in source
    assert "email=%r" not in source


def test_removed_track_routes_are_not_left_in_manual_verifier():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "/api/track/upgrade_page_view" not in source
    assert "/api/track/upgrade_interest" not in source


def test_formal_admin_replacements_remain():
    source = BACKEND.read_text(encoding="utf-8")
    assert '@app.get("/admin/users")' in source
    assert '@app.get("/admin/practice-volume")' in source
    assert '@app.patch("/admin/user/{user_id}/pro_grant")' in source


def test_waitlist_exposure_migration_is_narrow_and_defensive():
    sql = MIGRATION.read_text(encoding="utf-8")
    assert sql.count("DROP POLICY IF EXISTS") == 3
    assert "allow_anon_insert" in sql
    assert "anon_insert_upgrade_intent" in sql
    assert "authenticated_insert_upgrade_intent" in sql
    assert "REVOKE INSERT ON TABLE public.upgrade_intent FROM anon" in sql
    assert "to_regclass('public.upgrade_intent')" in sql
    assert "DROP TABLE" not in sql
    assert "DELETE FROM" not in sql
    assert "TRUNCATE" not in sql
    for unrelated in ("subscriptions", "payment_events", "gmail_", "omg_", "npc_"):
        assert unrelated not in sql
