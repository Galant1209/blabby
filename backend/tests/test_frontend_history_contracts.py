"""Regression contracts for the History page's three independent history feeds."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


APP = Path(__file__).parents[2] / "frontend" / "app"
HARNESS = Path(__file__).with_name("frontend_history_behavior.mjs")


def _source() -> str:
    return (APP / "history.html").read_text(encoding="utf-8")


def test_history_has_speaking_writing_and_reading_sections():
    src = _source()
    for marker in (
        'id="history-list"',
        'id="writing-history-list"',
        'id="reading-history-list"',
        "/api/writing/history?limit=10",
        "/reading/history?limit=20",
    ):
        assert marker in src


def test_history_uses_authenticated_api_requests_without_protected_table_queries():
    src = _source()
    assert "headers: { 'Authorization': `Bearer ${token}` }" in src
    assert "supabase.from" not in src.lower()
    for protected_table in ("writing_submissions", "reading_attempts", "reading_questions"):
        assert protected_table not in src


def test_history_contracts_real_response_fields_and_safe_rendering():
    src = _source()
    for field in (
        "submissions",
        "attempts",
        "task_type",
        "band_overall",
        "priority_fix",
        "word_count",
        "passage_title",
        "score",
        "total",
        "band_estimate",
        "submitted_at",
    ):
        assert field in src
    assert "Promise.allSettled" in src
    assert "replaceChildren" in src
    assert "textContent = String(value)" in src
    assert "encodeURIComponent" in src


@pytest.mark.parametrize(
    "status_marker",
    ("error.status === 401", "error.status === 403", "error.status === 404", "error.status >= 500"),
)
def test_history_has_explicit_http_failure_contracts(status_marker: str):
    assert status_marker in _source()
    assert "Network error. Please try again." in _source()


def test_history_partial_failure_behavior_harness():
    if not HARNESS.exists():
        pytest.fail(f"missing harness: {HARNESS}")
    result = subprocess.run(
        ["node", str(HARNESS)],
        cwd=str(APP.parent.parent),
        capture_output=True,
        text=True,
        timeout=60,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    assert result.returncode == 0, (
        f"history behavior harness failed (exit {result.returncode}):\n"
        f"{result.stdout}\n{result.stderr}"
    )
