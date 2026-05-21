"""
Unit tests for GET /api/progress.

Mocks supabase_admin and verify_token so the endpoint's grouping +
delta logic can be exercised without hitting Supabase or the auth
service. Mirrors the pure-function style of test_reading_validators.py
(no env vars, no live HTTP).

Skipped cleanly if main.py's runtime deps (fastapi, anthropic, supabase)
aren't installed locally — happens in barebones dev venvs that only
have pytest. In production / CI / Render shell, all deps are present
and these tests run normally.
"""

from __future__ import annotations

import asyncio

import pytest

# Skip the entire module if any required runtime dep is missing.
pytest.importorskip("fastapi")
pytest.importorskip("anthropic")
pytest.importorskip("supabase")

from unittest.mock import MagicMock, patch

from fastapi import HTTPException

import main  # noqa: E402  (must follow importorskip)


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────

def _make_request():
    """Minimal FastAPI Request stub. get_progress only passes it to
    the slowapi limiter decorator's introspection; the body and headers
    are read from the explicit `authorization` parameter instead."""
    return MagicMock()


def _mock_supabase_chain(records: list[dict]) -> MagicMock:
    """
    Build a chained MagicMock that mirrors:

        supabase_admin.table("practice_records")
                      .select(...)
                      .eq("user_id", user_id)
                      .order("created_at", desc=False)
                      .execute()

    The terminal .execute() returns a MagicMock whose .data is `records`.
    Each intermediate call returns the same chain so .table().select()...
    fluent chaining works regardless of order.
    """
    chain = MagicMock()
    chain.table.return_value = chain
    chain.select.return_value = chain
    chain.eq.return_value = chain
    chain.order.return_value = chain
    chain.execute.return_value = MagicMock(data=records)
    return chain


def _run(coro):
    """Run an async coroutine to completion. Used instead of pytest-asyncio
    so the test file has zero extra plugin requirements."""
    return asyncio.run(coro)


# ────────────────────────────────────────────────────────────────────────────
#  Happy path
# ────────────────────────────────────────────────────────────────────────────

def test_progress_happy_path_two_records_same_question():
    """
    Two attempts at the same question → one item with correct deltas.
    First: 5 words, 2 sentences. Latest: 14 words, 3 sentences.
    Expect word_delta = +9, sentence_delta = +1.
    """
    records = [
        {
            "question": "Describe your hometown.",
            "user_transcript": "I from Taipei. Good food.",
            "created_at": "2026-05-10T10:00:00Z",
        },
        {
            "question": "Describe your hometown.",
            "user_transcript": (
                "I'm from Taipei. The food is really good. "
                "Also the night markets are great."
            ),
            "created_at": "2026-05-20T10:00:00Z",
        },
    ]
    fake_sb = _mock_supabase_chain(records)

    with patch.object(main, "verify_token", return_value="user-abc"), \
         patch.object(main, "supabase_admin", fake_sb):
        result = _run(main.get_progress(_make_request(), "Bearer fake"))

    assert isinstance(result, list)
    assert len(result) == 1

    item = result[0]
    assert item["question"] == "Describe your hometown."
    assert item["first"]["created_at"] == "2026-05-10T10:00:00Z"
    assert item["latest"]["created_at"] == "2026-05-20T10:00:00Z"
    assert item["first"]["transcript"] == "I from Taipei. Good food."
    assert "night markets" in item["latest"]["transcript"]
    assert item["delta"]["word_delta"] == 14 - 5      # = 9
    assert item["delta"]["sentence_delta"] == 3 - 2   # = 1


# ────────────────────────────────────────────────────────────────────────────
#  Skip single
# ────────────────────────────────────────────────────────────────────────────

def test_progress_skips_questions_with_single_attempt():
    """
    A question with only one record → no comparison possible → excluded
    from the response. Endpoint returns an empty array.
    """
    records = [
        {
            "question": "Tell me about your studies.",
            "user_transcript": "I study English here.",
            "created_at": "2026-05-10T10:00:00Z",
        },
    ]
    fake_sb = _mock_supabase_chain(records)

    with patch.object(main, "verify_token", return_value="user-abc"), \
         patch.object(main, "supabase_admin", fake_sb):
        result = _run(main.get_progress(_make_request(), "Bearer fake"))

    assert result == []


# ────────────────────────────────────────────────────────────────────────────
#  Auth fail
# ────────────────────────────────────────────────────────────────────────────

def test_progress_rejects_missing_authorization_header():
    """
    No Authorization header → real verify_token raises 401 before any
    DB call is made. supabase_admin must be truthy so verify_token's
    earlier "Auth service not configured" 503 branch doesn't fire first.
    """
    with patch.object(main, "supabase_admin", MagicMock()), \
         pytest.raises(HTTPException) as excinfo:
        _run(main.get_progress(_make_request(), authorization=None))

    assert excinfo.value.status_code == 401
