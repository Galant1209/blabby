"""
Regression tests for the /process question guard.

Reproduces the production 422 seen in logs: a normal (non-drill) submit
with an empty `question` field is rejected before Whisper/Claude run.
Confirms the inverse — a non-empty question clears the guard — and that
drill mode is exempt.

Mocks verify_token, supabase_admin, and get_user_pro_status so the guard
logic runs without auth, DB, or LLM. Each test stops the request at the
"no audio" 400 once the question guard passes, which is enough to prove
the question gate's behaviour without exercising transcription.

Skipped cleanly if main.py's runtime deps aren't installed locally.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("anthropic")
pytest.importorskip("supabase")

from unittest.mock import MagicMock, patch

from fastapi import HTTPException

import main  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


def _call_process(**overrides):
    """Invoke main.process with sane defaults; overrides win.

    verify_token and get_user_pro_status are patched so the guard runs
    without auth/DB. Pro=True skips the monthly-quota DB query entirely,
    isolating the question guard.
    """
    kwargs = dict(
        request=MagicMock(),
        audio=None,
        level="Band 5",
        topic="Free Time",
        question="",
        history="[]",
        text_override="",
        dev_bypass_secret="",
        mode="",
        drill_tag="",
        previous_transcript="",
        retry_of="",
        authorization="Bearer fake",
    )
    kwargs.update(overrides)
    # limiter.enabled=False bypasses slowapi's "request must be a real
    # starlette.Request" check so we can drive process() with a MagicMock
    # request. get_user_recent_records is mocked to [] so the valid-question
    # path never touches Supabase; it then stops at the "no audio" 400.
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", return_value="user-abc"), \
         patch.object(main, "get_user_pro_status", return_value=True), \
         patch.object(main, "get_user_recent_records", return_value=[]):
        return _run(main.process(**kwargs))


def test_empty_question_normal_mode_returns_422():
    """Non-drill submit with blank question → 422 (the production bug)."""
    with pytest.raises(HTTPException) as excinfo:
        _call_process(question="")
    assert excinfo.value.status_code == 422


def test_whitespace_question_normal_mode_returns_422():
    """A question of only whitespace is treated as empty → 422."""
    with pytest.raises(HTTPException) as excinfo:
        _call_process(question="   ")
    assert excinfo.value.status_code == 422


def test_valid_question_passes_guard():
    """A real question clears the guard; it then fails later at the
    'no audio' 400, proving the question gate did NOT fire."""
    with pytest.raises(HTTPException) as excinfo:
        _call_process(question="What do you usually do in your free time?")
    assert excinfo.value.status_code == 400


def test_drill_mode_exempt_from_question_guard():
    """Drill mode skips the question requirement. With an empty question
    and an invalid drill_tag, the drill-tag validation 422 fires — NOT
    the question-guard 422 — confirming the question gate was bypassed."""
    with pytest.raises(HTTPException) as excinfo:
        _call_process(question="", mode="drill", drill_tag="__not_a_real_tag__")
    assert excinfo.value.status_code == 422
    detail = excinfo.value.detail
    assert "drill_tag" in str(detail)
