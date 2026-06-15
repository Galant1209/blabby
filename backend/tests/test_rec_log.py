"""
Happy-path tests for the temporary /api/debug/rec-log observability endpoint.

The endpoint only writes the payload to the server log so we can see what
iOS Safari actually records (mime/size/ua) without a physical iPad. It must
never fail the caller — even a malformed body returns ok.
"""

from __future__ import annotations

import asyncio

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("anthropic")
pytest.importorskip("supabase")

from unittest.mock import AsyncMock, MagicMock, patch

import main  # noqa: E402


def _call(request):
    # limiter.enabled=False bypasses slowapi's "request must be a real
    # starlette.Request" check, same pattern as test_process_question_guard.
    with patch.object(main.limiter, "enabled", False):
        return asyncio.run(main.debug_rec_log(request))


def test_rec_log_happy_path():
    request = MagicMock()
    request.json = AsyncMock(return_value={
        "part": 1,
        "mime": "audio/mp4",
        "ext": "m4a",
        "size": 52341,
        "status": 200,
        "error": None,
        "ua": "Mozilla/5.0 (iPad; CPU OS 17_0 like Mac OS X)",
    })
    assert _call(request) == {"ok": True}


def test_rec_log_malformed_body_still_ok():
    request = MagicMock()
    request.json = AsyncMock(side_effect=ValueError("not json"))
    assert _call(request) == {"ok": True}
