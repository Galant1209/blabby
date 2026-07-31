"""
Regression tests for response_payload["memory_active"] (Repair Memory
frontend exposure, TASK A-implementation段).

memory_active must mirror repair_block — True only when repair_memory was
non-empty AND the caller is Pro (main.py:2213-2214). It must NOT be
confused with memory_snapshot, an unrelated weak-word/pattern detector
also present in the same response.

Runs the real process() endpoint end-to-end (Whisper bypassed via
text_override + DEV_BYPASS_SECRET) with a permissive fake Supabase client
and a mocked run_claude(), so the actual repair_block gating logic in
main.py executes unmodified.
"""

from __future__ import annotations

import asyncio
import os

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("anthropic")
pytest.importorskip("supabase")

from unittest.mock import MagicMock, patch

import main  # noqa: E402


def _run(coro):
    return asyncio.run(coro)


class _FakeResult:
    def __init__(self, data=None, count=0):
        self.data = data if data is not None else []
        self.count = count


class _FakeQuery:
    """Accepts any chained call (select/insert/update/eq/order/...);
    execute() always returns one row with an id so persistence code paths
    complete without raising. Not modelling real query semantics — this
    test only cares about response_payload, not what got written."""

    def insert(self, payload, *a, **k):
        return self

    def __getattr__(self, _name):
        def _chain(*a, **k):
            return self
        return _chain

    def execute(self):
        return _FakeResult(data=[{"id": "fake-record-id"}], count=0)


class _FakeSupabase:
    def table(self, _name):
        return _FakeQuery()


VALID_CORRECTION = {
    "correction": {
        "quoted": "I very like it",
        "why_it_hurts": "非正式且文法上不常見的強調方式。",
        "better_phrasing_en": "I really like it",
        "better_phrasing_zh": "我真的很喜歡",
        "next_task": "再說一次，把 very 換成 really。",
    },
    "on_topic": True,
    "tag": "grammar_minor",
    "progress_note": "",
}

QUALIFYING_RECORD = {
    "user_transcript": "I very like sports",
    "topic": "Hobbies",
    "question": "What do you usually do in your free time?",
    "created_at": "2026-07-01T00:00:00Z",
    "weakness_tag": "grammar_minor",
    "better_expression": "I really like sports",
    "coach_response": "你說：「I very like sports」\n\n非正式的強調方式。",
}


def _call_process(*, is_pro, recent_records, dev_secret="test-secret"):
    kwargs = dict(
        request=MagicMock(),
        audio=None,
        level="Band 5",
        topic="Hobbies",
        question="What do you usually do in your free time?",
        history="[]",
        text_override="I very like sports and I play them every weekend.",
        dev_bypass_secret=dev_secret,
        mode="",
        drill_tag="",
        previous_transcript="",
        retry_of="",
        authorization="Bearer fake",
    )
    with patch.dict(os.environ, {"DEV_BYPASS_SECRET": dev_secret}), \
         patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", return_value="user-abc"), \
         patch.object(main, "get_user_pro_status", return_value=is_pro), \
         patch.object(main, "get_user_recent_records", return_value=recent_records), \
         patch.object(main, "run_claude", return_value=dict(VALID_CORRECTION)), \
         patch.object(main, "supabase_admin", _FakeSupabase()):
        return _run(main.process(**kwargs))


def test_memory_active_true_when_pro_and_qualifying_history():
    result = _call_process(is_pro=True, recent_records=[QUALIFYING_RECORD])
    assert result["memory_active"] is True


def test_memory_active_false_when_pro_but_no_qualifying_history():
    """State 2: Pro, but repair_memory has nothing to inject."""
    result = _call_process(is_pro=True, recent_records=[])
    assert result["memory_active"] is False


def test_memory_active_false_when_free_despite_qualifying_history():
    """State 3: qualifying history exists, but caller is not Pro — gated."""
    result = _call_process(is_pro=False, recent_records=[QUALIFYING_RECORD])
    assert result["memory_active"] is False


def test_memory_active_is_distinct_from_memory_snapshot():
    """memory_snapshot (weak-word detector) must not be conflated with
    memory_active (repair memory injection signal) — both present,
    independently correct."""
    result = _call_process(is_pro=True, recent_records=[QUALIFYING_RECORD])
    assert "memory_snapshot" in result
    assert isinstance(result["memory_snapshot"], dict)
    assert "memory_active" in result
    assert isinstance(result["memory_active"], bool)
