"""Retention Phase 2 evidence selection, conservatism, and tenant safety."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import main


def _row(record_id, user_id, tag, transcript, created_at, **extra):
    return {
        "id": record_id,
        "user_id": user_id,
        "weakness_tag": tag,
        "user_transcript": transcript,
        "created_at": created_at,
        "resolved": False,
        "better_expression": None,
        **extra,
    }


class _Store:
    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]
        self.filters = []
        self.limit_value = None

    def table(self, name):
        assert name == "practice_records"
        self.filters = []
        self.limit_value = None
        return self

    def select(self, _columns):
        return self

    def eq(self, column, value):
        self.filters.append((column, value))
        return self

    def in_(self, column, values):
        self.filters.append((column, set(values)))
        return self

    def order(self, _column, desc=False):
        self.rows.sort(key=lambda row: row.get("created_at", ""), reverse=desc)
        return self

    def limit(self, value):
        self.limit_value = value
        return self

    def execute(self):
        matches = [
            row for row in self.rows
            if all(
                row.get(column) in expected if isinstance(expected, set)
                else row.get(column) == expected
                for column, expected in self.filters
            )
        ]
        if self.limit_value is not None:
            matches = matches[:self.limit_value]
        return SimpleNamespace(data=[dict(row) for row in matches])


def _run(awaitable):
    return asyncio.run(awaitable)


def test_same_user_same_weakness_selects_ordered_usable_pair():
    rows = [
        _row("later", "user-a", "lack_detail", "I like it because it helps me relax after work.", "2026-08-12T10:00:00Z"),
        _row("other-user", "user-b", "lack_detail", "For example I visit a quiet park near home every Sunday.", "2026-08-13T10:00:00Z"),
        _row("blank", "user-a", "lack_detail", " ", "2026-08-11T10:00:00Z"),
        _row("invalid", "user-a", "future_tag", "This invalid tag should never affect evidence selection.", "2026-08-14T10:00:00Z"),
        _row("earlier", "user-a", "lack_detail", "I think it is good for me.", "2026-08-10T10:00:00Z"),
    ]
    store = _Store(rows)
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", return_value="user-a"), \
         patch.object(main, "supabase_admin", store):
        result = _run(main.get_progress_evidence(MagicMock(), "Bearer token"))

    assert result["has_evidence"] is True
    assert result["weakness"]["tag"] == "lack_detail"
    assert result["before"]["record_id"] == "earlier"
    assert result["after"]["record_id"] == "later"
    assert result["before"]["created_at"] < result["after"]["created_at"]
    assert "other-user" not in str(result)


def test_only_one_usable_record_is_insufficient():
    result = main._build_progress_evidence([
        _row("one", "user-a", "lack_detail", "I enjoy the park near my home.", "2026-08-10T10:00:00Z"),
        _row("short", "user-a", "lack_detail", "Too short", "2026-08-12T10:00:00Z"),
    ])
    assert result == {"has_evidence": False, "reason": "insufficient_evidence"}


def test_current_unresolved_focus_outranks_resolved_and_recurrent_tags():
    rows = [
        _row("v1", "user-a", "weak_vocab", "It was a very good experience for me.", "2026-08-01T10:00:00Z"),
        _row("v2", "user-a", "weak_vocab", "The trip was a very good experience overall.", "2026-08-02T10:00:00Z"),
        _row("v3", "user-a", "weak_vocab", "I had a very good time with friends.", "2026-08-03T10:00:00Z", resolved=True),
        _row("d1", "user-a", "lack_detail", "I like visiting parks near my home.", "2026-08-04T10:00:00Z"),
        _row("d2", "user-a", "lack_detail", "I like parks because they help me relax.", "2026-08-05T10:00:00Z"),
    ]
    result = main._build_progress_evidence(rows)
    assert result["weakness"]["tag"] == "lack_detail"


def test_lack_detail_requires_new_reason_or_example_not_length_alone():
    before = _row("a", "user-a", "lack_detail", "I enjoy this local park with friends.", "2026-08-01")
    longer = _row("b", "user-a", "lack_detail", "I enjoy this beautiful local park with several close friends every weekend.", "2026-08-02")
    reason = _row("c", "user-a", "lack_detail", "I enjoy this park because it is quiet after work.", "2026-08-03")
    assert main._progress_observation("lack_detail", before, longer)["status"] == "still_working"
    assert main._progress_observation("lack_detail", before, reason)["status"] == "improvement_observed"


def test_weak_vocab_requires_active_use_of_stored_taught_expression():
    before = _row(
        "a", "user-a", "weak_vocab", "The beach was very good for our family.", "2026-08-01",
        better_expression="peaceful and restorative",
    )
    used = _row("b", "user-a", "weak_vocab", "The beach felt peaceful and restorative after a busy week.", "2026-08-02")
    unused = _row("c", "user-a", "weak_vocab", "The beach was extremely good for our whole family.", "2026-08-03")
    assert main._progress_observation("weak_vocab", before, used)["status"] == "improvement_observed"
    assert main._progress_observation("weak_vocab", before, unused)["status"] == "still_working"


@pytest.mark.parametrize("tag", ["safe_answer", "grammar_minor", "off_topic"])
def test_unsupported_tags_never_claim_improvement(tag):
    before = _row("a", "user-a", tag, "This is my first complete answer today.", "2026-08-01")
    after = _row("b", "user-a", tag, "This answer is much longer and contains many extra words today.", "2026-08-02")
    assert main._progress_observation(tag, before, after)["status"] == "still_working"


def test_resolved_pair_is_historical_evidence_without_mastery_claim():
    rows = [
        _row("a", "user-a", "grammar_minor", "I very like walking near my home.", "2026-08-01"),
        _row("b", "user-a", "grammar_minor", "I like walking near my home every evening.", "2026-08-02", resolved=True),
    ]
    result = main._build_progress_evidence(rows)
    assert result["observation"]["status"] == "evidence_available"
    assert "已標記為解決" in result["observation"]["label"]
    assert "%" not in str(result)


def test_response_contains_short_snippets_not_full_transcript():
    long_text = " ".join(["specific"] * 100)
    result = main._build_progress_evidence([
        _row("a", "user-a", "lack_detail", long_text, "2026-08-01"),
        _row("b", "user-a", "lack_detail", long_text + " because it matters", "2026-08-02"),
    ])
    assert result["has_evidence"] is True
    assert len(result["before"]["snippet"]) <= main.PROGRESS_EVIDENCE_SNIPPET_CHARS + 1
    assert result["before"]["snippet"] != long_text
    assert "user_transcript" not in str(result)


def test_no_auth_rejected_before_database_access():
    store = MagicMock()
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", side_effect=HTTPException(status_code=401)), \
         patch.object(main, "supabase_admin", store), \
         pytest.raises(HTTPException) as excinfo:
        _run(main.get_progress_evidence(MagicMock(), None))
    assert excinfo.value.status_code == 401
    store.table.assert_not_called()


def test_endpoint_is_owner_scoped_and_has_no_llm_call():
    source = Path(main.__file__).read_text(encoding="utf-8")
    block = source.split('def get_progress_evidence(', 1)[1].split('@app.get("/api/diagnosis/timeline")', 1)[0]
    assert '.eq("user_id", user_id)' in block
    assert 'client.' not in block
    assert 'Groq' not in block
    assert 'OpenAI' not in block
