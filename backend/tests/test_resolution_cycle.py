"""Weakness resolution suggestions stay conservative, derived, and user-owned."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import main


def _row(record_id, tag, transcript, day, **extra):
    return {
        "id": record_id,
        "user_id": "user-a",
        "weakness_tag": tag,
        "user_transcript": transcript,
        "better_expression": None,
        "resolved": False,
        "mode": "normal",
        "created_at": f"2026-08-{day:02d}T10:00:00Z",
        **extra,
    }


def _lack_detail_history():
    return [
        _row("d1", "lack_detail", "I like this park near my home.", 1),
        _row("d2", "lack_detail", "I like this park because it is quiet after work.", 2),
        _row("o1", "safe_answer", "I prefer walking in the evening after work.", 3),
        _row("o2", "grammar_minor", "I usually visit it with my closest friend.", 4),
        _row("o3", "off_topic", "The question reminds me of my local area.", 5),
    ]


def test_lack_detail_requires_two_occurrences_three_clean_opportunities_and_improvement():
    rows = _lack_detail_history()
    candidate = main._resolution_candidate(rows)
    assert candidate["has_candidate"] is True
    assert candidate["candidate"]["record_id"] == "d2"
    assert candidate["candidate"]["weakness_tag"] == "lack_detail"
    assert candidate["candidate"]["observation"]["status"] == "improvement_observed"
    assert candidate["candidate"]["evidence"] == {
        "prior_occurrences": 2,
        "recent_opportunities": 3,
        "recent_recurrences": 0,
    }
    assert "user_transcript" not in str(candidate)


@pytest.mark.parametrize("cut", [1, 2, 3, 4])
def test_insufficient_occurrences_or_clean_opportunities_never_suggests(cut):
    result = main._resolution_candidate(_lack_detail_history()[:cut])
    assert result["has_candidate"] is False


def test_recent_recurrence_restarts_the_three_opportunity_window():
    rows = _lack_detail_history()
    rows.append(_row("d3", "lack_detail", "For example I go there each Sunday.", 6))
    rows.extend([
        _row("n1", "safe_answer", "I would choose the park.", 7),
        _row("n2", "grammar_minor", "I visit it every weekend.", 8),
    ])
    assert main._resolution_candidate(rows)["has_candidate"] is False


def test_weak_vocab_requires_repeated_tag_and_taught_expression_active_use():
    base = [
        _row(
            "v1", "weak_vocab", "The beach was very good for our family.", 1,
            better_expression="peaceful and restorative",
        ),
        _row("v2", "weak_vocab", "The beach felt peaceful and restorative after work.", 2),
        _row("n1", "safe_answer", "I would definitely return there.", 3),
        _row("n2", "grammar_minor", "My family visits every summer.", 4),
        _row("n3", "off_topic", "It is near the southern coast.", 5),
    ]
    assert main._resolution_candidate(base)["candidate"]["weakness_tag"] == "weak_vocab"

    base[1]["user_transcript"] = "The beach was extremely good for our family."
    assert main._resolution_candidate(base)["has_candidate"] is False
    assert main._resolution_candidate(base[:1])["has_candidate"] is False


@pytest.mark.parametrize("tag", ["safe_answer", "grammar_minor", "off_topic"])
def test_manual_only_taxonomy_never_becomes_candidate(tag):
    rows = [_row("a", tag, "This is my first complete answer today.", 1),
            _row("b", tag, "This is a much longer complete answer today.", 2),
            _row("c", "lack_detail", "I answer because it matters.", 3),
            _row("d", "weak_vocab", "I use a precise phrase here.", 4),
            _row("e", "grammar_minor", "I finish the answer clearly.", 5)]
    result = main._resolution_candidate(rows)
    assert not result.get("has_candidate") or result["candidate"]["weakness_tag"] != tag


def test_drill_and_invalid_rows_are_not_clean_opportunities():
    rows = _lack_detail_history()[:2]
    rows.extend([
        _row("drill", "weak_vocab", "A drill answer with precise language.", 3, mode="drill"),
        _row("invalid", "future_tag", "An invalid classifier outcome.", 4),
        _row("one", "safe_answer", "Only one real opportunity remains.", 5),
    ])
    assert main._resolution_candidate(rows)["has_candidate"] is False


def test_resolved_latest_occurrence_is_not_suggested():
    rows = _lack_detail_history()
    rows[1]["resolved"] = True
    assert main._resolution_candidate(rows)["has_candidate"] is False


def test_new_unresolved_same_tag_is_observed_as_recurrence_without_mutation():
    query = MagicMock()
    query.select.return_value = query
    query.eq.return_value = query
    query.neq.return_value = query
    query.limit.return_value = query
    query.execute.return_value = SimpleNamespace(data=[{"id": "previous-resolved"}])
    store = MagicMock()
    store.table.return_value = query
    with patch.object(main, "supabase_admin", store):
        assert main._resolved_weakness_recurred(
            "user-a", "lack_detail", "new-unresolved"
        ) is True
    query.eq.assert_any_call("user_id", "user-a")
    query.eq.assert_any_call("weakness_tag", "lack_detail")
    query.eq.assert_any_call("resolved", True)
    query.eq.assert_any_call("mode", "normal")
    query.neq.assert_called_once_with("id", "new-unresolved")
    query.update.assert_not_called()


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
        rows = [
            row for row in self.rows
            if all(
                row.get(column) in expected if isinstance(expected, set)
                else row.get(column) == expected
                for column, expected in self.filters
            )
        ]
        if self.limit_value is not None:
            rows = rows[:self.limit_value]
        return SimpleNamespace(data=[dict(row) for row in rows])


def _run(awaitable):
    return asyncio.run(awaitable)


def test_candidate_endpoint_is_authenticated_owner_scoped_and_redacted():
    store = _Store(_lack_detail_history() + [
        {**_lack_detail_history()[0], "id": "other", "user_id": "user-b"},
    ])
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", return_value="user-a"), \
         patch.object(main, "supabase_admin", store):
        result = _run(main.get_resolution_candidate(MagicMock(), "Bearer token"))
    assert result["has_candidate"] is True
    assert "user_transcript" not in str(result)
    assert "better_expression" not in str(result)


def test_candidate_endpoint_rejects_anonymous_before_database_access():
    store = MagicMock()
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", side_effect=HTTPException(status_code=401)), \
         patch.object(main, "supabase_admin", store), \
         pytest.raises(HTTPException) as excinfo:
        _run(main.get_resolution_candidate(MagicMock(), None))
    assert excinfo.value.status_code == 401
    store.table.assert_not_called()


def test_contract_has_no_auto_resolution_or_candidate_llm_call():
    source = Path(main.__file__).read_text(encoding="utf-8")
    assert "auto-resolved prior practice_record" not in source
    assert "force-resolved after" not in source
    block = source.split("def get_resolution_candidate(", 1)[1].split(
        '@app.get("/api/diagnosis/timeline")', 1
    )[0]
    assert '.eq("user_id", user_id)' in block
    assert "client." not in block
    assert "Groq" not in block
    assert "OpenAI" not in block
