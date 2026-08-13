"""Focused contracts for Retention Phase 1 current-focus truth and isolation."""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import main


class _PracticeRecords:
    def __init__(self, rows):
        self.rows = [dict(row) for row in rows]
        self.filters = []
        self.limit_value = None
        self.update_payload = None

    def table(self, name):
        assert name == "practice_records"
        self.filters = []
        self.limit_value = None
        self.update_payload = None
        return self

    def select(self, _columns):
        return self

    def update(self, payload):
        self.update_payload = dict(payload)
        return self

    def eq(self, column, value):
        self.filters.append((column, value))
        return self

    def in_(self, column, values):
        allowed = set(values)
        self.filters.append((column, allowed))
        return self

    def order(self, _column, desc=False):
        if desc:
            self.rows.sort(key=lambda row: row.get("created_at", ""), reverse=True)
        return self

    def limit(self, value):
        self.limit_value = value
        return self

    def execute(self):
        matches = [
            row for row in self.rows
            if all(
                row.get(column) in value if isinstance(value, set) else row.get(column) == value
                for column, value in self.filters
            )
        ]
        if self.update_payload is not None:
            for row in matches:
                row.update(self.update_payload)
            return SimpleNamespace(data=matches)
        if self.limit_value is not None:
            matches = matches[:self.limit_value]
        return SimpleNamespace(data=[dict(row) for row in matches])


def _run(awaitable):
    return asyncio.run(awaitable)


def test_last_unresolved_returns_only_latest_focus_for_authenticated_user():
    store = _PracticeRecords([
        {"id": "old", "user_id": "user-a", "weakness_tag": "weak_vocab", "resolved": False, "created_at": "2026-08-10"},
        {"id": "new", "user_id": "user-a", "weakness_tag": "lack_detail", "resolved": False, "created_at": "2026-08-12"},
        {"id": "other", "user_id": "user-b", "weakness_tag": "off_topic", "resolved": False, "created_at": "2026-08-13"},
        {"id": "blank", "user_id": "user-a", "weakness_tag": "", "resolved": False, "created_at": "2026-08-14"},
    ])
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", return_value="user-a"), \
         patch.object(main, "supabase_admin", store):
        result = _run(main.last_unresolved_practice_record(MagicMock(), "Bearer user-a"))
    assert result["id"] == "new"
    assert result["weakness_tag"] == "lack_detail"
    assert result["resolved"] is False
    assert result["user_id"] == "user-a"  # fake row proves tenant selection


def test_resolved_focus_disappears_and_next_unresolved_is_selected():
    store = _PracticeRecords([
        {"id": "11111111-1111-4111-8111-111111111111", "user_id": "user-a", "weakness_tag": "lack_detail", "resolved": False, "created_at": "2026-08-12"},
        {"id": "22222222-2222-4222-8222-222222222222", "user_id": "user-a", "weakness_tag": "weak_vocab", "resolved": False, "created_at": "2026-08-11"},
    ])
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", return_value="user-a"), \
         patch.object(main, "supabase_admin", store):
        response = _run(main.resolve_practice_record(
            MagicMock(), "11111111-1111-4111-8111-111111111111", "Bearer user-a"
        ))
        next_focus = _run(main.last_unresolved_practice_record(MagicMock(), "Bearer user-a"))
    assert response.status_code == 204
    assert next_focus["id"] == "22222222-2222-4222-8222-222222222222"


def test_user_cannot_resolve_another_users_focus():
    store = _PracticeRecords([
        {"id": "33333333-3333-4333-8333-333333333333", "user_id": "user-b", "weakness_tag": "off_topic", "resolved": False, "created_at": "2026-08-13"},
    ])
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", return_value="user-a"), \
         patch.object(main, "supabase_admin", store), \
         pytest.raises(HTTPException) as excinfo:
        _run(main.resolve_practice_record(
            MagicMock(), "33333333-3333-4333-8333-333333333333", "Bearer user-a"
        ))
    assert excinfo.value.status_code == 403
    assert store.rows[0]["resolved"] is False


def test_last_unresolved_requires_auth_before_database_access():
    store = MagicMock()
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", side_effect=HTTPException(status_code=401)), \
         patch.object(main, "supabase_admin", store), \
         pytest.raises(HTTPException) as excinfo:
        _run(main.last_unresolved_practice_record(MagicMock(), None))
    assert excinfo.value.status_code == 401
    store.table.assert_not_called()


def test_resume_endpoint_does_not_select_transcript_or_coach_response():
    source = Path(main.__file__).read_text(encoding="utf-8")
    block = source.split('def last_unresolved_practice_record(', 1)[1].split('def _build_weakness_summary(', 1)[0]
    assert "user_transcript" not in block
    assert "coach_response" not in block
    assert '"id, question, topic, weakness_tag, resolved, created_at"' in block
    assert '.in_("weakness_tag", sorted(ALLOWED_WEAKNESS_TAGS))' in block
