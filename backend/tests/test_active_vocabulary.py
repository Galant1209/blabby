import asyncio
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import main


NOW = datetime(2026, 8, 13, 8, 0, tzinfo=timezone.utc)


def _vocab(
    row_id: str,
    word: str,
    *,
    review_count: int = 1,
    last_reviewed_at: str = "2026-08-12T08:00:00Z",
    next_review_at: str = "2026-08-13T07:00:00Z",
    created_at: str = "2026-08-01T08:00:00Z",
    topic: str = "travel",
):
    return {
        "id": row_id,
        "vocabulary_item_id": f"catalog-{row_id}",
        "review_count": review_count,
        "last_reviewed_at": last_reviewed_at,
        "next_review_at": next_review_at,
        "created_at": created_at,
        "vocabulary_items": {"word": word, "topic": topic},
    }


def test_reviewed_unused_item_is_candidate_and_unreviewed_is_not():
    result = main._select_active_vocab_candidate(
        [
            _vocab("unreviewed", "picturesque", review_count=0, last_reviewed_at=""),
            _vocab("reviewed", "overwhelming"),
        ],
        ["I had a very busy day."],
        NOW,
    )
    assert result == {
        "id": "reviewed",
        "word": "overwhelming",
        "topic": "travel",
        "review_count": 1,
        "active_use_observed": False,
    }


def test_historical_active_use_excludes_item_across_case_and_punctuation():
    result = main._select_active_vocab_candidate(
        [_vocab("used", "overwhelming")],
        ["That first week was OVERWHELMING!"],
        NOW,
    )
    assert result is None


def test_candidate_ranking_is_deterministic_due_recent_then_complexity_and_id():
    rows = [
        _vocab("later", "picturesque", last_reviewed_at="2026-08-11T08:00:00Z"),
        _vocab("chosen", "vibrant", last_reviewed_at="2026-08-12T08:00:00Z"),
        _vocab(
            "not-due",
            "bustling",
            last_reviewed_at="2026-08-13T07:30:00Z",
            next_review_at="2026-08-14T08:00:00Z",
        ),
    ]
    forward = main._select_active_vocab_candidate(rows, [], NOW)
    reverse = main._select_active_vocab_candidate(list(reversed(rows)), [], NOW)
    assert forward["id"] == reverse["id"] == "chosen"


def test_phrase_match_and_unsupported_morphology_are_conservative():
    assert main._contains_lexical_expression(
        "It gives us a wide range of practical choices.",
        "a wide range of",
    ) is True
    assert main._contains_lexical_expression("I felt overwhelmed.", "overwhelm") is False
    assert main._contains_lexical_expression("I can express myself.", "press") is False


class _Query:
    def __init__(self, response):
        self.response = response
        self.filters = []
        self.not_ = self

    def select(self, *args, **kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def gt(self, *args):
        return self

    def is_(self, *args):
        return self

    def order(self, *args, **kwargs):
        return self

    def limit(self, *args):
        return self

    def maybe_single(self):
        return self

    def execute(self):
        return self.response(self.filters) if callable(self.response) else self.response


class _ObservationStore:
    def __init__(self, owner: str, row: dict):
        self.owner = owner
        self.row = row
        self.query = None

    def table(self, name):
        assert name == "user_vocabulary"

        def response(filters):
            requested_owner = dict(filters).get("user_id")
            return SimpleNamespace(data=self.row if requested_owner == self.owner else None)

        self.query = _Query(response)
        return self.query


class _QueuedStore:
    def __init__(self, responses):
        self.responses = {name: list(items) for name, items in responses.items()}
        self.queries = []

    def table(self, name):
        response = self.responses[name].pop(0)
        query = _Query(response)
        self.queries.append((name, query))
        return query


def test_process_observation_requires_persistence_review_and_owner_scope():
    target_id = "11111111-1111-4111-8111-111111111111"
    row = {
        "id": target_id,
        "review_count": 2,
        "last_reviewed_at": "2026-08-12T08:00:00Z",
        "vocabulary_items": {"word": "overwhelming"},
    }
    store = _ObservationStore("user-a", row)
    with patch.object(main, "supabase_admin", store):
        assert main._active_vocab_process_observation(
            "user-a", target_id, "It was overwhelming.", False,
        ) is None
        assert main._active_vocab_process_observation(
            "user-b", target_id, "It was overwhelming.", True,
        ) is None
        assert main._active_vocab_process_observation(
            "user-a", target_id, "It was overwhelming.", True,
        ) == {"item_id": target_id, "active_use_observed": True}
    assert ("user_id", "user-a") in store.query.filters


def test_endpoint_rejects_anonymous_before_database_access():
    store = MagicMock()
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", side_effect=HTTPException(status_code=401)), \
         patch.object(main, "supabase_admin", store), \
         pytest.raises(HTTPException) as excinfo:
        asyncio.run(main.vocabulary_active_use_current(MagicMock(), None))
    assert excinfo.value.status_code == 401
    store.table.assert_not_called()


def test_endpoint_checks_complete_owner_history_and_returns_no_transcript():
    target_id = "11111111-1111-4111-8111-111111111111"
    store = _QueuedStore({
        "user_vocabulary": [SimpleNamespace(data=[_vocab(target_id, "overwhelming")])],
        "practice_records": [
            SimpleNamespace(data=[{"id": "only"}], count=1),
            SimpleNamespace(data=[{"user_transcript": "It was an intense first week."}]),
        ],
    })
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", return_value="user-a"), \
         patch.object(main, "supabase_admin", store):
        result = asyncio.run(main.vocabulary_active_use_current(MagicMock(), "Bearer token"))
    assert result["has_target"] is True
    assert result["target"]["id"] == target_id
    assert "transcript" not in str(result).lower()
    assert all(("user_id", "user-a") in query.filters for _, query in store.queries)


def test_endpoint_refuses_unbounded_history_without_querying_transcripts():
    target_id = "22222222-2222-4222-8222-222222222222"
    store = _QueuedStore({
        "user_vocabulary": [SimpleNamespace(data=[_vocab(target_id, "picturesque")])],
        "practice_records": [
            SimpleNamespace(data=[{"id": "sample"}], count=main.ACTIVE_VOCAB_MAX_HISTORY_RECORDS + 1),
        ],
    })
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", return_value="user-a"), \
         patch.object(main, "supabase_admin", store):
        result = asyncio.run(main.vocabulary_active_use_current(MagicMock(), "Bearer token"))
    assert result == {"has_target": False, "reason": "history_safety_bound"}
    assert len(store.queries) == 2


def test_endpoint_contract_is_owner_scoped_bounded_and_never_returns_transcripts():
    source = Path(main.__file__).read_text(encoding="utf-8")
    block = source.split('async def vocabulary_active_use_current(', 1)[1].split(
        '@app.get("/api/vocabulary/items")', 1,
    )[0]
    assert block.count('.eq("user_id", user_id)') >= 3
    assert "ACTIVE_VOCAB_MAX_HISTORY_RECORDS" in block
    assert '"user_transcript":' not in block
    assert "Groq" not in block
    assert "client." not in block
