"""
POST /reading/questions/generate must not ship the answer key.

The endpoint has two return paths and both used to hand the client whole
reading_questions rows — including correct_answer, explanation and
evidence_quote — before the user had answered anything:

  * idempotent replay (questions already exist for the passage)
  * fresh insert (PostgREST returns the full-row representation)

The 20260714 column-level grant cannot defend this path: the backend reads
these rows as service_role, so whatever it puts in a response never passes
through PostgREST. The response allowlist is the only gate, and this module
is what holds it in place.

Hermetic: supabase, verify_token, the band lookup and the LLM question
generator are all patched. No credentials, no network, no DB.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import main


ANSWER_FIELDS = ("correct_answer", "explanation", "evidence_quote")

PASSAGE_ID = "11111111-1111-1111-1111-111111111111"
USER_ID = "user-abc"


# ── fakes ────────────────────────────────────────────────────────────────
class _Resp:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    def __init__(self, owner, name):
        self._owner = owner
        self._name = name
        self._op = "select"

    def select(self, *a, **k):
        self._op = "select"
        return self

    def insert(self, rows):
        self._op = "insert"
        self._owner.inserted_payload = rows
        return self

    def eq(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def execute(self):
        return _Resp(self._owner.resolve(self._name, self._op))


class _FakeSupabase:
    def __init__(self, passage_rows, existing_questions, inserted_questions):
        self.passage_rows = passage_rows
        self.existing_questions = existing_questions
        self.inserted_questions = inserted_questions
        self.inserted_payload = None

    def table(self, name):
        return _FakeTable(self, name)

    def resolve(self, name, op):
        if name == "reading_passages":
            return self.passage_rows
        if name == "reading_questions":
            return self.existing_questions if op == "select" else self.inserted_questions
        raise AssertionError(f"unexpected table {name!r}")


def _passage_row():
    return [{
        "id": PASSAGE_ID,
        "body": "Paragraph one.\n\nParagraph two.",
        "difficulty_band": 6.5,
        "topic": "Urban planning",
        "created_by": USER_ID,
    }]


def _stored_row(i):
    """A row shaped like the idempotent branch's SELECT — answers included."""
    return {
        "id": f"q-{i}",
        "question_type": "mcq",
        "question_text": f"Question {i}?",
        "options": ["A", "B", "C", "D"],
        "correct_answer": "B",
        "explanation": f"Because of paragraph {i}.",
        "evidence_quote": f"quoted sentence {i}",
        "order_idx": i,
    }


def _inserted_row(i):
    """PostgREST full-row representation after INSERT — answers included."""
    row = _stored_row(i)
    row["passage_id"] = PASSAGE_ID
    row["created_at"] = "2026-07-25T00:00:00+00:00"
    return row


def _generated_questions():
    return {"questions": [
        {
            "question_type": "mcq",
            "question_text": f"Question {i}?",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "B",
            "explanation": f"Because of paragraph {i}.",
            "evidence_quote": f"quoted sentence {i}",
            "order_idx": i,
        }
        for i in range(1, main.READING_TOTAL_QUESTIONS + 1)
    ]}


def _call(fake_supabase):
    request = MagicMock()
    request.json = AsyncMock(return_value={"passage_id": PASSAGE_ID})
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", return_value=USER_ID), \
         patch.object(main, "supabase_admin", fake_supabase), \
         patch.object(main, "_get_user_band_reading", return_value=6.0), \
         patch.object(main, "_generate_questions_for_passage",
                      return_value=_generated_questions()):
        return asyncio.run(
            main.reading_generate_questions(request=request,
                                            authorization="Bearer fake")
        )


# ── the two branches ─────────────────────────────────────────────────────
def test_idempotent_branch_withholds_the_answer_key():
    stored = [_stored_row(i) for i in range(1, main.READING_TOTAL_QUESTIONS + 1)]
    # Guard against a vacuous test: the simulated DB rows really do carry answers.
    assert all(f in stored[0] for f in ANSWER_FIELDS)

    body = _call(_FakeSupabase(_passage_row(), stored, []))

    assert len(body["questions"]) == main.READING_TOTAL_QUESTIONS
    for q in body["questions"]:
        for field in ANSWER_FIELDS:
            assert field not in q, f"{field} leaked on the idempotent branch"


def test_fresh_insert_branch_withholds_the_answer_key():
    inserted = [_inserted_row(i) for i in range(1, main.READING_TOTAL_QUESTIONS + 1)]
    assert all(f in inserted[0] for f in ANSWER_FIELDS)

    body = _call(_FakeSupabase(_passage_row(), [], inserted))

    assert len(body["questions"]) == main.READING_TOTAL_QUESTIONS
    for q in body["questions"]:
        for field in ANSWER_FIELDS:
            assert field not in q, f"{field} leaked on the fresh-insert branch"


def test_no_answer_substring_survives_serialisation_on_either_branch():
    """Belt-and-braces: the answer text must not reach the wire at all."""
    import json

    stored = [_stored_row(i) for i in range(1, main.READING_TOTAL_QUESTIONS + 1)]
    inserted = [_inserted_row(i) for i in range(1, main.READING_TOTAL_QUESTIONS + 1)]

    for fake in (_FakeSupabase(_passage_row(), stored, []),
                 _FakeSupabase(_passage_row(), [], inserted)):
        blob = json.dumps(_call(fake))
        assert "quoted sentence" not in blob
        assert "Because of paragraph" not in blob


# ── the allowlist itself ─────────────────────────────────────────────────
def test_response_fields_are_a_subset_of_the_granted_columns():
    inserted = [_inserted_row(i) for i in range(1, main.READING_TOTAL_QUESTIONS + 1)]
    body = _call(_FakeSupabase(_passage_row(), [], inserted))
    allowed = set(main.READING_QUESTION_CLIENT_FIELDS)
    for q in body["questions"]:
        assert set(q) <= allowed, f"unexpected keys: {set(q) - allowed}"


def test_allowlist_excludes_every_answer_field():
    for field in ANSWER_FIELDS:
        assert field not in main.READING_QUESTION_CLIENT_FIELDS


def test_answering_phase_fields_survive():
    """Stripping answers must not strip what the user needs to answer."""
    inserted = [_inserted_row(i) for i in range(1, main.READING_TOTAL_QUESTIONS + 1)]
    body = _call(_FakeSupabase(_passage_row(), [], inserted))
    first = body["questions"][0]
    for field in ("id", "question_type", "question_text", "options", "order_idx"):
        assert field in first
    assert first["options"] == ["A", "B", "C", "D"]


def test_questions_are_ordered_by_order_idx():
    shuffled = [_stored_row(i) for i in (4, 1, 9, 2, 7, 3, 8, 5, 6)]
    body = _call(_FakeSupabase(_passage_row(), shuffled, []))
    assert [q["order_idx"] for q in body["questions"]] == list(
        range(1, main.READING_TOTAL_QUESTIONS + 1)
    )


def test_projection_does_not_mutate_the_source_rows():
    """Internal reads keep their answer columns — only the response is trimmed.

    The submit path scores against these same columns; the fix must remove
    them from the payload, not from what the backend read.
    """
    rows = [_stored_row(i) for i in range(1, main.READING_TOTAL_QUESTIONS + 1)]
    main._client_reading_questions(rows)
    for row in rows:
        for field in ANSWER_FIELDS:
            assert field in row
