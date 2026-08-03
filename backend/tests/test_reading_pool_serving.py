"""
GET /api/reading/passage — pool-first serving endpoint.

Covers:
  - pool-hit: passage + all 9 questions returned in one call, used_count
    incremented
  - serving-time re-check retires a passage whose question count no
    longer matches READING_TOTAL_QUESTIONS and retries the next candidate
  - the three fallback_reason classifications: pool_empty,
    pool_exhausted_rejects, pool_lookup_error

Hermetic: supabase_admin, verify_token, quota enforcement and the Pro
lookup are all patched. No credentials, no network, no DB, no background
thread actually spawned for the on-demand replenish.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import main


USER_ID = "user-abc"


# ── fakes ────────────────────────────────────────────────────────────────
class _Resp:
    def __init__(self, data=None, count=None):
        self.data = data if data is not None else []
        self.count = count


class _FakeTable:
    def __init__(self, owner, name):
        self._owner = owner
        self._name = name
        self._op = None
        self._select_cols = None
        self._payload = None

    def select(self, cols, count=None):
        self._op = "select"
        self._select_cols = cols
        return self

    def update(self, values):
        self._op = "update"
        self._payload = values
        return self

    def eq(self, *a, **k):
        return self

    def order(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    @property
    def not_(self):
        return self

    def in_(self, *a, **k):
        return self

    def execute(self):
        return self._owner.resolve(self._name, self._op, self._select_cols, self._payload)


class _FakeSupabase:
    def __init__(self):
        self.calls = []
        self._responses = {}

    def queue(self, table, op, resp):
        self._responses.setdefault((table, op), []).append(resp)

    def table(self, name):
        return _FakeTable(self, name)

    def resolve(self, name, op, select_cols, payload):
        self.calls.append({"table": name, "op": op, "select": select_cols, "payload": payload})
        key = (name, op)
        queue = self._responses.get(key)
        if not queue:
            raise AssertionError(
                f"no queued response for table={name!r} op={op!r} "
                f"select={select_cols!r} payload={payload!r}"
            )
        item = queue.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


def _passage_row(passage_id, topic="general", used_count=0, vocab_targets=None):
    return {
        "id": passage_id,
        "title": "The domestication of animals.",
        "body": "Paragraph one.\n\nParagraph two.",
        "difficulty_band": 6.0,
        "topic": topic,
        "word_count": 400,
        "vocab_targets": vocab_targets or [],
        "used_count": used_count,
    }


def _question_rows(passage_id):
    return [
        {
            "id": f"q-{i}",
            "passage_id": passage_id,
            "question_type": "mcq",
            "question_text": f"Question {i}?",
            "options": ["A", "B", "C", "D"],
            "correct_answer": "B",
            "explanation": f"Because {i}.",
            "evidence_quote": f"quote {i}",
            "order_idx": i,
        }
        for i in range(1, main.READING_TOTAL_QUESTIONS + 1)
    ]


def _call(fake, topic=None):
    request = MagicMock()
    request.json = AsyncMock(return_value={})
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", return_value=USER_ID), \
         patch.object(main, "supabase_admin", fake), \
         patch.object(main, "_enforce_reading_quota", return_value=None), \
         patch.object(main, "get_user_pro_status", return_value=False), \
         patch.object(main, "_replenish_reading_async") as replenish_mock:
        result = asyncio.run(
            main.reading_get_pool_passage(request=request, topic=topic, authorization="Bearer fake")
        )
    return result, replenish_mock


# ── pool-hit ─────────────────────────────────────────────────────────────
def test_pool_hit_returns_passage_and_all_nine_questions_in_one_call():
    fake = _FakeSupabase()
    fake.queue("reading_passages", "select", _Resp(data=[_passage_row("P1", used_count=3)]))
    fake.queue("reading_questions", "select", _Resp(data=[{"id": f"q-{i}"} for i in range(9)], count=9))
    fake.queue("reading_questions", "select", _Resp(data=_question_rows("P1")))
    fake.queue("reading_passages", "update", _Resp(data=[{"id": "P1"}]))
    fake.queue("reading_passages", "select", _Resp(data=[{"id": "x"}] * 5, count=5))

    body, replenish_mock = _call(fake)

    assert body["pool_hit"] is True
    assert body["passage_id"] == "P1"
    assert len(body["questions"]) == main.READING_TOTAL_QUESTIONS
    for q in body["questions"]:
        for forbidden in ("correct_answer", "explanation", "evidence_quote"):
            assert forbidden not in q, f"{forbidden} leaked on the pool-hit path"

    used_count_update = next(c for c in fake.calls if c["op"] == "update")
    assert used_count_update["payload"] == {"used_count": 4}
    replenish_mock.assert_not_called()  # remaining=5 >= READING_POOL_LOW_WATERMARK


def test_pool_hit_triggers_replenish_below_low_watermark():
    fake = _FakeSupabase()
    fake.queue("reading_passages", "select", _Resp(data=[_passage_row("P1", topic="linguistics")]))
    fake.queue("reading_questions", "select", _Resp(data=[{"id": f"q-{i}"} for i in range(9)], count=9))
    fake.queue("reading_questions", "select", _Resp(data=_question_rows("P1")))
    fake.queue("reading_passages", "update", _Resp(data=[{"id": "P1"}]))
    fake.queue("reading_passages", "select", _Resp(data=[{"id": "x"}], count=1))

    body, replenish_mock = _call(fake)

    assert body["pool_hit"] is True
    replenish_mock.assert_called_once_with("linguistics")


# ── serving-time retire + retry ────────────────────────────────────────
def test_incomplete_question_set_is_retired_and_next_candidate_served():
    fake = _FakeSupabase()
    # Iteration 1: candidate P1 has only 5 questions -> retire, exclude, retry.
    fake.queue("reading_passages", "select", _Resp(data=[_passage_row("P1")]))
    fake.queue("reading_questions", "select", _Resp(data=[{"id": f"q-{i}"} for i in range(5)], count=5))
    fake.queue("reading_passages", "update", _Resp(data=[{"id": "P1"}]))
    # Iteration 2: candidate P2 is complete -> served.
    fake.queue("reading_passages", "select", _Resp(data=[_passage_row("P2")]))
    fake.queue("reading_questions", "select", _Resp(data=[{"id": f"q-{i}"} for i in range(9)], count=9))
    fake.queue("reading_questions", "select", _Resp(data=_question_rows("P2")))
    fake.queue("reading_passages", "update", _Resp(data=[{"id": "P2"}]))
    fake.queue("reading_passages", "select", _Resp(data=[{"id": "x"}] * 5, count=5))

    body, _ = _call(fake)

    assert body["pool_hit"] is True
    assert body["passage_id"] == "P2"

    retire_update = fake.calls[2]
    assert retire_update["table"] == "reading_passages"
    assert retire_update["op"] == "update"
    assert retire_update["payload"] == {"is_pregenerated": False}


# ── fallback_reason classification ─────────────────────────────────────
def test_fallback_reason_pool_empty_when_no_candidates_at_all():
    fake = _FakeSupabase()
    fake.queue("reading_passages", "select", _Resp(data=[]))

    body, _ = _call(fake)

    assert body == {"pool_hit": False, "fallback_reason": "pool_empty"}


def test_fallback_reason_pool_exhausted_rejects_after_three_bad_candidates():
    fake = _FakeSupabase()
    for i in range(3):
        fake.queue("reading_passages", "select", _Resp(data=[_passage_row(f"P{i}")]))
        fake.queue("reading_questions", "select", _Resp(data=[{"id": "q-1"}], count=1))
        fake.queue("reading_passages", "update", _Resp(data=[{"id": f"P{i}"}]))

    body, _ = _call(fake)

    assert body == {"pool_hit": False, "fallback_reason": "pool_exhausted_rejects"}
    retire_updates = [c for c in fake.calls if c["op"] == "update"]
    assert len(retire_updates) == 3


def test_fallback_reason_pool_lookup_error_on_db_exception():
    fake = _FakeSupabase()
    fake.queue("reading_passages", "select", _Resp(data=[_passage_row("P1")]))
    fake.queue("reading_questions", "select", RuntimeError("connection reset"))

    body, _ = _call(fake)

    assert body == {"pool_hit": False, "fallback_reason": "pool_lookup_error"}


def test_invalid_topic_is_rejected_before_any_db_call():
    fake = _FakeSupabase()
    request = MagicMock()
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", return_value=USER_ID), \
         patch.object(main, "supabase_admin", fake), \
         patch.object(main, "_enforce_reading_quota", return_value=None):
        try:
            asyncio.run(
                main.reading_get_pool_passage(
                    request=request, topic="not-a-real-topic", authorization="Bearer fake",
                )
            )
            assert False, "expected HTTPException"
        except main.HTTPException as exc:
            assert exc.status_code == 400
    assert fake.calls == []
