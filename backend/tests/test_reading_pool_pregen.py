"""
Reading pool pregen worker — two-phase insert integrity.

_pregenerate_reading_topic() inserts a passage row first
(is_pregenerated=False, questions_ready=False), then all 9 questions; only
after the questions insert succeeds does it flip both flags with a single
update. A questions-insert failure must NOT roll back the passage — the
passage is left as a "soft orphan" (questions_ready stays False), the same
shape as the 5 pre-existing orphans that predate pool tracking. This module
proves that contract holds and that the flip-to-ready update is never
reached on the failure path.

Hermetic: supabase_admin and the LLM generation helper are both patched.
No credentials, no network, no DB.
"""

import logging
from unittest.mock import patch

import main


TOPIC = "general"


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

    def insert(self, rows):
        self._op = "insert"
        self._payload = rows
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


def _fake_generation():
    passage_data = {
        "title": "The domestication of animals.",
        "body": "Paragraph one.\n\nParagraph two.",
        "word_count": 400,
        "topic": TOPIC,
    }
    questions_data = {
        "questions": [
            {
                "question_type": "mcq",
                "question_text": f"Question {i}?",
                "options": ["A", "B", "C", "D"],
                "correct_answer": "B",
                "explanation": f"Because {i}.",
                "evidence_quote": f"quote {i}",
                "order_idx": i,
            }
            for i in range(1, main.READING_TOTAL_QUESTIONS + 1)
        ],
        "vocab_targets": ["foster", "considerate"],
    }
    return passage_data, questions_data


# ── tests ────────────────────────────────────────────────────────────────
def test_questions_insert_failure_leaves_passage_as_soft_orphan(caplog):
    """
    Mock the questions insert to raise. The passage row must have already
    been inserted (is_pregenerated=False, questions_ready=False) and the
    flip-to-ready update must NEVER be called — the passage stays excluded
    from the pool exactly like the 5 pre-existing orphans.
    """
    fake = _FakeSupabase()
    # Bucket count check: 0 existing pool-ready passages, target=1 -> need 1.
    fake.queue("reading_passages", "select", _Resp(data=[], count=0))
    fake.queue("reading_passages", "insert", _Resp(data=[{"id": "passage-orphan-1"}]))
    fake.queue("reading_questions", "insert", RuntimeError("simulated insert failure"))
    # Deliberately no queued response for ("reading_passages", "update") —
    # if the code path reaches the flip-to-ready update despite the
    # failure above, resolve() raises AssertionError and the test fails
    # loudly rather than silently passing.

    with caplog.at_level(logging.INFO, logger="main"), \
         patch.object(main, "supabase_admin", fake), \
         patch.object(main, "_generate_reading_passage_and_questions",
                      return_value=_fake_generation()):
        main._pregenerate_reading_topic(TOPIC, target_per_topic=1)

    insert_calls = [c for c in fake.calls if c["op"] == "insert"]
    assert [c["table"] for c in insert_calls] == ["reading_passages", "reading_questions"]

    passage_insert_payload = insert_calls[0]["payload"]
    assert passage_insert_payload["is_pregenerated"] is False
    assert passage_insert_payload["questions_ready"] is False
    assert passage_insert_payload["topic"] == TOPIC

    update_calls = [c for c in fake.calls if c["op"] == "update"]
    assert update_calls == [], (
        "flip-to-ready update must not run when the questions insert failed"
    )

    messages = [r.getMessage() for r in caplog.records]
    assert any("soft orphan" in m and "passage-orphan-1" in m for m in messages), (
        "questions-insert failure must be logged with the passage_id"
    )
    assert any("generated=0 failed=1" in m for m in messages)


def test_full_success_flips_both_flags_in_a_single_update():
    """Baseline: when both inserts succeed, exactly one update flips both
    is_pregenerated and questions_ready to True together."""
    fake = _FakeSupabase()
    fake.queue("reading_passages", "select", _Resp(data=[], count=0))
    fake.queue("reading_passages", "insert", _Resp(data=[{"id": "passage-ok-1"}]))
    fake.queue(
        "reading_questions", "insert",
        _Resp(data=[{"id": f"q-{i}"} for i in range(1, main.READING_TOTAL_QUESTIONS + 1)]),
    )
    fake.queue("reading_passages", "update", _Resp(data=[{"id": "passage-ok-1"}]))

    with patch.object(main, "supabase_admin", fake), \
         patch.object(main, "_generate_reading_passage_and_questions",
                      return_value=_fake_generation()):
        main._pregenerate_reading_topic(TOPIC, target_per_topic=1)

    update_calls = [c for c in fake.calls if c["op"] == "update"]
    assert len(update_calls) == 1
    assert update_calls[0]["payload"] == {"questions_ready": True, "is_pregenerated": True}


def test_bucket_already_at_target_skips_generation_entirely():
    """current_count >= target_per_topic must short-circuit before any
    LLM call or insert — the pregen loop must not run at all."""
    fake = _FakeSupabase()
    fake.queue("reading_passages", "select", _Resp(data=[{"id": "x"}] * 6, count=6))

    with patch.object(main, "supabase_admin", fake), \
         patch.object(main, "_generate_reading_passage_and_questions") as gen_mock:
        main._pregenerate_reading_topic(TOPIC, target_per_topic=6)

    gen_mock.assert_not_called()
    assert [c["op"] for c in fake.calls] == ["select"]
