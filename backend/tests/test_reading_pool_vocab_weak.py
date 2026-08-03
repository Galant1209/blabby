"""
_tag_vocab_weakness — Pro-gated is_weak tagging for the reading pool
serving response.

vocab_targets is stored as list[str] at generation time (SSE path: Haiku;
blocking path: Sonnet — both untouched by this feature). This function
reshapes it into list[{"word", "is_weak"}] at serving time by joining
against the reader's own user_vocabulary, via vocabulary_items for the
word text. is_weak must be gated on Pro status: a Free user must get
is_weak=False for every word regardless of what's in their
user_vocabulary, and the gate must short-circuit before any DB query is
made — a Free reader's vocab list must never even be looked up.

Hermetic: supabase_admin is patched. No credentials, no network, no DB.
"""

from unittest.mock import patch

import main


USER_ID = "user-abc"


class _Resp:
    def __init__(self, data=None):
        self.data = data if data is not None else []


class _FakeTable:
    def __init__(self, owner, name):
        self._owner = owner
        self._name = name
        self._op = None

    def select(self, cols):
        self._op = "select"
        return self

    def eq(self, *a, **k):
        return self

    def in_(self, *a, **k):
        return self

    def execute(self):
        return self._owner.resolve(self._name, self._op)


class _FakeSupabase:
    def __init__(self):
        self.calls = []
        self._responses = {}

    def queue(self, table, op, resp):
        self._responses.setdefault((table, op), []).append(resp)

    def table(self, name):
        return _FakeTable(self, name)

    def resolve(self, name, op):
        self.calls.append((name, op))
        key = (name, op)
        queue = self._responses.get(key)
        if not queue:
            raise AssertionError(f"no queued response for table={name!r} op={op!r}")
        return queue.pop(0)


def test_pro_user_with_a_weak_word_gets_it_flagged():
    fake = _FakeSupabase()
    fake.queue("user_vocabulary", "select", _Resp(data=[{"vocabulary_item_id": "vi-1"}]))
    fake.queue("vocabulary_items", "select", _Resp(data=[{"word": "Foster"}]))

    with patch.object(main, "supabase_admin", fake):
        result = main._tag_vocab_weakness(["foster", "apple"], USER_ID, is_pro=True)

    assert result == [
        {"word": "foster", "is_weak": True},
        {"word": "apple", "is_weak": False},
    ]


def test_pro_user_with_no_matching_weak_words_flags_none():
    fake = _FakeSupabase()
    fake.queue("user_vocabulary", "select", _Resp(data=[]))

    with patch.object(main, "supabase_admin", fake):
        result = main._tag_vocab_weakness(["foster", "apple"], USER_ID, is_pro=True)

    assert result == [
        {"word": "foster", "is_weak": False},
        {"word": "apple", "is_weak": False},
    ]


def test_free_user_never_flagged_even_with_matching_weak_words():
    """Gate must short-circuit before touching the DB — a Free reader's
    vocab list must not even be queried, let alone used to tag words."""
    fake = _FakeSupabase()
    # Deliberately no queued responses: if the gate leaked and the code
    # tried to query either table, resolve() raises AssertionError and
    # this test fails loudly instead of silently passing.

    with patch.object(main, "supabase_admin", fake):
        result = main._tag_vocab_weakness(["foster", "apple"], USER_ID, is_pro=False)

    assert result == [
        {"word": "foster", "is_weak": False},
        {"word": "apple", "is_weak": False},
    ]
    assert fake.calls == [], "Free-tier gate must not query user_vocabulary or vocabulary_items"


def test_empty_vocab_targets_short_circuits_without_a_db_call():
    fake = _FakeSupabase()
    with patch.object(main, "supabase_admin", fake):
        result = main._tag_vocab_weakness([], USER_ID, is_pro=True)
    assert result == []
    assert fake.calls == []


def test_join_failure_is_non_fatal_and_falls_back_to_unweak():
    fake = _FakeSupabase()
    fake.queue("user_vocabulary", "select", RuntimeError("connection reset"))

    with patch.object(main, "supabase_admin", fake):
        result = main._tag_vocab_weakness(["foster"], USER_ID, is_pro=True)

    assert result == [{"word": "foster", "is_weak": False}]
