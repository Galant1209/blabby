"""
Unit tests for the pure validators in reading_validator.py.

These run without any network, DB, or LLM credentials. They exercise the
single most critical rule (evidence_quote must be a literal substring of
the passage) plus all the shape/count checks.
"""

import re

import pytest

from reading_validator import validate_passage, validate_questions


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────
PASSAGE_BODY = (
    "Paragraph one establishes the topic and outlines the principal claims.\n\n"
    "Paragraph two examines the historical antecedents in some detail.\n\n"
    "Paragraph three considers competing methodologies. "
    + " ".join(f"Sentence {i}." for i in range(120))
)

HEADING_OPTIONS = [
    "First idea",
    "Second idea",
    "Third idea",
    "Distractor X",
    "Distractor Y",
]
MCQ_OPTIONS = ["A. one", "B. two", "C. three", "D. four"]


def _evidence_in_body():
    return "Paragraph one establishes the topic"


def _q(qtype, idx, **overrides):
    base = {
        "question_type":  qtype,
        "question_text":  "Sample question?",
        "options":        None,
        "correct_answer": "True",
        "explanation":    "Because the passage explicitly states so.",
        "evidence_quote": _evidence_in_body(),
        "order_idx":      idx,
    }
    base.update(overrides)
    return base


def _happy_pack():
    # vocab_targets must all be lowercased alphabetic words that appear as
    # whole words in PASSAGE_BODY. 7 satisfies the 6–10 range.
    return {
        "questions": [
            _q("mcq",     1, options=MCQ_OPTIONS, correct_answer="A"),
            _q("mcq",     2, options=MCQ_OPTIONS, correct_answer="B"),
            _q("mcq",     3, options=MCQ_OPTIONS, correct_answer="C"),
            _q("tfng",    4, correct_answer="True"),
            _q("tfng",    5, correct_answer="False"),
            _q("tfng",    6, correct_answer="Not Given"),
            _q("heading", 7, options=HEADING_OPTIONS, correct_answer="First idea"),
            _q("heading", 8, options=HEADING_OPTIONS, correct_answer="Second idea"),
            _q("heading", 9, options=HEADING_OPTIONS, correct_answer="Third idea"),
        ],
        "vocab_targets": [
            "establishes", "outlines", "principal",
            "examines", "antecedents", "considers", "methodologies",
        ],
    }


# ────────────────────────────────────────────────────────────────────────────
#  validate_passage
# ────────────────────────────────────────────────────────────────────────────
def test_passage_happy_path():
    body = " ".join(["word"] * 750)
    p = {"title": "A Reasonable Title", "body": body, "topic": "x", "word_count": 750}
    assert validate_passage(p) == (True, None)


def test_passage_rejects_short_body():
    p = {"title": "T", "body": "word " * 300, "topic": "x", "word_count": 300}
    ok, reason = validate_passage(p)
    assert ok is False
    assert "outside 600" in reason


def test_passage_rejects_word_count_mismatch():
    body = " ".join(["word"] * 800)
    p = {"title": "T", "body": body, "topic": "x", "word_count": 500}
    ok, reason = validate_passage(p)
    assert ok is False
    assert "word_count mismatch" in reason


def test_passage_rejects_missing_keys():
    ok, reason = validate_passage({"title": "T"})
    assert ok is False
    assert "missing key" in reason


def test_passage_rejects_oversized_title():
    body = " ".join(["word"] * 750)
    p = {
        "title": "x" * 200,
        "body": body, "topic": "x", "word_count": 750,
    }
    ok, reason = validate_passage(p)
    assert ok is False
    assert "title too long" in reason


# ────────────────────────────────────────────────────────────────────────────
#  validate_questions
# ────────────────────────────────────────────────────────────────────────────
def test_questions_happy_path():
    assert validate_questions(_happy_pack(), PASSAGE_BODY) == (True, None)


def test_questions_rejects_fabricated_evidence():
    pack = _happy_pack()
    pack["questions"][0]["evidence_quote"] = (
        "This sentence does not appear anywhere in the supplied passage."
    )
    ok, reason = validate_questions(pack, PASSAGE_BODY)
    assert ok is False
    assert "evidence_quote not found" in reason


def test_questions_rejects_monotone_tfng():
    pack = _happy_pack()
    for i in (3, 4, 5):
        pack["questions"][i]["correct_answer"] = "True"
    ok, reason = validate_questions(pack, PASSAGE_BODY)
    assert ok is False
    assert "tfng answers lack variety" in reason


def test_questions_rejects_wrong_count():
    pack = _happy_pack()
    pack["questions"] = pack["questions"][:8]
    ok, reason = validate_questions(pack, PASSAGE_BODY)
    assert ok is False
    assert "expected 9 questions" in reason


def test_questions_rejects_mcq_bad_letter():
    pack = _happy_pack()
    pack["questions"][0]["correct_answer"] = "E"
    ok, reason = validate_questions(pack, PASSAGE_BODY)
    assert ok is False
    assert "mcq correct_answer" in reason


def test_questions_rejects_heading_correct_not_in_options():
    pack = _happy_pack()
    pack["questions"][6]["correct_answer"] = "Heading not present"
    ok, reason = validate_questions(pack, PASSAGE_BODY)
    assert ok is False
    assert "heading correct_answer not in options" in reason


def test_questions_rejects_duplicate_order_idx():
    pack = _happy_pack()
    pack["questions"][1]["order_idx"] = 1  # duplicate of q[0]
    ok, reason = validate_questions(pack, PASSAGE_BODY)
    assert ok is False
    assert "duplicate order_idx" in reason


# ────────────────────────────────────────────────────────────────────────────
#  vocab_targets (Sprint Reading-2)
# ────────────────────────────────────────────────────────────────────────────
def test_questions_happy_path_includes_vocab_targets():
    pack = _happy_pack()
    assert isinstance(pack.get("vocab_targets"), list)
    assert 6 <= len(pack["vocab_targets"]) <= 10
    ok, reason = validate_questions(pack, PASSAGE_BODY)
    assert ok is True
    assert reason is None


def test_questions_rejects_vocab_targets_too_few():
    pack = _happy_pack()
    pack["vocab_targets"] = pack["vocab_targets"][:5]
    ok, reason = validate_questions(pack, PASSAGE_BODY)
    assert ok is False
    assert "vocab_targets count 5" in reason


def test_questions_rejects_vocab_targets_too_many():
    pack = _happy_pack()
    # 11 distinct whole-word matches present in PASSAGE_BODY.
    pack["vocab_targets"] = [
        "establishes", "outlines", "principal", "examines", "antecedents",
        "considers", "methodologies", "historical", "competing", "topic",
        "claims",
    ]
    ok, reason = validate_questions(pack, PASSAGE_BODY)
    assert ok is False
    assert "vocab_targets count 11" in reason


def test_questions_rejects_vocab_target_not_in_passage():
    pack = _happy_pack()
    pack["vocab_targets"][0] = "unicorn"  # not in PASSAGE_BODY
    ok, reason = validate_questions(pack, PASSAGE_BODY)
    assert ok is False
    assert "'unicorn'" in reason
    assert "not a whole word" in reason


def test_questions_rejects_vocab_target_substring_match():
    # Build a passage where "artisan" appears but "art" does not stand alone.
    # Prepend the happy-pack's evidence quote so the earlier evidence check
    # passes and the failure is genuinely from the vocab_targets rule.
    custom_body = (
        "Paragraph one establishes the topic and outlines the principal "
        "claims. "
        + "The artisan worked the loom. " * 30
        + "The methodologies described here illustrate the principle."
        + " ".join(f"Sentence {i}." for i in range(60))
    )
    pack = _happy_pack()
    pack["vocab_targets"] = [
        "art",              # ← only appears inside "artisan", not as whole word
        "artisan", "methodologies", "principle", "loom", "sentence",
    ]
    ok, reason = validate_questions(pack, custom_body)
    assert ok is False
    assert "'art'" in reason
    assert "not a whole word" in reason


def test_questions_rejects_vocab_target_duplicate():
    pack = _happy_pack()
    pack["vocab_targets"][1] = pack["vocab_targets"][0]  # duplicate
    ok, reason = validate_questions(pack, PASSAGE_BODY)
    assert ok is False
    assert "duplicate" in reason


def test_questions_rejects_vocab_target_with_punctuation():
    pack = _happy_pack()
    pack["vocab_targets"][0] = "aristocracy,"  # trailing comma
    ok, reason = validate_questions(pack, PASSAGE_BODY)
    assert ok is False
    assert "not lowercase alphabetic" in reason


def test_questions_rejects_vocab_target_with_apostrophe():
    # Apostrophes are explicitly disallowed by the alphabetic-only rule.
    pack = _happy_pack()
    pack["vocab_targets"][0] = "don't"
    ok, reason = validate_questions(pack, PASSAGE_BODY)
    assert ok is False
    assert "not lowercase alphabetic" in reason


def test_questions_rejects_vocab_target_uppercase():
    pack = _happy_pack()
    pack["vocab_targets"][0] = "Aristocracy"
    ok, reason = validate_questions(pack, PASSAGE_BODY)
    assert ok is False
    assert "not lowercase alphabetic" in reason


def test_questions_rejects_missing_vocab_targets():
    pack = _happy_pack()
    del pack["vocab_targets"]
    ok, reason = validate_questions(pack, PASSAGE_BODY)
    assert ok is False
    assert "vocab_targets" in reason
