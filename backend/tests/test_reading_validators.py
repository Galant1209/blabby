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
        ]
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
