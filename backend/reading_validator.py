"""
Pure validators for Reading-module LLM output. No I/O, no side effects.

Mirrors the structural pattern of validate_correction_response() in main.py:
returns (is_valid, reason). The reason is for logging only — never surfaced
to the client.
"""

from __future__ import annotations

import re
from typing import Optional


_TFNG_VALUES = {"True", "False", "Not Given"}
_MCQ_LETTERS = {"A", "B", "C", "D"}


def _word_count(text: str) -> int:
    return len(re.findall(r"\b\w+\b", text or ""))


def _normalise(s: str) -> str:
    """Lowercase + collapse all whitespace runs to single spaces."""
    return re.sub(r"\s+", " ", (s or "").strip().lower())


def validate_passage(data: dict) -> tuple[bool, Optional[str]]:
    """
    Validate a passage JSON object emitted by the LLM.

    Required keys: title (str), body (str), topic (str), word_count (int).
    Body must be 600–1000 words (slight tolerance around the 700–900 target).
    word_count must agree with actual count within ±10%.
    Title must be non-empty and ≤ 120 characters.
    """
    if not isinstance(data, dict):
        return False, f"passage is {type(data).__name__}, not dict"

    for key in ("title", "body", "topic", "word_count"):
        if key not in data:
            return False, f"missing key: {key}"

    title = data.get("title")
    body = data.get("body")
    topic = data.get("topic")
    word_count = data.get("word_count")

    if not isinstance(title, str) or not title.strip():
        return False, "title empty or non-string"
    if len(title) > 120:
        return False, f"title too long: {len(title)} chars"

    if not isinstance(body, str) or not body.strip():
        return False, "body empty or non-string"

    if not isinstance(topic, str) or not topic.strip():
        return False, "topic empty or non-string"

    if not isinstance(word_count, int):
        return False, f"word_count is {type(word_count).__name__}, not int"

    actual = _word_count(body)
    if actual < 600 or actual > 1000:
        return False, f"body word count {actual} outside 600–1000 range"

    # ±10% tolerance between claimed and actual word_count
    if actual > 0:
        ratio = abs(actual - word_count) / actual
        if ratio > 0.10:
            return False, f"word_count mismatch: claimed={word_count}, actual={actual}"

    return True, None


def _quote_appears_in(passage: str, quote: str) -> bool:
    """
    Case-insensitive, whitespace-normalised substring check. Tolerates the
    LLM rewrapping line breaks inside the quote.
    """
    if not quote or not quote.strip():
        return False
    return _normalise(quote) in _normalise(passage)


def validate_questions(
    data: dict,
    passage_body: str,
) -> tuple[bool, Optional[str]]:
    """
    Validate the question pack emitted by the LLM.

    Structural rules:
        - Exactly 9 questions, split 3 MCQ / 3 TFNG / 3 Heading
        - order_idx values 1–9, no duplicates
        - Every evidence_quote is a literal substring of passage_body
          (case-insensitive, whitespace-normalised)
        - MCQ: 4 options, correct_answer ∈ {A,B,C,D}
        - TFNG: options is null, correct_answer ∈ {True,False,Not Given},
          and the 3 TFNG items cover at least 2 distinct values
        - Heading: options has exactly 5 entries; correct_answer is one of
          the options (case-insensitive, whitespace-normalised match)
    """
    if not isinstance(data, dict):
        return False, f"questions payload is {type(data).__name__}, not dict"

    questions = data.get("questions")
    if not isinstance(questions, list):
        return False, "questions is not a list"
    if len(questions) != 9:
        return False, f"expected 9 questions, got {len(questions)}"

    counts: dict[str, int] = {"mcq": 0, "tfng": 0, "heading": 0}
    order_idx_seen: set[int] = set()
    tfng_answers: set[str] = set()

    for i, q in enumerate(questions):
        if not isinstance(q, dict):
            return False, f"q[{i}] not a dict"

        for key in (
            "question_type", "question_text", "options",
            "correct_answer", "explanation", "evidence_quote", "order_idx",
        ):
            if key not in q:
                return False, f"q[{i}] missing key: {key}"

        qtype = q.get("question_type")
        qtext = q.get("question_text")
        options = q.get("options")
        correct = q.get("correct_answer")
        explanation = q.get("explanation")
        evidence = q.get("evidence_quote")
        order_idx = q.get("order_idx")

        if qtype not in counts:
            return False, f"q[{i}] invalid question_type: {qtype!r}"
        counts[qtype] += 1

        if not isinstance(qtext, str) or not qtext.strip():
            return False, f"q[{i}] question_text empty"
        if not isinstance(correct, str) or not correct.strip():
            return False, f"q[{i}] correct_answer empty"
        if not isinstance(explanation, str) or not explanation.strip():
            return False, f"q[{i}] explanation empty"

        if not isinstance(order_idx, int):
            return False, f"q[{i}] order_idx not int"
        if order_idx in order_idx_seen:
            return False, f"q[{i}] duplicate order_idx={order_idx}"
        order_idx_seen.add(order_idx)
        if order_idx < 1 or order_idx > 9:
            return False, f"q[{i}] order_idx {order_idx} out of range"

        # Evidence check — the single most-critical rule. A fabricated quote
        # makes the question unusable; the explanation cannot be trusted.
        # TFNG "Not Given" still requires a quote (the most-relevant span);
        # this catches LLMs that emit empty strings for that case.
        if not isinstance(evidence, str) or not evidence.strip():
            return False, f"q[{i}] evidence_quote empty"
        if not _quote_appears_in(passage_body, evidence):
            return False, f"q[{i}] evidence_quote not found in passage"

        if qtype == "mcq":
            if not isinstance(options, list) or len(options) != 4:
                return False, f"q[{i}] mcq options must be 4-item list"
            if not all(isinstance(o, str) and o.strip() for o in options):
                return False, f"q[{i}] mcq options must be non-empty strings"
            if correct.strip().upper() not in _MCQ_LETTERS:
                return False, f"q[{i}] mcq correct_answer {correct!r} not A/B/C/D"

        elif qtype == "tfng":
            if options is not None:
                return False, f"q[{i}] tfng options must be null"
            normalised_correct = correct.strip()
            # tolerate "true"/"false"/"not given" casing
            canonical = {v.lower(): v for v in _TFNG_VALUES}
            if normalised_correct.lower() not in canonical:
                return False, f"q[{i}] tfng correct_answer {correct!r} invalid"
            tfng_answers.add(canonical[normalised_correct.lower()])

        elif qtype == "heading":
            if not isinstance(options, list) or len(options) != 5:
                return False, f"q[{i}] heading options must be 5-item list"
            if not all(isinstance(o, str) and o.strip() for o in options):
                return False, f"q[{i}] heading options must be non-empty strings"
            normalised_options = {_normalise(o) for o in options}
            if _normalise(correct) not in normalised_options:
                return False, f"q[{i}] heading correct_answer not in options"

    if counts != {"mcq": 3, "tfng": 3, "heading": 3}:
        return False, f"type counts wrong: {counts}"
    if len(tfng_answers) < 2:
        return False, f"tfng answers lack variety: {sorted(tfng_answers)}"

    # ---- vocab_targets validation ------------------------------------
    if "vocab_targets" not in data:
        return False, "missing key: vocab_targets"
    targets = data.get("vocab_targets")
    if not isinstance(targets, list):
        return False, "vocab_targets is not a list"
    if not (6 <= len(targets) <= 10):
        return False, f"vocab_targets count {len(targets)} outside 6–10"

    seen: set[str] = set()
    # Pre-compute a lowercased copy of the passage for word-boundary checks.
    passage_lower = (passage_body or "").lower()
    for i, w in enumerate(targets):
        if not isinstance(w, str):
            return False, f"vocab_targets[{i}] is not a string"
        if not w:
            return False, f"vocab_targets[{i}] is empty"
        # Must be lowercase alphabetic only (no digits, no punctuation,
        # no whitespace, no hyphens or apostrophes). The prompt enforces
        # single-token English; the validator enforces it strictly.
        if not re.fullmatch(r"[a-z]+", w):
            return False, f"vocab_targets[{i}] {w!r} not lowercase alphabetic"
        if w in seen:
            return False, f"vocab_targets[{i}] duplicate: {w!r}"
        seen.add(w)
        # Whole-word match against the passage. \b anchors prevent
        # "art" matching inside "artisan".
        if not re.search(r"\b" + re.escape(w) + r"\b", passage_lower):
            return False, f"vocab_targets[{i}] {w!r} not a whole word in passage"

    return True, None
