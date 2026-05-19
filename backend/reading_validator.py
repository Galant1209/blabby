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


# Common English inflectional and derivational suffixes used to bridge
# the gap between LLM-emitted lemmas (e.g. "nominalisation") and
# surface forms appearing in the passage (e.g. "nominalisations").
#
# Ordering matters: LONGER suffixes first, so that "ations" is tested
# before "ation" — otherwise stripping "ation" from "nominalisations"
# leaves "nominalis", which is no real word; we want to strip "ations"
# in one go to reach the lemma "nominalis" or further "nominalise".
#
# Two-level inflection (e.g. -ation then -s) is handled by trying
# additive matches in both directions.
_INFLECTION_SUFFIXES = (
    # 5-letter
    "ations",
    # 4-letter
    "ation", "ising", "izing", "ously", "ition", "ities",
    # 3-letter
    "ies", "ing", "ous", "ity", "ise", "ize", "ial",
    "ant", "ent", "ism", "ist", "ive",
    # 2-letter
    "es", "ed", "er", "or", "al", "ly",
    # 1-letter
    "s", "d",
)


def _word_appears_in_passage(target: str, passage_lower: str) -> bool:
    """
    Whole-word search for `target` (a lowercase lemma) in passage_lower,
    tolerant of common English inflection and derivation in either
    direction:

      (a) target appears verbatim                  -- exact match
      (b) target + any common suffix appears       -- LLM gave lemma,
                                                       passage has surface
      (c) some passage token strips to target      -- LLM gave deeper lemma,
                                                       passage has derived form

    Returns True on first match. Does not handle:
      - irregular forms (run/ran/running — passage rarely uses these as
        learner vocabulary anyway)
      - compound words / hyphenation
      - spelling variants (analyse/analyze — caller's prompt asks for
        the form used in the passage)

    Designed for IELTS academic vocabulary where most inflection follows
    regular suffix patterns.
    """
    # (a) exact whole-word match
    if re.search(r"\b" + re.escape(target) + r"\b", passage_lower):
        return True

    # (b) target + suffix, whole-word match
    for suffix in _INFLECTION_SUFFIXES:
        if re.search(
            r"\b" + re.escape(target + suffix) + r"\b",
            passage_lower,
        ):
            return True

    # (b') E-drop forward: English verbs ending in -e drop the e before
    # vowel-initial suffixes (nominalise → nominalisation, regulate →
    # regulation). If target ends in 'e', try stem + vowel-initial suffix.
    if target.endswith("e") and len(target) >= 3:
        stem = target[:-1]
        for suffix in _INFLECTION_SUFFIXES:
            if suffix and suffix[0] in "aeiou":
                if re.search(
                    r"\b" + re.escape(stem + suffix) + r"\b",
                    passage_lower,
                ):
                    return True

    # (c) passage tokens that strip to target. Extract candidate tokens
    # of length up to len(target) + longest_suffix; check each one's
    # possible lemma forms against target.
    longest_suffix = len(_INFLECTION_SUFFIXES[0])  # "ations" = 5
    max_token_len = len(target) + longest_suffix
    # Bound candidate tokens to plausibly-inflected forms of target. We
    # only check tokens that start with at least 3 chars of target —
    # cheaper than scanning every word.
    if len(target) >= 4:
        prefix = target[:4]
        candidate_pattern = (
            r"\b" + re.escape(prefix) + r"[a-z]{0," +
            str(max_token_len - len(prefix)) + r"}\b"
        )
        for match in re.finditer(candidate_pattern, passage_lower):
            token = match.group()
            # Try stripping each suffix from the token; if any strip
            # yields target, accept.
            for suffix in _INFLECTION_SUFFIXES:
                if token.endswith(suffix):
                    stripped = token[: -len(suffix)]
                    if stripped == target:
                        return True
                    # E-drop reverse: passage form may be the stem (with
                    # the e dropped); target may be the verb retaining e.
                    # If suffix is vowel-initial and stripped + 'e' equals
                    # target, treat as match.
                    if (suffix and suffix[0] in "aeiou"
                            and stripped + "e" == target):
                        return True
                    # One more level: strip another suffix.
                    for suffix2 in _INFLECTION_SUFFIXES:
                        if stripped.endswith(suffix2):
                            stripped2 = stripped[: -len(suffix2)]
                            if stripped2 == target:
                                return True
                            # E-drop at the second-level strip too.
                            if (suffix2 and suffix2[0] in "aeiou"
                                    and stripped2 + "e" == target):
                                return True

    return False


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
    # Strategy: targets that match the passage (exact or lemma-aware)
    # are kept; targets that don't match are silently dropped. The final
    # count must be >= 6 (preserving the original lower bound) but may
    # be < 10 if some LLM-emitted targets couldn't be found.
    #
    # Rationale: vocab_targets drive a dotted-underline UI affordance.
    # A target absent from the passage simply doesn't get rendered as
    # tappable — harmless. Rejecting the entire response over one
    # un-locatable target trades a small UI degradation for a hard 500
    # to the user. The lemma-aware matcher (_word_appears_in_passage)
    # catches most inflection gaps; this filter handles the rest.
    #
    # Other vocab_targets rules remain strict — duplicates, non-strings,
    # non-lowercase-alphabetic, and structural shape violations still
    # reject. Only the "exists in passage" check is downgraded to filter.
    if "vocab_targets" not in data:
        return False, "missing key: vocab_targets"
    targets = data.get("vocab_targets")
    if not isinstance(targets, list):
        return False, "vocab_targets is not a list"
    if len(targets) > 10:
        return False, f"vocab_targets count {len(targets)} exceeds 10"
    # Lower bound deferred — checked after filtering below.

    seen: set[str] = set()
    kept: list[str] = []
    passage_lower = (passage_body or "").lower()
    for i, w in enumerate(targets):
        if not isinstance(w, str):
            return False, f"vocab_targets[{i}] is not a string"
        if not w:
            return False, f"vocab_targets[{i}] is empty"
        if not re.fullmatch(r"[a-z]+", w):
            return False, f"vocab_targets[{i}] {w!r} not lowercase alphabetic"
        if w in seen:
            return False, f"vocab_targets[{i}] duplicate: {w!r}"
        seen.add(w)
        # Lemma-aware whole-word match. Targets not present in any
        # tolerated form are silently dropped (see strategy note above).
        if _word_appears_in_passage(w, passage_lower):
            kept.append(w)

    if len(kept) < 6:
        return (
            False,
            f"vocab_targets: only {len(kept)} of {len(targets)} found in "
            f"passage (need >= 6)",
        )

    # Mutate data so callers persist the filtered list, not the raw
    # LLM output. This is the only place where validator mutates input,
    # justified because the alternative (returning a separate clean list)
    # would require changing the validator's API.
    data["vocab_targets"] = kept

    return True, None
