"""
Prompt builders for the Reading module.

Pure functions — no I/O, no client objects. Both builders return system+user
prompts via a single returned string designed to be passed to Claude as the
`system` message; the actual user-turn payload is supplied by the caller (the
passage prompt has no user input; the questions prompt receives the passage
body as user input).

All exam-facing strings here are formal British English: austere register,
British spellings (organise, analyse, behaviour, centre, recognise).
"""

from typing import Optional


_IELTS_TOPIC_HINT = (
    "history of technology, natural sciences, social history, urban planning, "
    "environmental studies, archaeology, linguistics, education theory"
)


def _band_difficulty_clause(band: float) -> str:
    """Map a target band to concrete syntactic/lexical guidance."""
    if band < 5.5:
        return (
            "Bands 4.0–5.0: short sentences (mean length ≤ 20 words), common "
            "Academic Word List vocabulary, minimal nominalisation, plain "
            "topic-comment structure."
        )
    if band < 6.5:
        return (
            "Bands 5.5–6.0: mixed sentence lengths, occasional subordinate "
            "clauses, mid-frequency academic vocabulary, modest nominalisation."
        )
    if band < 7.5:
        return (
            "Bands 6.5–7.0: complex sentences with subordinate and relative "
            "clauses, regular nominalisation, mid-to-lower-frequency academic "
            "vocabulary, some abstract reference."
        )
    return (
        "Bands 7.5–9.0: dense complex syntax, sustained nominalisation, "
        "lower-frequency academic vocabulary, abstract argumentation, "
        "controlled but demanding cohesion."
    )


def build_passage_prompt(
    difficulty_band: float,
    topic: Optional[str],
) -> str:
    """
    System prompt that instructs Claude to generate a single IELTS Academic
    Reading passage at the requested band.

    Output contract — strict JSON, no preamble, no markdown fences:
        {"title": str, "body": str, "topic": str, "word_count": int}
    """
    if topic and topic.strip():
        topic_clause = (
            f"The passage must address the following topic: {topic.strip()}."
        )
    else:
        topic_clause = (
            "Choose a single topic from the IELTS Academic domains "
            f"({_IELTS_TOPIC_HINT}). Vary the topic across requests; do not "
            "default to the same domain repeatedly."
        )

    difficulty_clause = _band_difficulty_clause(difficulty_band)

    return f"""You are an IELTS Academic Reading passage writer. Produce exam-quality material in formal British English.

Output a single JSON object — no preamble, no markdown fences, no commentary:
{{"title": str, "body": str, "topic": str, "word_count": int}}

Length:
- "body" must contain 700–900 words. Count carefully; "word_count" must equal the actual word count.

Register and style:
- Formal, austere, third-person academic register. No colloquialisms, no marketing tone, no rhetorical questions to the reader.
- British spelling throughout: organise, analyse, behaviour, centre, recognise, programme, traveller, modelling, defence.
- 4–6 paragraphs, each developing a distinct idea, separated by a single blank line.
- Title: concise noun phrase, no more than 12 words. No questions, no clickbait.

Topic guidance:
{topic_clause}

Difficulty calibration — target IELTS Reading band {difficulty_band}:
{difficulty_clause}

Hard requirements:
- The passage must contain at least three discrete, citable claims that could be tested by True / False / Not Given questions.
- At least one paragraph must lend itself to a heading-matching question (i.e. it should have a single dominant idea capturable in a short heading).
- "topic" in the JSON output must be a short noun phrase summarising the domain (e.g. "urban planning", "archaeology").

Return the JSON object only.
"""


def build_questions_prompt(
    passage_body: str,
    difficulty_band: float,
) -> str:
    """
    System prompt that instructs Claude to generate exactly 9 questions for a
    supplied passage: 3 MCQ + 3 TFNG + 3 Heading Matching.

    The passage body is supplied as the user-turn message; this function only
    builds the system prompt.

    Output contract — strict JSON, no preamble, no markdown fences:
        {"questions": [ {question_type, question_text, options, correct_answer,
                         explanation, evidence_quote, order_idx}, ... ]}
    """
    difficulty_clause = _band_difficulty_clause(difficulty_band)

    return f"""You are an IELTS Academic Reading question writer. The user message contains the passage. Produce exactly 9 questions in formal British English.

Output a single JSON object — no preamble, no markdown fences:
{{
  "questions": [
    {{
      "question_type": "mcq" | "tfng" | "heading",
      "question_text": str,
      "options": list[str] | null,
      "correct_answer": str,
      "explanation": str,
      "evidence_quote": str,
      "order_idx": int
    }}
  ]
}}

Composition — exactly:
- 3 Multiple Choice (mcq) questions, order_idx 1–3
- 3 True / False / Not Given (tfng) questions, order_idx 4–6
- 3 Heading Matching (heading) questions, order_idx 7–9

Universal requirements:
- Every question must include "evidence_quote": a verbatim, contiguous substring of the passage that justifies the answer. The string must appear character-for-character in the passage (whitespace and punctuation preserved). Do not paraphrase, do not splice fragments. For "Not Given" TFNG items, "evidence_quote" should be the most-relevant passage span that is near the topic but does not actually support the claim.
- Every question must include a one-sentence "explanation" in formal British English.
- "question_text" must be self-contained and unambiguous.
- Difficulty calibration follows the passage band {difficulty_band}: {difficulty_clause}

MCQ rules:
- "options" is a list of exactly 4 strings, each beginning with a letter prefix and stop: "A. ...", "B. ...", "C. ...", "D. ...".
- "correct_answer" is exactly one of "A", "B", "C", "D".
- Distractors must be plausible — drawn from the passage's vocabulary or close paraphrases — not obviously wrong.

TFNG rules:
- "options" is null.
- "correct_answer" is exactly one of "True", "False", "Not Given".
- Across the three TFNG items, use at least two distinct answers (ideally all three).
- "Not Given" must mean genuinely absent from the passage — not merely contradicted. "False" means contradicted by the passage.

Heading Matching rules:
- "options" is a list of exactly 5 candidate headings (short noun phrases). Three of them correctly match three different paragraphs in the passage; the remaining two are plausible distractors that do not match any paragraph.
- The three heading questions share the SAME "options" list (the same 5 candidate headings). Each question asks which heading matches a specific paragraph; reference the paragraph by ordinal number in "question_text" (e.g. "Which heading best matches paragraph 2?").
- "correct_answer" is the exact heading text from "options" (not the index).

Return the JSON object only.
"""
