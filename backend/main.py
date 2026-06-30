from fastapi import FastAPI, UploadFile, File, Request, Form, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse, RedirectResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime, timedelta, timezone
from typing import Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import asyncio
import logging
import os
import json
import random
import tempfile
import re
import time
import uuid
import hmac
import hashlib
from collections import Counter
from pathlib import Path
from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

load_dotenv()

GOOGLE_TTS_API_KEY           = os.getenv("GOOGLE_TTS_API_KEY")
SUPABASE_URL                 = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY         = os.getenv("SUPABASE_SERVICE_KEY")
LEMONSQUEEZY_WEBHOOK_SECRET  = os.getenv("LEMONSQUEEZY_WEBHOOK_SECRET", "")
ADMIN_EMAILS         = {
    email.strip().lower()
    for email in os.getenv("ADMIN_EMAILS", "").split(",")
    if email.strip()
}

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
import anthropic
anthropic_client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY", ""))

# Reading module — kept in dedicated files because the prompts and validators
# are large and orthogonal to the Speaking pipeline. Imported here so the
# /reading endpoints below can call them.
from reading_prompts import build_passage_prompt, build_questions_prompt
from reading_validator import validate_passage, validate_questions

app = FastAPI()

# ── Background scheduler (APScheduler) ────────────────────────────────────────
_scheduler = BackgroundScheduler(daemon=True)

@app.on_event("startup")
async def startup_event():
    """
    On startup: prime the writing question bank in a background thread
    (non-blocking — server accepts requests immediately).
    Nightly at 03:00 UTC: top up to target_per_subtype.
    """
    _scheduler.start()
    # Nightly top-up
    _scheduler.add_job(
        pregenerate_writing_questions,
        CronTrigger(hour=3, minute=0, timezone="UTC"),
        id="nightly_writing_pregen",
        replace_existing=True,
    )
    # Startup prime (30s delay so DB connections are ready)
    _scheduler.add_job(
        pregenerate_writing_questions,
        "date",
        run_date=datetime.now(timezone.utc) + timedelta(seconds=30),
        id="startup_writing_pregen",
        replace_existing=True,
    )
    logger.info("APScheduler started: writing pregeneration scheduled")

@app.on_event("shutdown")
async def shutdown_event():
    if _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("APScheduler shut down")

limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)


# 把 Pydantic/FormData validation 失敗的細節 log 出來,否則 422 在 log 裡是黑盒。
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"422 on {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )

# Supabase admin client (service role — bypasses RLS)
supabase_admin: Client = (
    create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    if SUPABASE_URL and SUPABASE_SERVICE_KEY else None
)

WEAK_WORDS = {"very", "good", "interesting", "thing", "things", "stuff"}
ALLOWED_WEAKNESS_TAGS = {
    "weak_vocab", "safe_answer", "lack_detail", "grammar_minor", "off_topic",
}

# Tag-specific drill prompts. Keyed by weakness_tag; each entry has:
#   - expected_axis:    the axis name the LLM must return in drill_score.axis
#   - system_injection: prompt text appended verbatim to the system prompt
#                       when /process is called with mode=drill + drill_tag
# Adding a tag here automatically makes mode=drill+drill_tag=<tag> valid.
# system_injection text is treated as a tested artifact — do not paraphrase
# or collapse whitespace; replace as a whole when iterating.
DRILL_PROMPTS: dict[str, dict] = {
    "weak_vocab": {
        "expected_axis": "vocab_precision_score",
        "system_injection": """This is DRILL MODE: vocabulary precision drill.

ALL existing rules in this prompt remain in force. You will still:
- Return ONE correction object (not multiple)
- Keep why_it_hurts under 60 characters
- Keep better_phrasing_en under 60 characters
- Keep next_task under 80 characters
- Follow on_topic, off-topic, and tone-tier (A/B/C) rules unchanged
- Use the same JSON schema PLUS the drill_score field

DRILL MODE ADDS three sharpening constraints to your correction:

1. correction.quoted MUST include at least one safe-word phrase
   the user actually said. See the safe-word definition below.
   Quote the user's exact phrasing containing the safe-word.

2. correction.better_phrasing_en MUST be a single B2+ word or phrase
   that directly replaces the safe-word in correction.quoted.
   No alternatives, no slash-separated options.

3. correction.next_task MUST instruct the user to re-record this
   answer replacing the specific safe-word with the suggested
   replacement. Use this format pattern (in 繁中):
   「重講一次,把 `[safe-word phrase]` 換成 `[B2+ replacement]`.」

DRILL MODE ADDS one new output field: drill_score.

drill_score is computed in TWO STEPS. Follow them strictly.

================================================================
SAFE-WORD DEFINITION (read this before STEP 1)
================================================================

The CORE safe-words are: very, good, nice, interesting, thing, stuff.
These appear in 90% of weak-vocab cases.

Additionally, IELTS examiners mark down these words when used as
low-precision filler:
- "really" used as intensifier (really good, really serious, really tired)
- "major" used as generic emphasis (major problem, major issue)
- "a lot of" / "lots of" without specifics
- "kind of" / "sort of" hedging
- generic adjectives like "important", "serious" used without
  concrete framing

If you find any such word being used as low-precision filler in the
user's transcript, INCLUDE it in safe_words_found.

Threshold for inclusion: would an IELTS examiner mark this word
down for being too vague in the user's specific context?
  - Yes → include in safe_words_found
  - No → do not include

Examples of context-dependent judgment:
- "It's a major historical landmark" — "major" is precise, EXCLUDE
- "It's a major problem" — "major" is filler, INCLUDE
- "I really studied hard" — "really" is precise emphasis, EXCLUDE
- "I really like it" — "really" is filler, INCLUDE

================================================================
STEP 1 — EVIDENCE GATHERING (drill_score.evidence)
================================================================

Before scoring, you MUST list the actual words/phrases you found
in the user's transcript. Do not estimate. Do not paraphrase.
Quote exact occurrences.

Discipline: mentally re-read the transcript word by word. For each
occurrence of a safe-word (core or extended), add it to the array.

Pay special attention to REPEATED forms:
- "good race" + "Good job" = 2 entries of "good", not 1
- "very big" + "very huge" = 2 entries of "very", not 1
- "really sleepy" + "really tired" + "really tired" + "really looks"
  = 4 entries of "really", not 1 or 2
Capitalization does not matter ("Good" and "good" are the same safe-word).

  evidence.safe_words_found:
    Array of strings. List every occurrence of a safe-word
    (core or extended) in the order they appear.
    Each occurrence is a separate array entry.
    Example: ["very", "very", "good", "very", "interesting"]
    If no safe-words appear, return an empty array [].

  evidence.b2_plus_found:
    Array of strings. List every B2+ vocabulary item the user
    actually spoke in their transcript.

    To produce this list: mentally re-read the user's transcript
    word by word, and identify B2+ vocabulary items present in
    their speech. Add each one to the array.

    A word counts as B2+ if it would be tagged B2 or above on the
    CEFR scale (e.g. iconic, distinctive, mesmerising, alarming,
    captivating, mouth-watering, exhausted, lingering, impressive,
    enormous, fascinating, adore, appreciate).

    Common words like "good", "nice", "important", "serious",
    "interesting", "major", "really" are NOT B2+ even if they
    sound formal. They are safe-words.

    If the transcript contains zero B2+ items, return an empty array [].

================================================================
JSON STRUCTURE RULE (read before writing evidence)
================================================================

Each item in the evidence arrays must be a single word or short
phrase from the transcript. Each entry stands alone — no commentary,
no annotations, no parenthetical notes attached to entries.

Format guidance:
"b2_plus_found": ["effective"]
"b2_plus_found": ["iconic", "lingering"]
"b2_plus_found": []   ← empty if no B2+ items

If your scoring logic involves nuance (e.g. you decided one word
is borderline), express that nuance in the feedback string, not
in the evidence arrays. Evidence arrays are pure listings of
transcript words — that is their only purpose.

================================================================
STEP 2 — SCORING (drill_score.score)
================================================================

Use this rubric and the evidence you gathered:

  Band 90-100 (Mastery)
  - len(b2_plus_found) >= 4
  - len(safe_words_found) == 0

  Band 70-89 (Strong)
  - len(b2_plus_found) is 2 or 3
  - len(safe_words_found) <= 1

  Band 50-69 (Mixed)
  - len(b2_plus_found) is 1 or 2
  - len(safe_words_found) is 2 or 3

  Band 30-49 (Weak)
  - len(b2_plus_found) <= 1
  - len(safe_words_found) is 4 or more,
    OR same safe-word repeated 3+ times

  Band 0-29 (Critical)
  - len(b2_plus_found) == 0
  - len(safe_words_found) >= 5

If the transcript falls between two bands, pick the LOWER one.
Pick a specific number within the chosen band based on severity.

================================================================
STEP 3 — DERIVE FEEDBACK (drill_score.feedback)
================================================================

Based on score, the band label is FIXED:
- score 0-29  → "Critical"
- score 30-49 → "Weak"
- score 50-69 → "Mixed"
- score 70-89 → "Strong"
- score 90-100 → "Mastery"

The band label MUST match the score range. Do not pick a band
inconsistent with your score.

feedback format (繁中, 50 字內, strict template):
"safe-words: <count> 個 (<list>); B2+: <count> 個 (<list 或 無>); 落在 <Band> band."

Examples:
- "safe-words: 5 個 (very×5); B2+: 0 個 (無); 落在 Critical band."
- "safe-words: 2 個 (very, good); B2+: 1 個 (impressive); 落在 Mixed band."
- "safe-words: 0 個; B2+: 3 個 (iconic, mesmerising, lingering); 落在 Strong band."

DO NOT add encouragement. DO NOT add suggestions. Only describe.

threshold_passed = (score >= 70).
axis = "vocab_precision_score".

================================================================
OUTPUT SCHEMA IN DRILL MODE
================================================================
{
  "correction": { ... existing fields per baseline rules ... },
  "tag": "weak_vocab" | "lack_detail" | etc per baseline rules,
  "progress_note": "" or per Tier-B/First-Touch rules,
  "on_topic": true | false,
  "drill_score": {
    "axis": "vocab_precision_score",
    "evidence": {
      "safe_words_found": ["..."],
      "b2_plus_found": ["..."]
    },
    "score": <int 0-100>,
    "feedback": "<繁中,50字內,strict template>",
    "threshold_passed": <bool>
  }
}

If on_topic is false (off-topic answer), drill_score still outputs
but score = 0, evidence arrays may be empty, threshold_passed = false.
Off-topic rule from baseline still wins for correction content."""
    },
    "lack_detail": {
        "expected_axis": "detail_density_score",
        "system_injection": """This is DRILL MODE: concrete detail drill.

ALL existing rules in this prompt remain in force. You will still:
- Return ONE correction object (not multiple)
- Keep why_it_hurts under 60 characters
- Keep better_phrasing_en under 60 characters
- Keep next_task under 80 characters
- Follow on_topic, off-topic, and tone-tier (A/B/C) rules unchanged
- Use the same JSON schema PLUS the drill_score field

DRILL MODE ADDS three sharpening constraints to your correction:

1. correction.quoted MUST be an abstract phrase from the user's
   transcript that lacks a concrete detail dimension.
   Quote the user's exact abstract phrasing.

2. correction.better_phrasing_en MUST inject ONE concrete dimension
   into the user's phrase. Pick ONE of: time / place / number /
   sense / person. The replacement must contain a specific element
   (e.g. "Last Tuesday", "鼎泰豐", "80 dollars", "smell of cinnamon",
   "my grandmother"). No abstract upgrades.

3. correction.next_task MUST instruct the user to re-record using
   that specific dimension. Use this format pattern (in 繁中):
   「重講一次,在第一句就放一個[時間/地點/數字/感官/人物].」

DRILL MODE ADDS one new output field: drill_score.

drill_score is computed in TWO STEPS. Follow them strictly.

================================================================
DIMENSION DEFINITION (read this before STEP 1)
================================================================

A "concrete dimension" is one of these five categories. Each
category requires a SPECIFIC element from the user's transcript,
not a generic mention.

  時間 (time): when, year, month, time-of-day, duration
    SPECIFIC: "Last Tuesday", "in 2018", "around 7am", "for 3 hours"
    GENERIC (does NOT count): "yesterday", "sometimes", "often"

  地點 (place): named or specifically-described location
    SPECIFIC: "鼎泰豐", "Shanghai", "the night market in 信義區"
    GENERIC (does NOT count): "a restaurant", "outside", "somewhere"

  數字 (number): price, quantity, frequency, age, count
    SPECIFIC: "80 dollars", "40,000 people", "twice a week", "age 25"
    GENERIC (does NOT count): "many people", "a few times", "expensive"

  感官 (sense): smell, sound, texture, color, taste, sight
    SPECIFIC: "the smell of cinnamon", "the sound of waves",
              "creamy texture", "bright red"
    GENERIC (does NOT count): "it was nice", "it tasted good"

  人物 (person): named or specifically-described individual
    SPECIFIC: "Charles Lelec", "my grandmother", "the waiter who..."
    GENERIC (does NOT count): "my friend", "people", "someone"

A dimension counts as PRESENT only when the user's transcript
contains a specific element, not a generic placeholder.

================================================================
STEP 1 — EVIDENCE GATHERING (drill_score.evidence)
================================================================

Before scoring, you MUST list the actual specific elements you
found in the user's transcript, sorted into the five dimensions.

Discipline: mentally re-read the transcript word by word. For each
specific element you find, add it to the matching array.

  evidence.time_dimensions_found:
    Array of strings. List specific time references from the
    transcript. Each entry is one element from the user's speech.
    Example: ["last Tuesday", "around 7am"]
    If no specific time references appear, return [].

  evidence.place_dimensions_found:
    Array of strings. List specific place references.
    Example: ["Shanghai", "Monaco"]
    If none, return [].

  evidence.number_dimensions_found:
    Array of strings. List specific numeric references.
    Example: ["40,000", "thousands of motorcycles"]
    If none, return [].

  evidence.sense_dimensions_found:
    Array of strings. List specific sensory references.
    Example: ["smell of cinnamon", "loud traffic noise"]
    If none, return [].

  evidence.person_dimensions_found:
    Array of strings. List specific person references.
    Example: ["Charles Lelec", "my grandmother"]
    If none, return [].

================================================================
JSON STRUCTURE RULE (read before writing evidence)
================================================================

Each item in the evidence arrays must be a single word or short
phrase from the transcript. Each entry stands alone — no commentary,
no annotations, no parenthetical notes attached to entries.

Format guidance:
"time_dimensions_found": ["last Tuesday"]
"place_dimensions_found": ["Shanghai", "Monaco"]
"sense_dimensions_found": []   ← empty if no sensory details

If your scoring logic involves nuance (e.g. you decided a phrase
is borderline-specific), express that nuance in the feedback string,
not in the evidence arrays. Evidence arrays are pure listings of
transcript phrases — that is their only purpose.

================================================================
STEP 2 — SCORING (drill_score.score)
================================================================

Count how many of the 5 dimensions have at least 1 entry. Call
this dimensions_present.

Use this rubric:

  Band 90-100 (Mastery)
  - dimensions_present == 5
  - At least one dimension has vivid, specific detail

  Band 70-89 (Strong)
  - dimensions_present == 4
  - Specifics are reasonably concrete

  Band 50-69 (Mixed)
  - dimensions_present == 3
  - Specifics may be thin but real

  Band 30-49 (Weak)
  - dimensions_present == 2
  - Otherwise abstract

  Band 0-29 (Critical)
  - dimensions_present <= 1
  - Answer is largely abstract ("it was nice", "we had fun")

If the transcript falls between two bands, pick the LOWER one.
Pick a specific number within the chosen band based on richness.

================================================================
STEP 3 — DERIVE FEEDBACK (drill_score.feedback)
================================================================

Based on score, the band label is FIXED:
- score 0-29  → "Critical"
- score 30-49 → "Weak"
- score 50-69 → "Mixed"
- score 70-89 → "Strong"
- score 90-100 → "Mastery"

The band label MUST match the score range. Do not pick a band
inconsistent with your score.

feedback format (繁中, 50 字內, strict template):
"時間: <count> | 地點: <count> | 數字: <count> | 感官: <count> | 人物: <count> → 落在 <Band> band."

Examples:
- "時間: 0 | 地點: 0 | 數字: 0 | 感官: 0 | 人物: 0 → 落在 Critical band."
- "時間: 1 | 地點: 1 | 數字: 1 | 感官: 0 | 人物: 0 → 落在 Mixed band."
- "時間: 2 | 地點: 1 | 數字: 1 | 感官: 1 | 人物: 1 → 落在 Strong band."

DO NOT add encouragement. DO NOT add suggestions. Only describe.

threshold_passed = (score >= 60).
axis = "detail_density_score".

================================================================
OUTPUT SCHEMA IN DRILL MODE
================================================================
{
  "correction": { ... existing fields per baseline rules ... },
  "tag": "weak_vocab" | "lack_detail" | etc per baseline rules,
  "progress_note": "" or per Tier-B/First-Touch rules,
  "on_topic": true | false,
  "drill_score": {
    "axis": "detail_density_score",
    "evidence": {
      "time_dimensions_found": ["..."],
      "place_dimensions_found": ["..."],
      "number_dimensions_found": ["..."],
      "sense_dimensions_found": ["..."],
      "person_dimensions_found": ["..."]
    },
    "score": <int 0-100>,
    "feedback": "<繁中,50字內,strict template>",
    "threshold_passed": <bool>
  }
}

If on_topic is false (off-topic answer), drill_score still outputs
but score = 0, evidence arrays may all be empty, threshold_passed = false.
Off-topic rule from baseline still wins for correction content."""
    },
}
# MIRROR of frontend WEAKNESS_LABELS at frontend/app/index.html — keep in sync when modified
WEAKNESS_LABELS = {
    "weak_vocab":    "用字太安全",
    "safe_answer":   "答得太保險",
    "lack_detail":   "缺少具體細節",
    "grammar_minor": "小文法",
    "off_topic":     "答非所問",
}

QUESTION_EXCLUSION_DAYS = 3
FREE_DRILL_QUOTA = 20
DRILL_QUOTA_WINDOW_DAYS = 7

PERSONA_PROMPTS: dict[str, str] = {
    "A": (
        "You are a patient and encouraging coach working with a beginner (IELTS Band 4 or below). "
        "This student is brave for speaking at all. Your priorities:\n"
        "1. ALWAYS find something genuine to praise first — a complete sentence, a correct word, "
        "an attempt at detail. Be specific, not generic.\n"
        "2. Correct ONLY the single most important thing. Nothing else.\n"
        "3. Tone: warm, safe, never clinical. This student must feel capable of continuing.\n"
        "Ratio: 70% encouragement, 30% correction."
    ),
    "B": (
        "You are a patient tutor working with an intermediate student (IELTS Band 5). "
        "This student is making real progress. Your priorities:\n"
        "1. Acknowledge what worked — structure, vocabulary, effort.\n"
        "2. Correct up to two things, delivered gently.\n"
        "3. Tone: supportive but honest. Build confidence while raising the bar slowly.\n"
        "Ratio: 50% encouragement, 50% correction."
    ),
    "C": (
        "You are a fair and precise teacher working with a capable student (IELTS Band 6). "
        "This student can handle direct feedback. Your priorities:\n"
        "1. Brief acknowledgment of strengths — one sentence only.\n"
        "2. Correct what matters for Band 7 progression. Be specific.\n"
        "3. Tone: professional, respectful, no fluff.\n"
        "Ratio: 30% encouragement, 70% correction."
    ),
    "D": (
        "You are an IELTS examiner working with an advanced student (IELTS Band 7+). "
        "This student wants precision, not comfort. Your priorities:\n"
        "1. Skip generic praise. Only note genuine strengths if present.\n"
        "2. Identify all significant issues. Be thorough.\n"
        "3. Tone: examiner-level, objective, exact.\n"
        "Ratio: 10% encouragement, 90% correction."
    ),
}

# Vocabulary SRS — days until next review by srs_level (0..5)
VOCAB_SRS_SCHEDULE = {0: 0, 1: 1, 2: 3, 3: 7, 4: 14, 5: 30}
VOCAB_VALID_RESULTS = {"again", "hard", "good", "easy"}
STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "because", "but", "by",
    "do", "for", "from", "get", "go", "had", "has", "have", "he", "her",
    "him", "his", "how", "i", "if", "in", "into", "is", "it", "its", "just",
    "me", "my", "of", "on", "or", "our", "she", "so", "than", "that", "the",
    "their", "them", "they", "this", "to", "us", "very", "was", "we", "were",
    "what", "when", "where", "which", "who", "why", "will", "with", "you",
    "your"
}
EXPANSION_SIGNAL_PATTERNS = [
    re.compile(r"\bbecause\b"),
    re.compile(r"\bwhen\b"),
    re.compile(r"\bwhere\b"),
    re.compile(r"\bwho\b"),
    re.compile(r"\bwhich\b"),
    re.compile(r"\bthat\b"),
    re.compile(r"\bsince\b"),
    re.compile(r"\busually\b"),
    re.compile(r"\brecently\b"),
    re.compile(r"\bfor example\b"),
    re.compile(r"\bfor instance\b"),
    re.compile(r"\bweekend\b"),
    re.compile(r"\bmorning\b"),
    re.compile(r"\bnight\b"),
    re.compile(r"\bwith\b"),
    re.compile(r"\bafter\b"),
    re.compile(r"\bbefore\b"),
]
SAFE_PATTERNS = [
    re.compile(r"\bit depends\b"),
    re.compile(r"\bi think it is good\b"),
    re.compile(r"\bi think it's good\b"),
    re.compile(r"\bi think so\b"),
    re.compile(r"\bi like it because it is convenient\b"),
    re.compile(r"\bi like it because it's convenient\b"),
]


def verify_token(authorization: Optional[str]) -> str:
    """Verify Supabase Bearer JWT and return the authenticated user id."""
    if not supabase_admin:
        raise HTTPException(status_code=503, detail="Auth service not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing or invalid authorization header")

    token = authorization.split(" ", 1)[1].strip()
    if not token:
        raise HTTPException(status_code=401, detail="Missing bearer token")

    try:
        resp = supabase_admin.auth.get_user(token)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=401, detail="Token verification failed") from exc

    user = getattr(resp, "user", None)
    if not user or not getattr(user, "id", None):
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    return user.id


def verify_admin(authorization: Optional[str]) -> str:
    user_id = verify_token(authorization)
    try:
        response = supabase_admin.auth.admin.get_user_by_id(user_id)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to verify admin access") from exc

    user = getattr(response, "user", None)
    email = (getattr(user, "email", None) or "").strip().lower()
    if email not in ADMIN_EMAILS:
        raise HTTPException(status_code=403, detail="Admin access required")

    return user_id


def get_user_recent_records(user_id: str, limit: int = 10) -> list[dict]:
    if not supabase_admin:
        raise HTTPException(status_code=503, detail="Database service not configured")

    try:
        response = (
            supabase_admin.table("practice_records")
            .select("user_transcript, topic, question, created_at, weakness_tag, better_expression, coach_response")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Failed to load practice history") from exc

    return response.data or []


def tokenize_words(text: str) -> list[str]:
    return re.findall(r"\b[a-zA-Z']+\b", (text or "").lower())


def has_pattern_match(text: str, patterns: list[re.Pattern]) -> bool:
    return any(pattern.search(text) for pattern in patterns)


def extract_dynamic_weak_words_from_history(
    transcripts: list[str],
    max_items: int = 5,
    min_ratio: float = 0.3,
) -> list[str]:
    non_empty_transcripts = [transcript for transcript in transcripts if transcript]
    if not non_empty_transcripts:
        return []

    doc_counts = Counter()
    total_counts = Counter()

    for transcript in non_empty_transcripts:
        tokens = tokenize_words(transcript)
        filtered = [
            token for token in tokens
            if len(token) < 10 and token not in STOPWORDS and token.isalpha()
        ]
        for token in filtered:
            total_counts[token] += 1
        for token in set(filtered):
            doc_counts[token] += 1

    transcript_count = len(non_empty_transcripts)
    ranked = sorted(
        (
            token for token, doc_count in doc_counts.items()
            if (doc_count / transcript_count) >= min_ratio
        ),
        key=lambda token: (doc_counts[token], total_counts[token]),
        reverse=True,
    )
    return ranked[:max_items]


def extract_weak_patterns(transcripts: list[str], max_items: int = 3) -> list[str]:
    counts = Counter()
    # NOTE: dynamic weak words are NOT mixed into prompt path.
    # They are observability-only in memory_snapshot.
    for transcript in transcripts:
        if not transcript:
            continue
        words = tokenize_words(transcript)
        for word in words:
            if word in WEAK_WORDS:
                counts[word] += 1

    ranked = [
        word
        for word, count in counts.most_common()
        if count >= 1
    ]
    return ranked[:max_items]


def detect_repeated_weak_words(
    user_text: str,
    weak_patterns: Optional[list[str]] = None,
) -> list[str]:
    """
    Detect which weak words from the user's history show up again in this answer.
    Used for prompt injection (memory system), NOT for tagging.

    Tagging (weakness_tag) is now done by Groq itself via the JSON response
    and validated against ALLOWED_WEAKNESS_TAGS in /process.
    """
    if not weak_patterns:
        return []
    text = (user_text or "").strip().lower()
    return [
        word for word in weak_patterns
        if re.search(rf"\b{re.escape(word.lower())}\b", text)
    ]


def count_weak_patterns(
    transcripts: list[str], words: list[str]
) -> dict[str, int]:
    """
    Count the total number of times each weak word appears across the user's
    past transcripts. Preserves the input ordering of `words` in the returned
    dict so Groq sees the most-frequent-first ordering from extract_weak_patterns.
    """
    if not words:
        return {}
    word_set = set(words)
    totals: dict[str, int] = {word: 0 for word in words}
    for transcript in transcripts:
        if not transcript:
            continue
        for token in tokenize_words(transcript):
            if token in word_set:
                totals[token] += 1
    return totals


def count_tag_patterns(records: list[dict]) -> dict[str, int]:
    valid_tags = {"weak_vocab", "safe_answer", "lack_detail", "grammar_minor", "off_topic"}
    counts = {}
    for r in records:
        tag = r.get("weakness_tag")
        if tag in valid_tags:
            counts[tag] = counts.get(tag, 0) + 1
    return counts


def is_intensity_calibration_enabled() -> bool:
    """
    Feature flag for the three-tier intensity calibration prompt change.

    Defaults to False — production behavior is unchanged unless
    INTENSITY_CALIBRATION_ENABLED is explicitly set to 'true' in env.

    Use case: gates new prompt logic for safe rollout. Mock-test with flag on,
    flip on for real users only after mock outputs verified.
    """
    return os.getenv("INTENSITY_CALIBRATION_ENABLED", "false").strip().lower() == "true"


def pick_next_question(current_topic: str, current_question: str, user_id: str) -> str:
    """
    Pick a next question biased toward the same topic as the current one,
    excluding the question just answered. Falls back to any unrecent question.
    Returns question text string, or "" on any failure.
    """
    try:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=QUESTION_EXCLUSION_DAYS)
        ).isoformat()
        recent_resp = (
            supabase_admin.table("practice_records")
            .select("question")
            .eq("user_id", user_id)
            .gte("created_at", cutoff)
            .execute()
        )
        recent_set = {
            (row.get("question") or "").strip()
            for row in (recent_resp.data or [])
            if (row.get("question") or "").strip()
        }
        recent_set.add(current_question.strip())

        questions_resp = (
            supabase_admin.table("questions")
            .select("text, topic, part")
            .eq("part", 1)
            .execute()
        )
        questions = questions_resp.data or []
        if not questions:
            return ""

        # Tier 1 — same topic, not recently practiced
        same_topic = [
            q for q in questions
            if (q.get("topic") or "").strip().lower() == current_topic.strip().lower()
            and (q.get("text") or "").strip() not in recent_set
        ]
        if same_topic:
            return random.choice(same_topic).get("text", "")

        # Tier 2 — any topic, not recently practiced
        not_recent = [
            q for q in questions
            if (q.get("text") or "").strip() not in recent_set
        ]
        if not_recent:
            return random.choice(not_recent).get("text", "")

        # Tier 3 — fallback random
        return random.choice(questions).get("text", "")
    except Exception:
        logger.exception("pick_next_question failed, returning empty")
        return ""


def build_witness_note(total_practice_count: int, tag_counts: dict, current_tag: str) -> dict:
    """
    Returns {"text": str, "is_milestone": bool}.
    text is empty string when there is nothing to say.
    """
    MILESTONES = {10, 20, 30, 50, 100}

    if total_practice_count in MILESTONES:
        return {
            "text": f"你今天完成了第 {total_practice_count} 次練習。",
            "is_milestone": True,
        }

    if current_tag and tag_counts.get(current_tag, 0) >= 3:
        count = tag_counts[current_tag]
        return {
            "text": f"這個問題你已經碰了 {count} 次。卡住不是能力問題，是還沒找到那個說法。今天繼續。",
            "is_milestone": False,
        }

    if total_practice_count > 0:
        return {
            "text": f"你已累計練習 {total_practice_count} 次。",
            "is_milestone": False,
        }

    return {"text": "", "is_milestone": False}


def build_memory_block(weak_pattern_counts: dict[str, int], tag_counts: dict[str, int] | None = None) -> str:
    """
    Render the 【使用者歷史弱點】 block that gets stitched into the system prompt.

    The count next to each word is what lets the LLM switch tone tiers:
      - count >= 3 + user uses it again → tier C (ask, don't lecture)
      - user avoided a historied word this turn → tier B (specific praise)
      - otherwise → tier A (default)

    These rules are spelled out textually here so the few-shot examples aren't
    the only carrier.
    """
    block = ""
    if weak_pattern_counts:
        lines = [
            f"- '{word}' 已出現 {count} 次"
            for word, count in weak_pattern_counts.items()
        ]
        block += (
            "\n【使用者歷史弱點】\n"
            + "\n".join(lines) + "\n"
            "- 若某個詞的 count 已達 3 次以上，且學生這次又用了 → 使用層級 C（直球問，不要訓話）\n"
            "- 若某個詞在歷史出現過，但學生這次沒再用，且回答中有具體細節出現 → 使用層級 B（具體肯定那個改變）\n"
            "- 其他情況 → 使用層級 A（預設語氣）\n"
            "- Blabby 鐵律不變：一次只點一個最關鍵的問題。\n"
        )
    if tag_counts:
        sorted_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)
        tag_lines = "\n".join(f"- {tag}: {count} 次" for tag, count in sorted_tags)
        block += f"\n【使用者歷史弱點類型】\n{tag_lines}\n如果這次的回答沒有犯最常見的弱點類型，且有具體細節，優先使用層級 B 肯定這個進步。\n"
    return block


def build_repair_memory(records: list[dict]) -> str:
    """
    Render the 【使用者近期被修復的說法】 block from past corrections.

    Pairs each historically-quoted mistake (extracted from coach_response,
    format 你說：「{quoted}」) with the better_expression the coach taught
    for it, so the LLM can recognise a repeated structural error returning.

    Scope: only structural tags (grammar_minor / weak_vocab / safe_answer).
    lack_detail is situational advice that doesn't transfer to a new
    question; off_topic carries no reusable phrasing. Most-recent-first
    (records arrive created_at DESC), deduped by quoted snippet so the same
    mistake never floods the block, capped at 5 lines. No is_pro gate here —
    the caller decides whether to inject.
    """
    structural_tags = {"grammar_minor", "weak_vocab", "safe_answer"}
    lines: list[str] = []
    seen: set[str] = set()
    for r in records:
        if r.get("weakness_tag") not in structural_tags:
            continue
        better = (r.get("better_expression") or "").strip()
        if not better:
            continue
        match = re.search(r"你說：「(.+?)」", r.get("coach_response") or "")
        if not match:
            continue
        quoted = match.group(1).strip()
        if not quoted or quoted in seen:
            continue
        seen.add(quoted)
        lines.append(f"- 曾說「{quoted}」，教過更好的說法：「{better}」")
        if len(lines) >= 5:
            break
    if not lines:
        return ""
    return "\n【使用者近期被修復的說法】\n" + "\n".join(lines) + "\n"


def build_intensity_block(
    total_practice_count: int,
    historical_top_tag: str,
    historical_top_tag_count: int,
) -> str:
    """
    Determine feedback intensity level based on two axes:
    - Global trust: how many sessions has this user logged
    - Tag familiarity: how many times has the historical top tag been flagged

    Take the SOFTER of the two axes — being too harsh too early
    is more damaging than being too soft too late.

    Returns a system-prompt directive that overrides the base A/B/C tier logic
    in build_system_prompt(). Empty string if feature flag is off.
    """
    if not is_intensity_calibration_enabled():
        return ""

    # Global trust axis
    if total_practice_count < 3:
        global_intensity = "first_touch"
    elif total_practice_count < 8:
        global_intensity = "calibration"
    else:
        global_intensity = "direct"

    # Tag familiarity axis (uses historical top tag as proxy —
    # actual weakness_tag for THIS turn is generated by Groq later)
    if historical_top_tag_count == 0:
        tag_intensity = "first_touch"
    elif historical_top_tag_count <= 2:
        tag_intensity = "calibration"
    else:
        tag_intensity = "direct"

    # Take softer side: first_touch < calibration < direct
    rank = {"first_touch": 0, "calibration": 1, "direct": 2}
    final_rank = min(rank[global_intensity], rank[tag_intensity])
    final_intensity = ["first_touch", "calibration", "direct"][final_rank]

    if final_intensity == "first_touch":
        return f"""
【本次力道強制指令：First Touch（建立信任）】
這位使用者是新用戶（總練習 {total_practice_count} 次）或本系統還沒看過這類錯誤。
- 必須先引用使用者原句的某個具體片段，並指出這個片段為什麼有效（例如某個動詞用得準、某個轉折自然）。
- 找不到任何具體優點時，跳過稱讚，直接進入批改但語氣保持中性、不銳利。
- 然後才指出本次唯一痛點。批改要解釋「為什麼這樣傷害表達」，不要只說「這樣不好」。
- 用觀察句而非評判句：「我注意到你這 60 秒裡有 5 個 very good」優於「你用了 5 次 very good」。
- 不要使用「又」「再次」「還是」這類預設使用者已被提醒過的詞。
- progress_note 欄位填入那個具體優點。
這條指令凌駕於 base prompt 裡的 A/B/C 層級判斷。
"""
    elif final_intensity == "calibration":
        return f"""
【本次力道強制指令：Calibration（直接但給機制）】
使用者已練習 {total_practice_count} 次，系統認識他了，但這類錯誤還不算反覆。
- 直接點名痛點，不需要先稱讚。
- 但批改必須附帶機制解釋——不只是「這樣不好」，要說明「為什麼這個錯誤傷害表達」。
- 語氣是「在觀察」，不是「在糾正」。
- 如果層級判斷為 B（答案有具體細節或弱詞消失），progress_note 填入一句具體觀察，例如「你這次補了時間和地點，畫面立起來了」。否則留空。
這條指令凌駕於 base prompt 裡的 A 層級指令；B/C 層級的觸發條件如果同時成立，依然優先（見下方協調規則）。
"""
    else:
        return f"""
【本次力道強制指令：Direct（無情但簡短）】
使用者已練習 {total_practice_count} 次，這類錯誤已被點過 {historical_top_tag_count} 次。
- 直接、簡短、不解釋機制。他已經知道為什麼了。
- 不要說「我們之前提過」「這已經是第 N 次」這種訓話語言。
- correction.why_it_hurts 一句話到位，correction.next_task 一句帶過，不要展開。
- progress_note 欄位留空。
這條指令凌駕於 base prompt 裡的 A 層級指令；B/C 層級的觸發條件如果同時成立，依然優先（見下方協調規則）。
"""


def _err_resp(message: str, status: int = 500):
    """Consistent JSON error response helper."""
    return {"error": message, "status": status}


_ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS", "https://blabby.vercel.app").split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_ALLOWED_ORIGINS,
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=False,
    expose_headers=["X-Script-Bytes"],
)


def run_groq(messages):
    """
    Call Groq with bounded retries for JSON-shaped failures only.

    Retry budget: 3 attempts total (initial + 2 retries) with exponential
    backoff 1s → 2s between attempts. We retry two failure modes:
      (a) the response body fails our local JSON parse / shape check
      (b) Groq's server-side validator rejects the model output with
          error code `json_validate_failed` (raised as a Groq API error)

    Other Groq errors (auth, rate limit, network, server) propagate
    immediately — those won't be fixed by retrying with the same prompt.
    """
    max_attempts = 3
    last_error = None
    for attempt in range(max_attempts):
        try:
            response = groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                response_format={"type": "json_object"}
            )
            content = ((response.choices[0].message.content) or "").strip()
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("Groq returned non-object JSON")
            return parsed
        except (json.JSONDecodeError, ValueError, AttributeError, IndexError, TypeError) as exc:
            last_error = exc
            logger.warning(
                "run_groq parse attempt %s/%s failed: %s",
                attempt + 1, max_attempts, exc,
            )
        except Exception as exc:
            # Groq raises BadRequestError(code="json_validate_failed") when
            # the model's output fails server-side json_object validation.
            # Treat that the same as a local parse failure for retry; let
            # everything else (auth, 429, 5xx, network) bubble up so callers
            # see the real cause instead of a coerced 503.
            error_code = getattr(exc, "code", "") or ""
            if error_code != "json_validate_failed" and "json_validate_failed" not in str(exc):
                raise
            last_error = exc
            logger.warning(
                "run_groq json_validate_failed attempt %s/%s: %s",
                attempt + 1, max_attempts, exc,
            )
        if attempt < max_attempts - 1:
            # Exponential backoff between attempts: 1s before retry #1,
            # 2s before retry #2. No sleep after the final attempt.
            time.sleep(2 ** attempt)
    raise HTTPException(
        status_code=503, detail="批改服務暫時不可用，請重試"
    ) from last_error


def run_claude(messages: list[dict]) -> dict:
    """
    Claude Sonnet 4.6 for feedback + drill scoring.
    Retry logic mirrors run_groq(): 3 attempts, exponential backoff 1s→2s.
    Returns parsed dict. Raises on final failure.
    """
    max_attempts = 3
    last_error = None
    for attempt in range(max_attempts):
        try:
            # Claude API: system message is separate from messages array
            system_content = next(
                (m["content"] for m in messages if m["role"] == "system"), ""
            )
            user_messages = [m for m in messages if m["role"] != "system"]
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=system_content,
                messages=user_messages,
            )
            content = (response.content[0].text or "").strip()
            # Strip ```json fences if present
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
            parsed = json.loads(content)
            if not isinstance(parsed, dict):
                raise ValueError("Claude returned non-object JSON")
            return parsed
        except (json.JSONDecodeError, ValueError, AttributeError, IndexError, TypeError) as exc:
            last_error = exc
            if attempt < max_attempts - 1:
                time.sleep(2 ** attempt)
            continue
        except Exception:
            raise
    raise last_error


def validate_correction_response(
    data: dict,
    expected_drill_axis: Optional[str] = None,
) -> tuple[bool, str]:
    """
    Validate the LLaMA response shape against the new structured contract.
    Returns (is_valid, error_reason). Reasons are for logging only,
    not exposed to the user.

    Rules:
    1. data must be a dict
    2. data["correction"] must exist and be a dict (NOT array/list/string)
    3. correction must have all five fields: quoted / why_it_hurts /
       better_phrasing_en / better_phrasing_zh / next_task, each a string
    4. quoted, why_it_hurts, next_task must always be non-empty.
       better_phrasing_en + better_phrasing_zh may be empty ONLY when
       on_topic=false (off-topic carve-out). When on_topic=true, BOTH
       phrasing fields must be non-empty.
    5. why_it_hurts <= 60 characters (len(text), Chinese counts as 1 each)
    6. better_phrasing_en <= 60 characters (when non-empty)
    7. better_phrasing_zh <= 30 characters (when non-empty)
    8. next_task <= 80 characters
    9. When expected_drill_axis is set (drill mode), data["drill_score"] must
       exist as a dict with axis (str matching expected), score (int 0-100),
       feedback (non-empty str), threshold_passed (bool), AND a structured
       evidence sub-object whose required keys depend on the axis:
        - vocab_precision_score: evidence must be a dict containing
          "safe_words_found" and "b2_plus_found" (each a list of strings;
          may be empty).
        - detail_density_score: evidence must be a dict containing
          "time_dimensions_found", "place_dimensions_found",
          "number_dimensions_found", "sense_dimensions_found",
          "person_dimensions_found" (each a list of strings; may be empty).
       List items are NOT validated against the user's transcript — we
       trust the LLM's judgment and accept ~10-20% inaccuracy. List item
       count is NOT checked against drill_score.score — backend trusts
       whatever band the LLM picked.
       When expected_drill_axis is None, drill_score is not checked — its
       presence or absence is ignored for non-drill turns.

    This is the inner validation layer. Connection-level / JSON-parse retries
    live inside run_groq() and are independent.
    """
    if not isinstance(data, dict):
        return False, f"response is {type(data).__name__}, not dict"

    correction = data.get("correction")
    if correction is None:
        return False, "correction field missing"
    if not isinstance(correction, dict):
        return False, f"correction is {type(correction).__name__}, not dict"

    required_fields = (
        "quoted",
        "why_it_hurts",
        "better_phrasing_en",
        "better_phrasing_zh",
        "next_task",
    )
    for field in required_fields:
        if field not in correction:
            return False, f"correction.{field} missing"
        if not isinstance(correction[field], str):
            return False, f"correction.{field} is {type(correction[field]).__name__}, not string"

    quoted             = correction["quoted"].strip()
    why_it_hurts       = correction["why_it_hurts"].strip()
    better_phrasing_en = correction["better_phrasing_en"].strip()
    better_phrasing_zh = correction["better_phrasing_zh"].strip()
    next_task          = correction["next_task"].strip()

    on_topic = data.get("on_topic", True)

    if not quoted:
        return False, "correction.quoted is empty"
    if not why_it_hurts:
        return False, "correction.why_it_hurts is empty"
    if not next_task:
        return False, "correction.next_task is empty"
    if on_topic:
        if not better_phrasing_en:
            return False, "correction.better_phrasing_en is empty (only allowed when on_topic=false)"
        if not better_phrasing_zh:
            return False, "correction.better_phrasing_zh is empty (only allowed when on_topic=false)"

    if len(why_it_hurts) > 60:
        return False, f"why_it_hurts is {len(why_it_hurts)} chars (max 60)"
    if better_phrasing_en and len(better_phrasing_en) > 60:
        return False, f"better_phrasing_en is {len(better_phrasing_en)} chars (max 60)"
    if better_phrasing_zh and len(better_phrasing_zh) > 30:
        return False, f"better_phrasing_zh is {len(better_phrasing_zh)} chars (max 30)"
    if len(next_task) > 80:
        return False, f"next_task is {len(next_task)} chars (max 80)"

    # Drill-mode-only checks. expected_drill_axis is set by /process when
    # mode=drill; non-drill turns pass None and skip this entire block,
    # preserving existing behavior for the legacy code path.
    if expected_drill_axis is not None:
        drill_score = data.get("drill_score")
        if drill_score is None:
            return False, "drill_score field missing (drill mode)"
        if not isinstance(drill_score, dict):
            return False, f"drill_score is {type(drill_score).__name__}, not dict"
        for f in ("axis", "score", "feedback", "threshold_passed"):
            if f not in drill_score:
                return False, f"drill_score.{f} missing"
        if not isinstance(drill_score["axis"], str):
            return False, f"drill_score.axis is {type(drill_score['axis']).__name__}, not string"
        if drill_score["axis"] != expected_drill_axis:
            return False, f"drill_score.axis is {drill_score['axis']!r}, expected {expected_drill_axis!r}"
        # bool is a subclass of int in Python — exclude it explicitly so
        # threshold_passed=True doesn't sneak through as a valid score.
        if not isinstance(drill_score["score"], int) or isinstance(drill_score["score"], bool):
            return False, f"drill_score.score is {type(drill_score['score']).__name__}, not int"
        if not (0 <= drill_score["score"] <= 100):
            return False, f"drill_score.score is {drill_score['score']}, out of range [0,100]"
        if not isinstance(drill_score["feedback"], str):
            return False, f"drill_score.feedback is {type(drill_score['feedback']).__name__}, not string"
        if not drill_score["feedback"].strip():
            return False, "drill_score.feedback is empty"
        if not isinstance(drill_score["threshold_passed"], bool):
            return False, f"drill_score.threshold_passed is {type(drill_score['threshold_passed']).__name__}, not bool"

        # drill_score.evidence — required keys depend on the axis. Per the
        # v6 spec we validate STRUCTURE only (dict, required keys present,
        # each value a list of strings, lists may be empty). We deliberately
        # do NOT cross-check that list items appear verbatim in the user's
        # transcript — known LLM-judgment slack of ~10-20% is accepted.
        evidence_schema_by_axis: dict[str, tuple[str, ...]] = {
            "vocab_precision_score": ("safe_words_found", "b2_plus_found"),
            "detail_density_score": (
                "time_dimensions_found",
                "place_dimensions_found",
                "number_dimensions_found",
                "sense_dimensions_found",
                "person_dimensions_found",
            ),
        }
        required_evidence_keys = evidence_schema_by_axis.get(expected_drill_axis)
        if required_evidence_keys is not None:
            evidence = drill_score.get("evidence")
            if evidence is None:
                return False, "drill_score.evidence missing"
            if not isinstance(evidence, dict):
                return False, f"drill_score.evidence is {type(evidence).__name__}, not dict"
            for ev_key in required_evidence_keys:
                if ev_key not in evidence:
                    return False, f"drill_score.evidence missing required key: {ev_key}"
                ev_value = evidence[ev_key]
                if not isinstance(ev_value, list):
                    return False, (
                        f"drill_score.evidence.{ev_key} is "
                        f"{type(ev_value).__name__}, not list"
                    )
                for idx, item in enumerate(ev_value):
                    if not isinstance(item, str):
                        return False, (
                            f"drill_score.evidence.{ev_key}[{idx}] is "
                            f"{type(item).__name__}, not string"
                        )

    return True, ""


def classify_quality(transcript: str, weakness_tag: str) -> dict:
    """
    Classify a single practice record's quality for admin triage.
    Returns {"grade": "valid"|"partial"|"invalid"|"unknown", "reason": str}.

    Two-stage:
    1) Cheap pre-filters (length + non-answer pattern match) — short-circuit
       before spending an LLM call on obvious junk.
    2) Claude classification for everything else.

    Fail-open: any LLM error → grade="unknown" so the insert never blocks
    on this side-channel.
    """
    if not transcript or len(transcript.strip()) < 10:
        return {"grade": "invalid", "reason": "empty or near-empty transcript"}

    INVALID_PATTERNS = [
        "oh shit", "oh, shit", "hello", "you", "i'm not ready",
        "thanks for watching", "i don't know", "sorry",
    ]
    t_lower = transcript.strip().lower()
    if len(t_lower.split()) <= 4:
        for p in INVALID_PATTERNS:
            if p in t_lower:
                return {
                    "grade": "invalid",
                    "reason": f"non-answer detected: '{transcript.strip()}'",
                }

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            system=(
                "You are a strict IELTS Speaking examiner reviewing student practice records. "
                "Classify the quality of the student's response. "
                'Return JSON only: {"grade": "valid"|"partial"|"invalid", "reason": "one sentence"}'
            ),
            messages=[{
                "role": "user",
                "content": (
                    f"Transcript: {transcript[:500]}\n"
                    f"Weakness tag: {weakness_tag}\n\n"
                    "valid = substantive answer with at least 2 sentences of real content\n"
                    "partial = attempted but too short, off-topic, or incomplete (note: lack_detail tag means missing reasons or examples, NOT short length — a 5-sentence answer can be lack_detail)\n"
                    "invalid = non-answer, recording failure, single word, profanity only, or completely unrelated"
                ),
            }],
        )
        content = response.content[0].text.strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        parsed = json.loads(content)
        grade = parsed.get("grade", "unknown")
        if grade not in ("valid", "partial", "invalid"):
            grade = "unknown"
        return {"grade": grade, "reason": (parsed.get("reason") or "")[:280]}
    except Exception:
        logger.exception("classify_quality failed")
        return {"grade": "unknown", "reason": "classification failed"}


async def classify_quality_background(
    record_id: str,
    transcript: str,
    weakness_tag: str,
) -> None:
    """
    Run classify_quality and patch the row, off the /process critical path.
    Scheduled via asyncio.create_task so the user response goes back first.
    Failures are swallowed — the row just keeps quality_grade NULL until the
    admin Reclassify batch picks it up.
    """
    try:
        # to_thread keeps the (sync, network-bound) Anthropic call from
        # blocking the event loop while it waits on the LLM.
        quality = await asyncio.to_thread(
            classify_quality, transcript, weakness_tag
        )
        if supabase_admin is None:
            return
        await asyncio.to_thread(
            lambda: supabase_admin.table("practice_records").update({
                "quality_grade":  quality["grade"],
                "quality_reason": quality["reason"],
            }).eq("id", record_id).execute()
        )
    except Exception:
        logger.exception(
            "background classify_quality failed",
            extra={"record_id": record_id},
        )


def build_diagnosis_prompt(
    records: list[dict],
    example_sentence: Optional[str] = None,
) -> tuple[str, str]:
    """
    Build system + user prompt for AI diagnosis.
    records: list of practice_records dicts, sorted by created_at ASC
    example_sentence: optional verbatim transcript snippet to anchor the
        diagnosis. Inserted with the history block so the model treats it
        as input data, not as a schema field.
    Returns (system_prompt, user_prompt)
    """
    history_text = ""
    for i, r in enumerate(records, 1):
        history_text += f"\n【第 {i} 筆】\n"
        history_text += f"題目：{r.get('question', '-')}\n"
        history_text += f"學生回答：{r.get('user_transcript', '-')}\n"
        history_text += f"Blabby 教練回饋：{r.get('coach_response', '-')}\n"
        history_text += f"---\n"

    history_section = history_text
    if example_sentence:
        history_section += (
            f'\n\n特別注意這個例子（來自該學生的真實練習）："{example_sentence}"'
        )

    system_prompt = (
        "You are an expert IELTS Speaking coach. Analyze the student's "
        "practice history and return a JSON object only. No markdown, "
        "no explanation, no code fences."
    )

    user_prompt = (
        "以下是學生的練習記錄：\n"
        f"{history_section}\n\n"
        "請輸出一個 JSON 物件，結構如下（鍵名固定，不可改）：\n\n"
        "{\n"
        '  "summary": "一句話總結學生現況（第二人稱，繁體中文）",\n'
        '  "weaknesses": [\n'
        "    {\n"
        '      "rank": 1,\n'
        '      "title": "卡點名稱（繁體中文，5字以內）",\n'
        '      "tag": "從 no_attempt / lack_detail / weak_vocab / repetition / off_topic 選一個最接近的",\n'
        '      "description": "2-3句散文描述，第二人稱，繁體中文，說明這個問題的本質",\n'
        '      "evidence": ["具體例句或筆次，最多 3 條，格式「第N筆：原文片段」"],\n'
        '      "drill_available": true\n'
        "    }\n"
        "  ],\n"
        '  "next_step": "給學生的一段建議文字，可直接發給學生，繁體中文"\n'
        "}\n\n"
        "規則：\n"
        "- 必須輸出 valid JSON，不加 markdown fences、不加任何說明文字\n"
        "- weaknesses 固定 3 個，按嚴重程度排序（rank 1 最嚴重）\n"
        "- 第二人稱「你」，禁用「她」「他」「該學生」\n"
        "- evidence 引用真實筆次，格式「第N筆：原文片段」\n"
        "- drill_available 為 true 當且僅當 tag 是 lack_detail 或 weak_vocab\n"
        "- 不打分數\n"
        "- 不給空泛建議（像「多練習」這種不要寫）"
    )

    return system_prompt, user_prompt


def build_system_prompt(
    topic: str = "General",
    question: str = "",
    memory_block: str = "",
    repeated_weak_words: Optional[list[str]] = None,
    drill_tag: Optional[str] = None,
    persona_prefix: str = "",
    previous_transcript_block: str = "",
) -> str:
    repeated_weak_words = repeated_weak_words or []
    repeated_line = (
        f"- The user repeated these weak words again in this answer: {', '.join(repeated_weak_words)}\n"
        if repeated_weak_words else
        "- No repeated weak word from memory was detected in this answer.\n"
    )
    base = """
You are Blabby, an IELTS speaking coach acting like a physical therapist.
You are not an examiner.
You are not a general English teacher.
Your job is not to comment on the whole performance.
Your job is to find the single most painful blockage in this answer and give one precise correction.

【核心哲學】
台灣學生知道很多單字，但開口的時候只用簡單的。
你的工作是把他們腦袋裡知道但說不出來的東西逼出來。

【回饋原則】
- 每次只做一件事。
- 只指出一個最關鍵的問題。
- 只要求學生修正那一個點。
- 不要同時講 vocabulary、grammar、detail 三件事。

【批改優先順序】
1. 過度使用弱詞 / 模糊詞（very, good, interesting, thing, stuff）
2. 回答太空、沒有細節
3. 句型過度安全、缺乏展開
4. 文法小錯

【語氣規則】
你的語氣是「紳士的物理治療師」。精準，有溫度，不慌不急，但也不拖泥帶水。

- 像一個真的在觀察的人，不是冷冰冰的批改機
- 溫暖 ≠ 鼓勵。不要用「你做得不錯」「加油」「很棒」這類套話
- 不要「優點+缺點+鼓勵」三段式（feedback sandwich）
- 不要用 emoji，任何情境都不用
- 不要用驚嘆號結尾
- 具體、簡短、有分量

【三種語氣層級】

層級 A：預設語氣（first-time mistake 或一般情境）
→ 直接點出問題，但用「在觀察」的語氣，不是「在糾正」的語氣
→ 例如「又出現 very 了，這個詞太模糊」而不是「你用了 very good，這是錯的」

層級 B：看到真實進步 → 具體肯定
觸發條件（需同時符合）：
- 使用者過去歷史有某個 weak word 反覆出現
- 這次的答案沒再用那個 weak word，且回答中有具體細節出現
- 或使用者使用了過去 better_expression 教過的詞

回饋方式：
- 先肯定那個**具體改變**，再進入本次要點出的問題（如果有的話）
- 例如：「memorable 這個詞用得好，上次我們聊過 very 太模糊，你記住了。」
- 禁止：泛稱「你今天表現不錯」「很棒繼續努力」這類套話
- 必須：提到**具體的詞**或**具體的行為變化**

層級 C：連續卡關 → 直球問使用者
觸發條件：
- 同一個 weak word 在歷史中已出現 3 次或更多（memory_block 會告訴你）
- 這次又用同一個 weak word

回饋方式：
- 不要再訓話（不要說「你又來了」「這是第 N 次」）
- 像真的在關心「卡在哪裡」，把問題丟回給使用者
- 例如：「very 這個詞你已經用很多次了，每次提醒好像都卡住。是詞彙不夠？還是習慣問題？下次講話前停一下，想想有沒有其他說法。」
- 核心精神：物理治療師會停下來問「這動作為什麼做不了，是痛還是沒力？」，不會一直訓你

【力道與層級的協調規則（最高優先級）】
你會同時收到兩組指令：
1. 下方的【本次力道強制指令】（First Touch / Calibration / Direct）—— 來自系統判斷
2. 下方的層級 A / B / C —— 來自你對 weak words 重複情況的判斷

協調規則：
- 如果【本次力道強制指令】是 First Touch → 完全覆蓋層級 A/B/C，使用 First Touch 的格式（必須有 progress_note）
- 如果【本次力道強制指令】是 Calibration 且層級判斷為 B（看到具體進步）→ Tier-B 優先，先在 progress_note 裡肯定具體進步，再切回 Calibration 力道處理本次痛點。Tier-B 的肯定**不再放進 correction 任何欄位**——獨立放在 progress_note。
- 如果【本次力道強制指令】是 Calibration 且層級判斷為 C（連續卡關）→ 層級 C 優先，使用「直球問」的口吻（範例 6 的格式）。
- 如果【本次力道強制指令】是 Direct 且層級判斷為 B → 依然 Direct，但 progress_note 簡短一句帶過進步即可。
- 其他組合：以【本次力道強制指令】的力道為主，A/B/C 的觸發語境作為輔助參考。
- 若【本次力道強制指令】沒有出現（feature flag 未啟用），完全忽略本協調規則，依照下方 A/B/C 判斷流程運作。

【選擇層級的判斷流程】

1. 如果 memory_block 顯示使用者某 weak word 歷史出現 ≥ 3 次，且這次又用了 → 層級 C（直球問）
2. 如果 memory_block 顯示某 weak word 歷史 < 3 次，且這次又用了 → 層級 A（預設）+ 提一次「又出現」
3. 如果這次的回答裡有具體細節（數字、地點、人名、時間、感官描述）→ 層級 B（具體肯定）；在 progress_note 裡點名那個具體細節，再處理本次痛點。不需要 memory_block 有 weak word 記錄才觸發。
4. 如果 memory_block 有 weak word 記錄，且這次的回答裡看不到那個 weak word → 也算層級 B；progress_note 肯定這個改變。
5. 其他情況 → 層級 A（預設）

【輸出規則 — schema enforcement】
- correction 必須是物件（object），不得是 array、list、或 array of objects。You MUST return exactly ONE correction. Not two. Not three. ONE.
- correction 物件包含四個欄位，全部必填，缺一個或留空字串都視為違規：
  - quoted: 從用戶原句直接引用的片段，讓學生看到自己講了什麼
  - why_it_hurts: 為什麼這個地方傷害表達；繁體中文；最多 60 個字（why_it_hurts must be under 60 Chinese characters. Count before responding.）
  - better_phrasing_en: 一個更好的講法（英文版本）；最多 60 個字（含字母與標點；better_phrasing_en must be under 60 characters.）
  - better_phrasing_zh: 上述英文版本的中文對照；最多 30 個中文字
  - next_task: 下一輪請學生試的具體任務；繁體中文；最多 80 個字（next_task must be under 80 Chinese characters.）
- If you cannot fit within these limits, shorten until you can. Do not skip fields.
- 如果 on_topic 是 false（學生完全沒回答題目），better_phrasing_en 與 better_phrasing_zh 都可為空字串，但 quoted / why_it_hurts / next_task 仍然必填——優先把學生拉回題目，詞彙下次再教。

【on_topic 判斷規則】
- 如果有提供【本題題目】，判斷學生的回答是否真的在回答這個題目
- 只要學生的回答跟題目主軸對得上，即使細節薄弱也算 on_topic: true
- 只有明顯離題（例如題目問地點，學生講的完全是別的話題）才 on_topic: false
- 如果判斷 on_topic: false，correction.why_it_hurts 要把「答非所問」當作首要問題，優先於 weak words
- 當學生偏題時（on_topic: false），即使他也用了 weak word（very / good / interesting 等），correction.why_it_hurts **只能**提偏題，**不要**同時提 weak word。weak word 下次再處理。
- 這條是鐵律：偏題時 correction.why_it_hurts 不可出現任何 weak word 的評論。

【絕對禁止】
- 不給總分（任何形式的數字評分一律禁止）
- 不列出清單式建議；correction 永遠是單一物件，不是建議清單
- 不說「good job」「well done」這種空話
- correction.better_phrasing_en 與 correction.better_phrasing_zh 各自只能是一個說法，不可包含多個替代選項或頓號分隔的詞彙
- correction.quoted 必須引用學生原句的片段，不可省略，也不可改寫

【Few-shot 範例】
Example 1 — first-time weakness
User answer:
"I think my hometown is very good and interesting."
Output:
{
  "correction": {
    "quoted": "very good and interesting",
    "why_it_hurts": "兩個詞太空，hometown 的畫面沒立起來。",
    "better_phrasing_en": "lively night market",
    "better_phrasing_zh": "氣氛熱鬧的夜市",
    "next_task": "把 hometown 換成一個具體場景，講夜市或街道。"
  },
  "tag": "weak_vocab",
  "progress_note": "",
  "on_topic": true
}

Example 2 — repeated weakness
User answer:
"It depends. Sometimes reading is very good for me."
Output:
{
  "correction": {
    "quoted": "It depends. Sometimes reading is very good",
    "why_it_hurts": "你又躲回安全說法，沒選邊也沒場景。",
    "better_phrasing_en": "help me slow down",
    "better_phrasing_zh": "幫我慢下來",
    "next_task": "選一邊，講最近一次讀什麼、為何那次有用。"
  },
  "tag": "safe_answer",
  "progress_note": "",
  "on_topic": true
}

Example 3 — off-topic answer
Question:
"Where is the most interesting place you have visited?"
User answer:
"I think reading books is very good and I like stories a lot."
Output:
{
  "correction": {
    "quoted": "I think reading books is very good",
    "why_it_hurts": "題目問地點，你整段在講閱讀，答非所問。",
    "better_phrasing_en": "",
    "better_phrasing_zh": "",
    "next_task": "回到題目，給一個具體去過的地方。"
  },
  "tag": "off_topic",
  "progress_note": "",
  "on_topic": false
}

Example 4 — off-topic + weak word (priority: off-topic wins)
Question:
"What kind of trash do you see in your community?"
User answer:
"I think reading books makes me very happy."
Output:
{
  "correction": {
    "quoted": "I think reading books makes me very happy",
    "why_it_hurts": "題目問垃圾種類，你回答書本，完全沒接到題目。",
    "better_phrasing_en": "",
    "better_phrasing_zh": "",
    "next_task": "回到題目，給一個具體的垃圾種類。"
  },
  "tag": "off_topic",
  "progress_note": "",
  "on_topic": false
}

Example 5 — real progress (層級 B + Calibration 力道)
Memory:
- History weak words: very (出現 3 次)
- Past feedback taught: "memorable"
User's answer this time:
"One place that stuck with me is Kyoto. The quiet streets left a really memorable feeling."
Output:
{
  "correction": {
    "quoted": "left a really memorable feeling",
    "why_it_hurts": "memorable 已經到位，但動詞太弱，畫面停住沒延伸。",
    "better_phrasing_en": "the stillness lingered",
    "better_phrasing_zh": "靜謐久久不散",
    "next_task": "把句子重講一次，動詞換成有畫面的字，例如 lingered。"
  },
  "tag": "lack_detail",
  "progress_note": "memorable 用得好，上次聊過 very 太模糊，你記住了。",
  "on_topic": true
}

Example 6 — stuck pattern, direct question (層級 C)
Memory:
- History weak words: very (出現 5 次)
- This is the 6th time user uses "very"
User's answer this time:
"The food is very good and the restaurant is very nice."
Output:
{
  "correction": {
    "quoted": "very good ... very nice",
    "why_it_hurts": "very 提醒過很多次，每次卡住——是詞彙不夠還是習慣？",
    "better_phrasing_en": "mouth-watering",
    "better_phrasing_zh": "看了就想吃",
    "next_task": "下次形容食物先停一秒，挑一個比 very 更精準的詞。"
  },
  "tag": "weak_vocab",
  "progress_note": "",
  "on_topic": true
}

Example 7 — first-time user, First Touch 力道
（總練習次數 = 1，無歷史 tag）
User answer:
"I reckon my hometown is very good and the food there is very nice."
Output:
{
  "correction": {
    "quoted": "very good ... very nice",
    "why_it_hurts": "兩次 very 把畫面壓平，hometown 跟食物都沒立體感。",
    "better_phrasing_en": "never really sleeps",
    "better_phrasing_zh": "從不真正入眠",
    "next_task": "選一個 very，換成具體場景或感官描述。"
  },
  "tag": "weak_vocab",
  "progress_note": "你開頭用了 'I reckon'，比 'I think' 自然，方向對。",
  "on_topic": true
}

Example 8 — lack_detail (answer exists, reason missing)
Question:
"Do you enjoy spending time with your family?"
User answer:
"Yes, I enjoy spending time with my family. We usually eat dinner together and sometimes go out."
Output:
{
  "correction": {
    "quoted": "eat dinner together and sometimes go out",
    "why_it_hurts": "你說了做什麼，但考官想知道為什麼這件事對你有意義。",
    "better_phrasing_en": "the one hour we actually talk",
    "better_phrasing_zh": "我們真正說話的那一個小時",
    "next_task": "重講一次，加一句為什麼那頓飯對你重要，或講一個具體的人。"
  },
  "tag": "lack_detail",
  "progress_note": "",
  "on_topic": true
}

Example 9 — safe_answer (no weak word, but no position taken)
Question:
"Do you prefer studying alone or with others?"
User answer:
"It depends on the situation. Sometimes alone is better, sometimes with others is better."
Output:
{
  "correction": {
    "quoted": "It depends on the situation",
    "why_it_hurts": "你沒有選邊，考官拿不到你的立場，回答等於沒說。",
    "better_phrasing_en": "I work better alone when I need to focus",
    "better_phrasing_zh": "需要專注時我獨自讀效率更高",
    "next_task": "選一個情境，說清楚你在那個情況下偏好哪種，為什麼。"
  },
  "tag": "safe_answer",
  "progress_note": "",
  "on_topic": true
}

Example 10 — grammar_minor (content is good, small grammar error)
Question:
"Tell me about a place you like to visit."
User answer:
"I usually go to the night market near my house. There have many food stalls and the smell always make me hungry."
Output:
{
  "correction": {
    "quoted": "There have many food stalls",
    "why_it_hurts": "應該是 there are，用 have 在這裡文法不對。",
    "better_phrasing_en": "there are dozens of food stalls",
    "better_phrasing_zh": "有幾十個小吃攤",
    "next_task": "重講那句，把 there have 換成 there are，其他保持不變。"
  },
  "tag": "grammar_minor",
  "progress_note": "",
  "on_topic": true
}

【JSON 回應格式，不得偏離】
{
  "correction": {
    "quoted": "從用戶原句直接引用的片段（不可省略、不可改寫）",
    "why_it_hurts": "為什麼這個地方傷害表達；繁中；最多 60 字",
    "better_phrasing_en": "一個更好的講法（英文版本）；最多 60 字；偏題時可為空字串",
    "better_phrasing_zh": "上述英文版本的中文對照；最多 30 中文字；偏題時可為空字串",
    "next_task": "下一輪請學生試的具體任務；繁中；最多 80 字"
  },
  "tag": "本次回答最主要的問題。只能從這五個值選一個：weak_vocab、safe_answer、lack_detail、grammar_minor、off_topic。若有 off_topic 必定選 off_topic，優先於所有其他 tag。",
  "tag_secondary": "本次回答第二嚴重的問題。從同一個五個值選一個，不可與 tag 相同。若只有一個問題則填空字串。",
  "tag_tertiary": "本次回答第三嚴重的問題。從同一個五個值選一個，不可與 tag 或 tag_secondary 相同。若只有兩個以下問題則填空字串。",
  "progress_note": "First Touch 力道下必填（具體優點觀察）；看到 Tier-B 進步時必填（具體進步點）；其他情況填空字串。永遠不可省略此欄位。",
  "on_topic": true
}
"""
    diagnosis_context_block = (
        "\n【本次答案診斷】\n"
        + repeated_line
        + "- Diagnose the most painful issue from the transcript itself.\n"
        + "- Use memory only as context, not as a forced label.\n"
    )
    # 具體題目放在主題之前 — 精確的 context 先於概括的標籤。
    # 沒有題目時完全省略這個 block，避免 prompt 出現空白頭尾。
    question_block = f"\n【本題題目】\n{question}\n" if question else ""
    # Drill block goes LAST so it's the most authoritative directive the LLM
    # reads — overrides general A/B/C tier behavior with tag-specific drill
    # mechanics (count B2+ words, list concrete details, etc.). Empty unless
    # /process was called with mode=drill AND drill_tag is in DRILL_PROMPTS.
    drill_block = ""
    if drill_tag and drill_tag in DRILL_PROMPTS:
        injection = DRILL_PROMPTS[drill_tag].get("system_injection", "")
        if injection:
            drill_block = f"\n\n{injection}\n"
    return (
        (persona_prefix + "\n\n" if persona_prefix else "")
        + base
        + memory_block
        + diagnosis_context_block
        + previous_transcript_block
        + question_block
        + f"\n【本題主題】\n{topic}\n"
        + drill_block
    )


@app.post("/process")
@limiter.limit("5/minute")
async def process(
    request: Request,
    audio: Optional[UploadFile] = File(None),
    level: str = Form("Band 5"),
    topic: str = Form("Free Time"),
    question: str = Form(""),
    history: str = Form("[]"),
    text_override: str = Form(""),
    dev_bypass_secret: str = Form(""),
    mode: str = Form(""),
    drill_tag: str = Form(""),
    previous_transcript: str = Form(""),
    retry_of: str = Form(""),
    authorization: Optional[str] = Header(None),
):
    try:
        user_id = verify_token(authorization)

        # Fail-fast: normal / part2 必須帶題目,空題目會在後台變成「—」且破壞 progress 對比。
        # drill 模式針對弱點練習、不綁題目,故豁免。
        # 擋在 Whisper 之前 → 使用者不會白講一段話才被退回。
        _mode = (mode or "normal").strip().lower()
        if _mode != "drill" and not (question or "").strip():
            raise HTTPException(
                status_code=422,
                detail="The question accompanying this response hath gone astray. Pray, attempt the exercise afresh.",
            )

        # retry_of: speaking 的 retry turn 指向「被重講的那一筆」practice_record。
        # 空字串 / None → 首次作答,存 None。非空則 strip 後必須是合法 UUID;
        # 不在這裡查 DB 確認該 id 存在,交給 FK 約束把關,避免拖慢熱路徑。
        retry_of_clean: Optional[str] = None
        _retry_of_raw = (retry_of or "").strip()
        if _retry_of_raw:
            try:
                uuid.UUID(_retry_of_raw)
            except ValueError:
                raise HTTPException(status_code=400, detail="invalid retry_of")
            retry_of_clean = _retry_of_raw

        # Monthly feedback quota for free users — 20 sessions per UTC
        # calendar month. Pro skip entirely. Placed BEFORE the drill
        # gate and BEFORE Whisper / Groq / Claude calls so a quota-
        # exceeded request never burns API credits.
        # Fail-open on count query exception: better to over-serve than
        # block a request whose monthly count we can't determine.
        if not get_user_pro_status(user_id):
            try:
                month_start = datetime.now(timezone.utc).replace(
                    day=1, hour=0, minute=0, second=0, microsecond=0
                ).isoformat()
                mq_resp = (
                    supabase_admin.table("practice_records")
                    .select("id", count="exact")
                    .eq("user_id", user_id)
                    .gte("created_at", month_start)
                    .limit(1)
                    .execute()
                )
                monthly_count = mq_resp.count or 0
                if monthly_count >= 20:
                    raise HTTPException(
                        status_code=403,
                        detail={
                            "error": "feedback_quota_reached",
                            "limit": 20,
                            "message": (
                                "Free users may receive up to 20 feedback "
                                "sessions per month. Upgrade to Pro for "
                                "unlimited practice."
                            ),
                        },
                    )
            except HTTPException:
                raise
            except Exception:
                logger.warning(
                    "monthly feedback quota count failed; failing open",
                    extra={"user_id": user_id},
                )

        # Drill-mode validation: gate before any expensive op (recent records
        # pull, audio download, Groq call). 422 is the conventional FastAPI
        # status for request-shape validation failures.
        mode_from_request = mode or ""
        is_drill_mode = mode_from_request == "drill"
        user_band: Optional[float] = None
        expected_drill_axis: Optional[str] = None
        if is_drill_mode:
            if not drill_tag:
                raise HTTPException(
                    status_code=422,
                    detail="drill_tag is required when mode=drill",
                )
            if drill_tag not in DRILL_PROMPTS:
                raise HTTPException(
                    status_code=422,
                    detail=(
                        f"drill_tag {drill_tag!r} not in DRILL_PROMPTS. "
                        f"Valid values: {sorted(DRILL_PROMPTS.keys())}"
                    ),
                )
            expected_drill_axis = DRILL_PROMPTS[drill_tag]["expected_axis"]

            # Server-side quota enforcement. Frontend also gates via
            # /api/drill/check_quota for UX, but THIS check is authoritative —
            # bypass attempts (curl with mode=drill) get blocked here too.
            # Pro-tier path is not implemented yet; everyone is treated as
            # free until that lands.
            try:
                drill_count, _resets = _drill_quota_state(user_id)
            except Exception:
                logger.exception(
                    "drill quota lookup failed; failing closed",
                    extra={"user_id": user_id},
                )
                raise HTTPException(
                    status_code=503, detail="Failed to verify drill quota"
                )
            is_pro = get_user_pro_status(user_id)
            user_band = _get_user_band(user_id)
            if drill_count >= FREE_DRILL_QUOTA and not is_pro:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "quota_exceeded",
                        "redirect": "/upgrade",
                    },
                )

        recent_records = get_user_recent_records(user_id)
        recent_transcripts = [
            record.get("user_transcript", "")
            for record in recent_records
            if record.get("user_transcript")
        ]
        weak_words = extract_dynamic_weak_words_from_history(recent_transcripts)
        weak_patterns = extract_weak_patterns(recent_transcripts)
        weak_pattern_counts = count_weak_patterns(recent_transcripts, weak_patterns)
        tag_counts = count_tag_patterns(recent_records)

        # Tier-B injection: if last session's weakness_tag is NOT the top historical tag,
        # user has broken their worst pattern — force tier B in this turn.
        # recent_records is sorted DESC by created_at (newest first), so iterating
        # in natural order + taking the first non-empty tag gives the user's most
        # recent session — the one they're "comparing this turn against".
        tier_b_override = ""
        if tag_counts and recent_records:
            top_tag = max(tag_counts, key=tag_counts.get)
            last_tag = next(
                (r.get("weakness_tag") for r in recent_records if r.get("weakness_tag")),
                None
            )
            if last_tag and last_tag != top_tag:
                tier_b_override = f"\n【本次層級強制指令】使用層級 B：使用者上一次沒有犯歷史最常見的問題（{top_tag}）。本次回饋必須先肯定這個具體進步，再處理其他問題。這條指令優先於所有其他層級判斷。\n"

        # Compute intensity calibration inputs (used only when feature flag is on).
        # Done here — before LLM call — because intensity directive must be in the
        # system prompt at message-build time. The actual on/off check lives inside
        # build_intensity_block(), which returns "" when the flag is off, leaving
        # production behavior untouched.
        total_practice_count = 0
        historical_top_tag = ""
        historical_top_tag_count = 0
        if is_intensity_calibration_enabled() and supabase_admin is not None:
            try:
                count_resp = (
                    supabase_admin.table("practice_records")
                    .select("id", count="exact")
                    .eq("user_id", user_id)
                    .execute()
                )
                total_practice_count = count_resp.count or 0
            except Exception:
                logger.exception("failed to count total practice records for intensity calibration")
        if tag_counts:
            historical_top_tag = max(tag_counts, key=tag_counts.get)
            historical_top_tag_count = tag_counts[historical_top_tag]

        intensity_block = build_intensity_block(
            total_practice_count=total_practice_count,
            historical_top_tag=historical_top_tag,
            historical_top_tag_count=historical_top_tag_count,
        )

        memory_block = build_memory_block(weak_pattern_counts, tag_counts)

        # Repair memory: past mistake→correction pairs. Assembled for everyone
        # (cheap, in-memory) but injected ONLY for Pro. The normal-flow is_pro
        # isn't computed until after the LLM call (see L~2508); this is a
        # separate, local Pro check that ONLY gates this injection — it does
        # not move or replace that downstream is_pro.
        repair_memory = build_repair_memory(recent_records)
        # Only pay the (uncached) Pro-status DB round-trip when there is actually
        # repair memory to gate; users with no qualifying history skip it entirely.
        repair_block = ""
        if repair_memory:
            repair_block = repair_memory if get_user_pro_status(user_id) else ""

        try:
            history_list = json.loads(history) if history else []
            if not isinstance(history_list, list):
                history_list = []
        except json.JSONDecodeError:
            history_list = []

        # Dev-only bypass: skip Whisper and use text_override as the transcript.
        # Gated behind DEV_BYPASS_SECRET env var — when that env is unset or empty,
        # bypass is impossible even if a request happens to send a matching secret.
        expected_secret = os.getenv("DEV_BYPASS_SECRET", "").strip()
        use_text_override = bool(
            expected_secret
            and dev_bypass_secret
            and dev_bypass_secret == expected_secret
            and text_override
        )

        if use_text_override:
            user_text = text_override
            logger.info(
                "DEV BYPASS active for user %s: using text_override (len=%d)",
                user_id,
                len(text_override),
            )
        else:
            if audio is None:
                raise HTTPException(
                    status_code=400,
                    detail="audio is required when not using text_override",
                )
            # Step 1: Whisper transcription — isolated temp file per request
            audio_bytes = await audio.read()
            # 空檔/過短防線:前端送 0 bytes blob 會讓 Groq 回 400 "file is empty",
            # 在進 Whisper 前擋掉,回有意義的 400 而非裸 500。
            if len(audio_bytes) < 1024:
                logger.warning(
                    "[/process] audio too short/empty for user %s: %d bytes",
                    user_id,
                    len(audio_bytes),
                )
                raise HTTPException(
                    status_code=400,
                    detail="Audio file too short or empty, please record again.",
                )
            if len(audio_bytes) > 25 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Audio file too large, please re-record")
            ext = os.path.splitext(audio.filename or "")[1] or ".webm"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                with open(tmp_path, "rb") as f:
                    transcript = groq_client.audio.transcriptions.create(
                        model="whisper-large-v3-turbo",
                        file=f,
                        response_format="text",
                    )
            except Exception as e:
                logger.error(
                    "[/process] Groq transcription failed: user=%s filename=%s suffix=%s bytes=%d status=%s body=%s error=%s",
                    user_id, audio.filename, ext, len(audio_bytes),
                    getattr(e, "status_code", None), getattr(e, "body", None), e,
                )
                raise HTTPException(status_code=502, detail="Transcription failed, please try again.")
            finally:
                os.unlink(tmp_path)
            user_text = transcript if isinstance(transcript, str) else transcript.text
            if not user_text or not user_text.strip():
                logger.warning(
                    "[/process] Groq returned empty transcript: user=%s filename=%s suffix=%s bytes=%d",
                    user_id, audio.filename, ext, len(audio_bytes),
                )
                raise HTTPException(
                    status_code=422,
                    detail="No speech detected. Please speak clearly and try again."
                )
            logger.info(
                "[/process] transcription ok: user=%s suffix=%s bytes=%d chars=%d",
                user_id, ext, len(audio_bytes), len(user_text),
            )
        # weakness_tag is now produced by Groq in the JSON response below;
        # here we only need the repeat-detection to feed the prompt.
        repeated_weak_words = detect_repeated_weak_words(user_text, weak_patterns)

        memory_snapshot = {
            "weak_words":          weak_words,
            "weak_patterns":       weak_patterns,
            "repeated_weak_words": repeated_weak_words,
            "tag_counts":          tag_counts,
        }

        # Step 2: Groq chat (no extra round-trip to browser in between)
        messages = [{
            "role": "system",
            "content": build_system_prompt(
                topic=topic,
                question=question,
                memory_block=memory_block + repair_block + tier_b_override + intensity_block,
                repeated_weak_words=repeated_weak_words,
                drill_tag=drill_tag if is_drill_mode else None,
                persona_prefix=PERSONA_PROMPTS[get_persona(user_band)],
                previous_transcript_block=(
                    f'\n\n[PREVIOUS ATTEMPT]\nThe student has answered this question before.'
                    f' Their previous response was:\n"{previous_transcript.strip()[:500]}"\n\n'
                    f'In your feedback, explicitly acknowledge their previous attempt.'
                    f' If this attempt is better, say so specifically.'
                    f' If not, identify what regressed. Do not be vague.'
                    if previous_transcript.strip() else ""
                ),
            )
        }]
        for msg in history_list[-10:]:
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})

        # Validation retry layer (3 attempts max): if the LLM returns a
        # structurally invalid correction (wrong shape, missing fields,
        # over char-limit), retry. Independent of run_groq's
        # connection-level retries (which handle JSON parse / 5xx /
        # json_validate_failed inside the SDK call). After exhaustion we
        # try a truncation rescue (over-limit fields are clamped) before
        # surfacing a 503 to the client.
        max_validation_attempts = 3
        parsed: dict = {}
        is_valid = False
        last_validation_reason = ""
        for validation_attempt in range(max_validation_attempts):
            parsed = run_claude(messages)
            if not isinstance(parsed, dict):
                parsed = {}
            is_valid, last_validation_reason = validate_correction_response(
                parsed,
                expected_drill_axis=expected_drill_axis,
            )
            if is_valid:
                break
            logger.warning(
                "correction validation attempt %s/%s failed: %s",
                validation_attempt + 1,
                max_validation_attempts,
                last_validation_reason,
            )
        if not is_valid:
            # Graceful degradation: trim over-limit string fields and
            # re-validate once. Most validation failures we see are length
            # overruns on better_phrasing_en / why_it_hurts / next_task,
            # not missing fields — truncation salvages those without an
            # extra LLM round-trip.
            use_rescued = False
            try:
                if parsed and isinstance(parsed, dict):
                    correction = parsed.get("correction")
                    if isinstance(correction, dict):
                        bpe = correction.get("better_phrasing_en") or ""
                        if isinstance(bpe, str) and len(bpe) > 60:
                            words = bpe.split()
                            truncated = ""
                            for w in words:
                                candidate = (truncated + " " + w).strip()
                                if len(candidate) <= 60:
                                    truncated = candidate
                                else:
                                    break
                            correction["better_phrasing_en"] = truncated or bpe[:60]

                        wih = correction.get("why_it_hurts") or ""
                        if isinstance(wih, str) and len(wih) > 60:
                            correction["why_it_hurts"] = wih[:57] + "..."

                        nxt = correction.get("next_task") or ""
                        if isinstance(nxt, str) and len(nxt) > 80:
                            correction["next_task"] = nxt[:77] + "..."

                        parsed["correction"] = correction
                        rescued_valid, rescued_reason = validate_correction_response(
                            parsed,
                            expected_drill_axis=expected_drill_axis,
                        )
                        if rescued_valid:
                            logger.info(
                                "correction rescued by truncation after validation exhaustion"
                            )
                            use_rescued = True
                            is_valid = True
                        else:
                            logger.warning(
                                "truncation rescue still invalid: %s", rescued_reason
                            )
            except Exception as rescue_err:
                logger.warning("truncation rescue failed: %s", rescue_err)

            if not use_rescued:
                logger.error(
                    "correction validation exhausted after %s attempts; last reason: %s",
                    max_validation_attempts,
                    last_validation_reason,
                )
                raise HTTPException(
                    status_code=503,
                    detail="批改服務暫時不穩定，請稍後再試",
                )

        # New structured contract: correction is a dict with four required
        # subfields (quoted / why_it_hurts / better_phrasing / next_task),
        # plus top-level tag / progress_note / on_topic. Validator above
        # has already confirmed shape + char limits; defensive extract below.
        correction_obj = parsed.get("correction") if isinstance(parsed.get("correction"), dict) else {}
        quoted             = (correction_obj.get("quoted") or "").strip()
        why_it_hurts       = (correction_obj.get("why_it_hurts") or "").strip()
        better_phrasing_en = (correction_obj.get("better_phrasing_en") or "").strip()
        better_phrasing_zh = (correction_obj.get("better_phrasing_zh") or "").strip()
        next_task          = (correction_obj.get("next_task") or "").strip()
        on_topic           = parsed.get("on_topic", True)
        # progress_note: surfaced in the API response only.
        # TODO: persist progress_note once admin dashboard consumes it
        # (will require ALTER TABLE practice_records ADD COLUMN progress_note text).
        progress_note      = (parsed.get("progress_note") or "").strip()


        # Transform the structured object back to the flat fields the frontend
        # currently reads (frontend stays untouched per spec). Mapping:
        #   coach_response = 「你說：『{quoted}』」 + blank line + why_it_hurts.
        #                    next_task is intentionally NOT joined into
        #                    coach_response — "一次只打一個點" (single-pain rule).
        #                    next_task is the "say it again now" repair directive;
        #                    it is exposed separately in response_payload for the
        #                    frontend normal flow to render, open to Free and Pro.
        #   better_expression    = better_phrasing_en
        #   better_expression_zh = better_phrasing_zh
        #   weakness_tag         = tag (renamed)
        #   next_question        = ""  (deprecated by spec)
        coach_response = ""
        if quoted and why_it_hurts:
            coach_response = f"你說：「{quoted}」\n\n{why_it_hurts}"
        elif why_it_hurts:
            # defensive: validator should already block empty quoted, but
            # keep this branch so a degraded LLM response still produces
            # *some* coach text instead of a 502.
            coach_response = why_it_hurts

        if not coach_response:
            raise HTTPException(status_code=502, detail="Coach response was empty, please retry")

        better_expression    = better_phrasing_en
        better_expression_zh = better_phrasing_zh
        next_question = pick_next_question(
            current_topic=topic,
            current_question=question,
            user_id=user_id,
        )

        # Groq now produces tag itself. Validate against the allow-list
        # so a hallucinated value never pollutes the DB / admin dashboards.
        weakness_tag = (parsed.get("tag") or "").strip()
        if weakness_tag not in ALLOWED_WEAKNESS_TAGS:
            if weakness_tag:
                logger.warning(
                    "Groq returned invalid tag: %r, falling back to empty",
                    weakness_tag,
                )
            weakness_tag = ""

        tag_secondary = (parsed.get("tag_secondary") or "").strip()
        if tag_secondary not in ALLOWED_WEAKNESS_TAGS:
            tag_secondary = ""
        if tag_secondary == weakness_tag:
            tag_secondary = ""

        tag_tertiary = (parsed.get("tag_tertiary") or "").strip()
        if tag_tertiary not in ALLOWED_WEAKNESS_TAGS:
            tag_tertiary = ""
        if tag_tertiary in (weakness_tag, tag_secondary):
            tag_tertiary = ""

        # witness_note depends on weakness_tag being validated, so it must be
        # computed AFTER the validation block above (spec literal placed it
        # after progress_note parse, but weakness_tag was undefined there).
        _witness = build_witness_note(total_practice_count, tag_counts or {}, weakness_tag)
        witness_note = _witness["text"]
        witness_is_milestone = _witness["is_milestone"]

        # Server-side persistence (was previously done client-side).
        # Uses supabase_admin (service_role) to bypass RLS. Failures are logged
        # but do NOT fail the response — the student still gets their feedback
        # even if the write breaks. `persisted` surfaces the state for the
        # client (or test harness) to act on.
        persisted = False
        new_record_id: Optional[str] = None
        if supabase_admin is not None:
            if is_drill_mode:
                drill_score_obj = parsed.get("drill_score")
                if not isinstance(drill_score_obj, dict):
                    drill_score_obj = None

                evidence_obj = drill_score_obj.get("evidence") if drill_score_obj else None
                if not isinstance(evidence_obj, (dict, list)) or evidence_obj == {} or evidence_obj == []:
                    evidence_obj = None

                payload = {
                    "user_id":              user_id,
                    "topic":                "Drill",
                    "question":             (question or "").strip(),
                    "user_transcript":      user_text or "",
                    "coach_response":       coach_response,
                    "better_expression":    better_expression,
                    "better_expression_zh": better_expression_zh,
                    "next_question":        next_question,
                    "weakness_tag":         drill_tag,
                    "memory_snapshot":      memory_snapshot,
                    "mode":                 "drill",
                    "drill_tag":            drill_tag,
                    "drill_score":          drill_score_obj,
                    "evidence":             evidence_obj,
                }

                try:
                    insert_resp = supabase_admin.table("practice_records").insert(payload).execute()
                    # 不要信任 200:寫入沒回傳 row 代表這次 drill 紀錄沒落地,
                    # drill 是 Pro 路徑,寫入斷掉不可靜默放行讓 UI 顯示成功。
                    rows = insert_resp.data or []
                    if not rows:
                        raise HTTPException(
                            status_code=500,
                            detail="Failed to persist practice record",
                        )
                    persisted = True
                    new_record_id = rows[0].get("id")
                except HTTPException:
                    raise
                except Exception as e:
                    logger.exception(
                        "drill practice_record insert failed",
                        extra={"user_id": user_id, "drill_tag": drill_tag, "error": str(e)},
                    )
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to persist practice record",
                    ) from e

                if new_record_id:
                    asyncio.create_task(classify_quality_background(
                        new_record_id,
                        user_text or "",
                        drill_tag or "",
                    ))
            else:
                normal_payload = {
                    "user_id":              user_id,
                    "topic":                topic or "",
                    "question":             (question or "").strip(),
                    "user_transcript":      user_text or "",
                    "coach_response":       coach_response,
                    "better_expression":    better_expression,
                    "better_expression_zh": better_expression_zh,
                    "next_question":        next_question,
                    "weakness_tag":         weakness_tag,
                    "memory_snapshot":      memory_snapshot,
                    "retry_of":             retry_of_clean,   # None 或合法 UUID 字串
                }
                try:
                    insert_resp = supabase_admin.table("practice_records").insert(normal_payload).execute()
                    # 不要信任 200:寫入沒回傳 row 代表這次 practice_record(含 retry_of
                    # 鏈路)可能沒落地,不可靜默放行讓 UI 顯示成功。
                    rows = insert_resp.data or []
                    if not rows:
                        raise HTTPException(
                            status_code=500,
                            detail="Failed to persist practice record",
                        )
                    persisted = True
                    # Capture the fresh row's id so the resolution lookup below
                    # can cleanly exclude it (avoids relying on ordering races).
                    new_record_id = rows[0].get("id")
                except HTTPException:
                    raise
                except Exception as exc:
                    logger.exception(
                        "failed to insert practice_record",
                        extra={"user_id": user_id},
                    )
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to persist practice record",
                    ) from exc

                if new_record_id:
                    asyncio.create_task(classify_quality_background(
                        new_record_id,
                        user_text or "",
                        weakness_tag or "",
                    ))

        # Auto-resolve the prior unresolved record for the SAME question if the
        # new submission landed a DIFFERENT valid weakness_tag. Semantics match
        # the spec exactly: pick the most recent prior (any tag), then check
        # whether to flip it. If the most recent prior has the SAME tag, leave
        # it alone — we do NOT skip over it to an older mismatched row.
        #
        # Failure here is swallowed: the student already got their feedback
        # and the new record is already persisted; backfilling resolved state
        # is not worth failing the response over.
        if (
            persisted
            and question
            and weakness_tag
            and weakness_tag in ALLOWED_WEAKNESS_TAGS
            and not is_drill_mode
            and supabase_admin is not None
        ):
            try:
                # Don't filter by question at the SQL layer — strict string
                # equality silently skips retries that differ by trailing
                # whitespace, curly vs straight apostrophes, or stray unicode.
                # Pull the top 5 unresolved rows and do normalised matching
                # in Python so two logically-identical question strings are
                # treated as the same session.
                prior_query = (
                    supabase_admin.table("practice_records")
                    .select("id, question, weakness_tag")
                    .eq("user_id", user_id)
                    .eq("resolved", False)
                    .order("created_at", desc=True)
                    .limit(5)
                )
                if new_record_id:
                    prior_query = prior_query.neq("id", new_record_id)
                prior_resp = prior_query.execute()
                candidates = prior_resp.data or []
                target = (question or "").strip()

                # Count total attempts on this question (resolved + unresolved,
                # including the just-inserted current record). Python-side
                # whitespace-normalized matching keeps the count and the match
                # below logically aligned, so a question string that drifted by
                # whitespace / curly-vs-straight apostrophe doesn't undercount.
                attempts_resp = (
                    supabase_admin.table("practice_records")
                    .select("id, question")
                    .eq("user_id", user_id)
                    .execute()
                )
                total_attempts = sum(
                    1 for r in (attempts_resp.data or [])
                    if (r.get("question") or "").strip() == target
                )

                # Force-resolve after 3 attempts on the same question, regardless
                # of tag progression. Takes priority over the tag-comparison logic
                # below: picks the most recent unresolved match from the
                # candidates already pulled (DESC by created_at) and flips it.
                if total_attempts >= 3:
                    for rec in candidates:
                        if (rec.get("question") or "").strip() == target:
                            supabase_admin.table("practice_records").update(
                                {"resolved": True}
                            ).eq("id", rec["id"]).execute()
                            logger.info(
                                "force-resolved after %s attempts: %s",
                                total_attempts, rec["id"],
                            )
                            break

                prior = next(
                    (
                        c for c in candidates
                        if (c.get("question") or "").strip() == target
                    ),
                    None,
                )
                if prior:
                    prior_tag = prior.get("weakness_tag")
                    prior_id = prior.get("id")
                    if prior_id and prior_tag != weakness_tag:
                        supabase_admin.table("practice_records").update(
                            {"resolved": True}
                        ).eq("id", prior_id).execute()
                        logger.info(
                            "auto-resolved prior practice_record %s "
                            "(tag %r → %r) for user %s",
                            prior_id, prior_tag, weakness_tag, user_id,
                        )
            except Exception:
                logger.exception(
                    "auto-resolve of prior practice_record failed",
                    extra={"user_id": user_id, "question": question},
                )

        # drill_usage INSERT (fire-and-forget). Runs after the LLM response
        # has been validated and unpacked, BEFORE response_payload assembly,
        # so the very next quota check reflects this drill. Failure is
        # logged but never propagated — coaching response must still ship.
        if is_drill_mode and supabase_admin is not None:
            try:
                drill_score_for_log = parsed.get("drill_score") or {}
                raw_score = drill_score_for_log.get("score")
                drill_score_value = (
                    raw_score
                    if isinstance(raw_score, int) and not isinstance(raw_score, bool)
                    else None
                )
                supabase_admin.table("drill_usage").insert({
                    "user_id":         user_id,
                    "drill_tag":       drill_tag,
                    "drill_score":     drill_score_value,
                    "is_pro_at_time":  is_pro,
                }).execute()
            except Exception:
                logger.exception(
                    "drill_usage insert failed (fire-and-forget)",
                    extra={"user_id": user_id, "drill_tag": drill_tag},
                )

        # is_pro is conditionally set inside the drill branch above.
        # Re-evaluate here so non-drill turns also have it for the
        # tag_secondary / tag_tertiary gate below. Cheap RPC; for drill
        # users the second call returns the same value as the first.
        is_pro = get_user_pro_status(user_id)
        response_payload = {
            "text":                 user_text,
            "coach_response":       coach_response,
            "next_question":        next_question,
            "better_expression":    better_expression,
            "better_expression_zh": better_expression_zh,
            "next_task":            next_task,
            "on_topic":             on_topic,
            "weakness_tag":         weakness_tag,
            "tag_secondary":        tag_secondary if is_pro else "",
            "tag_tertiary":         tag_tertiary if is_pro else "",
            "memory_snapshot":      memory_snapshot,
            "progress_note":        progress_note,
            "witness_note":         witness_note,
            "witness_is_milestone": witness_is_milestone,
            "persisted":            persisted,
            "record_id":            new_record_id,
        }
        # Drill mode adds drill_score; non-drill turns return identical shape
        # to the previous version (acceptance: existing /process callers
        # unaffected when mode != "drill").
        if is_drill_mode:
            drill_score_data = parsed.get("drill_score")
            if isinstance(drill_score_data, dict):
                _ds = {
                    "axis":             drill_score_data.get("axis"),
                    "score":            drill_score_data.get("score"),
                    "feedback":         (drill_score_data.get("feedback") or "").strip(),
                    "threshold_passed": drill_score_data.get("threshold_passed"),
                }
                if is_pro:
                    _ds["evidence"] = drill_score_data.get("evidence")
                response_payload["drill_score"] = _ds
                raw_drill_score = drill_score_data.get("score")
                if isinstance(raw_drill_score, int) and not isinstance(raw_drill_score, bool):
                    try:
                        update_user_band(user_id, score_to_band_estimate(raw_drill_score))
                    except Exception:
                        logger.exception(
                            "update_user_band failed (non-fatal)",
                            extra={"user_id": user_id},
                        )

        # Vocab suggestion enrichment — fail-open, never blocks the main
        # response. Two optional fields:
        #   suggested_vocab        : best match to the better_phrasing_en hint
        #   vocab_recommendations  : 3 same-topic items when weakness=weak_vocab
        try:
            VOCAB_SUGGEST_STOPWORDS = {
                "a", "an", "the", "to", "it", "is", "be",
                "of", "in", "on", "at", "for", "and", "or", "but",
            }
            VOCAB_RETURN_COLS = "id, word, zh_meaning, common_chunk, topic"
            suggested_vocab: Optional[dict] = None

            if better_phrasing_en and supabase_admin is not None:
                tokens = [
                    t.strip(".,;:!?\"'()[]").lower()
                    for t in better_phrasing_en.split()
                ]
                target_word = next(
                    (t for t in tokens if t and t not in VOCAB_SUGGEST_STOPWORDS),
                    None,
                )
                if target_word:
                    try:
                        # Exact match (case-insensitive via lowercased compare)
                        exact = (
                            supabase_admin.table("vocabulary_items")
                            .select(VOCAB_RETURN_COLS)
                            .ilike("word", target_word)
                            .limit(1)
                            .execute()
                        )
                        rows = exact.data or []
                        if not rows:
                            # Substring fallback
                            fuzzy = (
                                supabase_admin.table("vocabulary_items")
                                .select(VOCAB_RETURN_COLS)
                                .ilike("word", f"%{target_word}%")
                                .limit(1)
                                .execute()
                            )
                            rows = fuzzy.data or []
                        if rows:
                            suggested_vocab = {
                                "id":           rows[0].get("id"),
                                "word":         rows[0].get("word"),
                                "zh_meaning":   rows[0].get("zh_meaning"),
                                "common_chunk": rows[0].get("common_chunk"),
                                "topic":        rows[0].get("topic"),
                            }
                    except Exception:
                        logger.exception(
                            "[/process] suggested_vocab lookup failed (omitted)",
                            extra={"target_word": target_word},
                        )

            if suggested_vocab:
                response_payload["suggested_vocab"] = suggested_vocab

            # vocab_recommendations: only for weak_vocab with a topic. Random
            # ordering done client-side (PostgREST has no order=random); the
            # per-topic catalog is small (≤ tens of rows) so this is cheap.
            practice_topic = (topic or "").strip()
            if (
                weakness_tag == "weak_vocab"
                and practice_topic
                and supabase_admin is not None
            ):
                try:
                    pool_resp = (
                        supabase_admin.table("vocabulary_items")
                        .select("id, word, zh_meaning, common_chunk")
                        .eq("topic", practice_topic.lower())
                        .execute()
                    )
                    pool = list(pool_resp.data or [])
                    if suggested_vocab and suggested_vocab.get("id"):
                        sid = suggested_vocab["id"]
                        pool = [r for r in pool if r.get("id") != sid]
                    if pool:
                        random.shuffle(pool)
                        response_payload["vocab_recommendations"] = pool[:3]
                except Exception:
                    logger.exception(
                        "[/process] vocab_recommendations lookup failed (omitted)",
                        extra={"topic": practice_topic, "weakness_tag": weakness_tag},
                    )
        except Exception:
            logger.exception("[/process] vocab enrichment outer guard tripped")

        return response_payload
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "process endpoint failed",
            extra={"user_id": user_id if "user_id" in locals() else None},
        )
        raise HTTPException(status_code=500, detail="Internal error, please try again") from e


@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


@app.post("/api/webhooks/lemonsqueezy")
async def lemonsqueezy_webhook(request: Request):
    """
    LemonSqueezy subscription webhook → flips profiles.is_pro.

    Signature verification uses HMAC-SHA256 over the raw body with
    LEMONSQUEEZY_WEBHOOK_SECRET (set in Render env). Without that env
    var we reject everything — never silently accept unsigned events.

    Email lookup goes through the get_user_id_by_email() Postgres
    helper because profiles has no email column; the actual email
    lives in auth.users which is only reachable via service role.
    """
    if not LEMONSQUEEZY_WEBHOOK_SECRET:
        logger.error("[webhook/ls] LEMONSQUEEZY_WEBHOOK_SECRET not set")
        raise HTTPException(status_code=503, detail="Webhook not configured")
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    body = await request.body()
    sig = request.headers.get("X-Signature", "")
    expected = hmac.new(
        LEMONSQUEEZY_WEBHOOK_SECRET.encode(), body, hashlib.sha256
    ).hexdigest()
    if not hmac.compare_digest(expected, sig):
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    event = payload.get("meta", {}).get("event_name", "")
    # Webhook ONLY controls profiles.is_pro (paid status). Manual admin
    # grants live in profiles.is_pro_grant and are never touched here —
    # so a cancellation can no longer silently revoke a beta-tester grant.
    HANDLED = {
        "subscription_created":         True,   # → is_pro = true
        "subscription_updated":         None,   # depends on attrs.status
        "subscription_resumed":         True,
        "subscription_unpaused":        True,
        "subscription_cancelled":       False,  # → is_pro = false
        "subscription_expired":         False,
        "subscription_paused":          False,
        "subscription_payment_failed":  False,  # treat as paid-status loss
    }
    if event not in HANDLED:
        return {"status": "ignored", "event": event}

    attrs = payload.get("data", {}).get("attributes", {})
    status = attrs.get("status")
    email = (attrs.get("user_email") or "").strip().lower()
    if not email:
        raise HTTPException(status_code=400, detail="No email in payload")

    try:
        result = supabase_admin.rpc(
            "get_user_id_by_email", {"email_input": email}
        ).execute()
    except Exception as exc:
        logger.exception("[webhook/ls] get_user_id_by_email rpc failed")
        raise HTTPException(status_code=500, detail="User lookup failed") from exc

    user_id = result.data
    if not user_id:
        logger.warning("[webhook/ls] unknown email: %s", email)
        return {"status": "user_not_found"}

    forced = HANDLED[event]
    is_pro = (status == "active") if forced is None else forced
    now_iso = datetime.now(timezone.utc).isoformat()

    try:
        upd = supabase_admin.table("profiles") \
            .update({"is_pro": is_pro, "updated_at": now_iso}) \
            .eq("id", user_id) \
            .execute()
    except Exception as exc:
        logger.exception("[webhook/ls] profiles update failed")
        raise HTTPException(status_code=500, detail="Profile update failed") from exc

    if not upd.data:
        logger.error("[webhook/ls] profile update returned no rows for %s", user_id)
        raise HTTPException(status_code=500, detail="Profile update returned no rows")

    logger.info(
        "[webhook/ls] %s → is_pro=%s (event=%s, status=%s)",
        email, is_pro, event, status,
    )
    return {"status": "ok", "is_pro": is_pro}


# ─── Covenant (entry ritual) ──────────────────────────────────────────────────
@app.get("/api/covenant/status")
@limiter.limit("60/minute")
async def covenant_status(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Has the caller signed the entry covenant? Drives the gate modal on the
    practice page. Missing profile → unsigned, never 500 (the ritual is not a
    paywall; a transient DB miss shouldn't lock returning users out)."""
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        resp = (
            supabase_admin.table("profiles")
            .select("covenant_signed_at, covenant_name")
            .eq("id", user_id)
            .limit(1)
            .execute()
        )
    except Exception:
        logger.exception("covenant status query failed", extra={"user_id": user_id})
        raise HTTPException(status_code=503, detail="Failed to load covenant status")

    rows = resp.data or []
    if not rows:
        return {"signed": False, "name": None}
    row = rows[0]
    return {
        "signed": bool(row.get("covenant_signed_at")),
        "name": row.get("covenant_name"),
    }


@app.post("/api/covenant/sign")
@limiter.limit("10/minute")
async def covenant_sign(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Record the entry-ritual signature. fail-fast: validates name length,
    verifies the row truly updated before reporting success — never trusts
    a 200 without inspecting `data`."""
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=422, detail="芳名不可空缺,須親署方得入室。")

    name = (body.get("name") if isinstance(body, dict) else "") or ""
    name = name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="芳名不可空缺,須親署方得入室。")
    if len(name) > 40:
        raise HTTPException(status_code=422, detail="芳名過長,還請簡署。")

    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        upd = (
            supabase_admin.table("profiles")
            .update({"covenant_signed_at": now_iso, "covenant_name": name})
            .eq("id", user_id)
            .execute()
        )
    except Exception:
        logger.exception("covenant sign update failed", extra={"user_id": user_id})
        raise HTTPException(status_code=500, detail="契約未能銘記,還請再署一次。")

    if not upd.data:
        logger.error("covenant sign returned no rows", extra={"user_id": user_id})
        raise HTTPException(status_code=500, detail="契約未能銘記,還請再署一次。")

    return {"signed": True, "name": name, "signed_at": now_iso}


# ─── Payment (ECPay/綠界 skeleton) ────────────────────────────────────────────
# Skeleton only — replace payment_url and add CheckMacValue verification once
# the merchant credentials are issued. Pro entitlement uses the existing
# profiles.is_pro mechanism; subscriptions table is the audit/billing log.

@app.post("/api/payment/create-order")
@limiter.limit("5/minute")
async def payment_create_order(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """建立訂單骨架，等綠界 API key 到再填入真實付款 URL。"""
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    order_id = f"BLABBY-{uuid.uuid4().hex[:12].upper()}"

    try:
        supabase_admin.table("subscriptions").insert({
            "user_id": user_id,
            "order_id": order_id,
            "plan":     "monthly",
            "status":   "pending",
            "amount":   299,
        }).execute()
    except Exception:
        logger.exception("create_order insert failed", extra={"user_id": user_id})
        raise HTTPException(status_code=503, detail="Failed to create order")

    # TODO: replace with real ECPay payment page URL once credentials land.
    return {
        "order_id":    order_id,
        "payment_url": f"/upgrade.html?pending={order_id}",
        "amount":      299,
    }


@app.post("/api/payment/callback")
async def payment_callback(request: Request):
    """
    綠界付款完成 webhook。Form-encoded body.
    TODO: verify CheckMacValue before trusting MerchantTradeNo / RtnCode.
    成功 (RtnCode=1) → subscriptions.status=active + profiles.is_pro=true。
    Always 200 — webhook providers retry on non-2xx.
    """
    try:
        form = await request.form()
        order_id       = form.get("MerchantTradeNo") or ""
        payment_status = form.get("RtnCode") or ""
        logger.info("payment_callback order_id=%s status=%s", order_id, payment_status)

        if payment_status == "1" and supabase_admin is not None:
            now = datetime.now(timezone.utc)
            now_iso = now.isoformat()
            expires_iso = (now + timedelta(days=30)).isoformat()

            supabase_admin.table("subscriptions").update({
                "status":     "active",
                "started_at": now_iso,
                "expires_at": expires_iso,
                "updated_at": now_iso,
            }).eq("order_id", order_id).execute()

            sub_resp = (
                supabase_admin.table("subscriptions")
                .select("user_id")
                .eq("order_id", order_id)
                .limit(1)
                .execute()
            )
            if sub_resp.data:
                uid = sub_resp.data[0]["user_id"]
                # Mirror the LemonSqueezy webhook path: paid status lives
                # on profiles.is_pro (PK = id, not user_id). Admin grants
                # are kept separate on profiles.is_pro_grant.
                supabase_admin.table("profiles").update({
                    "is_pro":     True,
                    "updated_at": now_iso,
                }).eq("id", uid).execute()

        return {"status": "ok"}
    except Exception:
        logger.exception("payment_callback failed")
        return {"status": "error"}


@app.get("/api/payment/return")
async def payment_return(request: Request):
    """用戶付款後跳回。TODO: 改成完整 frontend URL 後再給綠界 OrderResultURL。"""
    order_id = request.query_params.get("MerchantTradeNo", "")
    return RedirectResponse(url=f"/success.html?order={order_id}")


@app.get("/api/user/subscription")
@limiter.limit("20/minute")
async def get_user_subscription(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """回傳用戶目前的有效訂閱（status=active, 取最新一筆）。"""
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        resp = (
            supabase_admin.table("subscriptions")
            .select("id, plan, status, amount, started_at, expires_at, created_at")
            .eq("user_id", user_id)
            .eq("status", "active")
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception:
        logger.exception("get_user_subscription failed", extra={"user_id": user_id})
        raise HTTPException(status_code=503, detail="Failed to load subscription")
    return {"subscription": resp.data[0] if resp.data else None}


# ─── Admin: subscription management ──────────────────────────────────────────

@app.get("/api/admin/subscriptions")
@limiter.limit("30/minute")
async def admin_list_subscriptions(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """List recent 100 subscriptions with email resolved.
    PostgREST nested-select on auth.users isn't allowed (different schema),
    so we resolve emails per row via the admin auth API. Slow on big lists
    but admin-only and capped at 100.
    """
    verify_admin(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        resp = (
            supabase_admin.table("subscriptions")
            .select("id, user_id, order_id, plan, status, amount, started_at, expires_at, created_at, updated_at")
            .order("created_at", desc=True)
            .limit(100)
            .execute()
        )
    except Exception:
        logger.exception("admin_list_subscriptions select failed")
        raise HTTPException(status_code=503, detail="Failed to load subscriptions")

    rows = resp.data or []
    # Cache email lookups so 5 subs from the same user → 1 API call.
    email_cache: dict[str, str] = {}
    for row in rows:
        uid = row.get("user_id")
        if not uid:
            row["email"] = None
            continue
        if uid not in email_cache:
            try:
                u = supabase_admin.auth.admin.get_user_by_id(uid)
                email_cache[uid] = (getattr(getattr(u, "user", None), "email", None) or "")
            except Exception:
                email_cache[uid] = ""
        row["email"] = email_cache[uid] or None
    return {"subscriptions": rows}


@app.post("/api/admin/subscription/extend")
@limiter.limit("20/minute")
async def admin_extend_subscription(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Push expires_at forward by `days` (default 30) and force status=active."""
    verify_admin(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    body = await request.json()
    sub_id = body.get("subscription_id")
    days = int(body.get("days") or 30)
    if not sub_id:
        raise HTTPException(status_code=400, detail="subscription_id required")
    try:
        sub = (
            supabase_admin.table("subscriptions")
            .select("expires_at, user_id")
            .eq("id", sub_id)
            .limit(1)
            .execute()
        )
    except Exception:
        logger.exception("admin_extend_subscription select failed")
        raise HTTPException(status_code=503, detail="Subscription lookup failed")
    if not sub.data:
        raise HTTPException(status_code=404, detail="Subscription not found")
    current = sub.data[0].get("expires_at")
    base = (
        datetime.fromisoformat(current.replace("Z", "+00:00"))
        if current else datetime.now(timezone.utc)
    )
    # Don't extend from a past date — anchor on max(now, current_expiry).
    base = max(base, datetime.now(timezone.utc))
    new_expires = base + timedelta(days=days)
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        supabase_admin.table("subscriptions").update({
            "expires_at": new_expires.isoformat(),
            "status":     "active",
            "updated_at": now_iso,
        }).eq("id", sub_id).execute()
        # Re-promote the user to Pro in case status had lapsed.
        uid = sub.data[0].get("user_id")
        if uid:
            supabase_admin.table("profiles").update({
                "is_pro":     True,
                "updated_at": now_iso,
            }).eq("id", uid).execute()
    except Exception:
        logger.exception("admin_extend_subscription update failed")
        raise HTTPException(status_code=503, detail="Update failed")
    return {"extended_to": new_expires.isoformat()}


@app.post("/api/admin/subscription/cancel")
@limiter.limit("20/minute")
async def admin_cancel_subscription(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Mark subscription cancelled. Does NOT immediately revoke profiles.is_pro
    — that's a separate call. The expires_at remains, so the user keeps Pro
    features until their paid window ends (industry-standard behaviour)."""
    verify_admin(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    body = await request.json()
    sub_id = body.get("subscription_id")
    if not sub_id:
        raise HTTPException(status_code=400, detail="subscription_id required")
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        supabase_admin.table("subscriptions").update({
            "status":     "cancelled",
            "updated_at": now_iso,
        }).eq("id", sub_id).execute()
    except Exception:
        logger.exception("admin_cancel_subscription update failed")
        raise HTTPException(status_code=503, detail="Cancel failed")
    return {"status": "cancelled"}


# ─── Resume flow (Practice Hub) ───────────────────────────────────────────────
# These back the "你上次卡在：..." CTA. Source of truth is the DB's
# practice_records.resolved column; localStorage is no longer consulted.

@app.get("/api/practice-records/last-unresolved")
@limiter.limit("10/minute")
async def last_unresolved_practice_record(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Return the caller's most recent practice_record where resolved=false.
    No unresolved record is a normal state, not an error — respond with
    HTTP 200 and `null` so the client can render Scenario B cleanly.
    """
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        response = (
            supabase_admin.table("practice_records")
            .select("id, question, topic, weakness_tag, coach_response, created_at")
            .eq("user_id", user_id)
            .eq("resolved", False)
            .order("created_at", desc=True)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.exception("last-unresolved query failed", extra={"user_id": user_id})
        raise HTTPException(status_code=500, detail="Failed to load last session") from exc

    rows = response.data or []
    return rows[0] if rows else None


def _build_weakness_summary(rows: list[dict], is_pro: bool = False) -> dict:
    """
    Pure function: rows → response shape. Separated so tests can drive
    the aggregation without touching the DB.

    is_pro is currently a no-op flag — see TODO below.
    """
    # TODO[pro-gate]: free 顯示 top 3, pro 顯示全部 + trend over time
    # When the gate ships, slice the distribution at 3 for !is_pro and
    # add a trend payload (counts per day for the last 14 days).
    counts: Counter = Counter()
    for row in rows:
        tag = (row.get("weakness_tag") or "").strip()
        if not tag:
            continue
        counts[tag] += 1

    # Sort: count desc, then tag alphabetical (stable tie-break per spec).
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    distribution = [
        {
            "tag":   tag,
            # Unknown tags fall back to the raw tag string — never raise,
            # never return None, so frontend rendering never breaks on a
            # future backend tag that the labels mirror hasn't picked up.
            "label": WEAKNESS_LABELS.get(tag, tag),
            "count": count,
        }
        for tag, count in ordered
    ]
    return {
        "total_practices": sum(counts.values()),
        "weakness_distribution": distribution,
    }


@app.get("/api/history")
@limiter.limit("30/minute")
async def get_history(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Return most-recent practice_records for the caller, newest first.
    Used by /history.html to render the "Your Practice History" page.

    Pro gating is enforced SERVER-SIDE:
      - Free users: 10 most-recent records
      - Pro users:  up to 50 most-recent records (the legacy cap)
    Response includes "capped": True if the user hit the free limit, so
    the frontend can render an upgrade nudge without re-deriving plan
    status from a separate endpoint.
    """
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    is_pro = get_user_pro_status(user_id)
    history_limit = 50 if is_pro else 10
    try:
        resp = (
            supabase_admin.table("practice_records")
            .select(
                "id, created_at, topic, question, user_transcript, "
                "coach_response, better_expression, better_expression_zh, "
                "weakness_tag, drill_score, mode"
            )
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(history_limit)
            .execute()
        )
    except Exception:
        logger.exception("get_history query failed", extra={"user_id": user_id})
        raise HTTPException(status_code=503, detail="Failed to load history")
    return {"records": resp.data or [], "capped": not is_pro}


@app.get("/api/progress")
@limiter.limit("30/minute")
async def get_progress(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Returns first vs latest attempt comparison for each question the user
    has practised at least twice. Drives /progress.html, which makes
    perceived improvement legible — directly targets the "感覺像只有一個
    功能，太簡單了" feedback by giving the user evidence of their own
    longitudinal change.

    For each question with >= 2 records:
      - first  = earliest attempt (transcript + created_at)
      - latest = most recent attempt (transcript + created_at)
      - delta  = word_delta + sentence_delta, computed server-side so the
                 client never has to reason about tokenisation

    Sorted by latest.created_at DESC so recently revisited questions
    appear first. Returns a top-level JSON array.

    Records with blank question or blank user_transcript are skipped —
    they can't form a meaningful comparison.
    """
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        resp = (
            supabase_admin.table("practice_records")
            .select("question, user_transcript, created_at")
            .eq("user_id", user_id)
            # Ascending so attempts[0] is the first attempt for each
            # question and attempts[-1] is the latest, without a second
            # min/max pass per group.
            .order("created_at", desc=False)
            .execute()
        )
    except Exception:
        logger.exception("get_progress query failed", extra={"user_id": user_id})
        raise HTTPException(status_code=503, detail="Failed to load progress")

    records = resp.data or []

    by_question: dict[str, list[dict]] = {}
    for r in records:
        q = (r.get("question") or "").strip()
        if not q:
            continue
        t = (r.get("user_transcript") or "").strip()
        if not t:
            # No transcript = nothing to compare. Skip rather than
            # surfacing a misleading 0-word "first attempt".
            continue
        by_question.setdefault(q, []).append(r)

    def _sentence_count(text: str) -> int:
        # Split on terminal punctuation; count non-empty fragments. Treats
        # "..." and "?!" as single boundaries. Good enough for IELTS
        # speaking transcripts where punctuation is Whisper-supplied.
        return len([frag for frag in re.split(r"[.!?]+", text) if frag.strip()])

    items: list[dict] = []
    for question, attempts in by_question.items():
        if len(attempts) < 2:
            continue
        first = attempts[0]
        latest = attempts[-1]
        first_t = (first.get("user_transcript") or "").strip()
        latest_t = (latest.get("user_transcript") or "").strip()

        items.append({
            "question": question,
            "first": {
                "transcript": first_t,
                "created_at": first.get("created_at"),
            },
            "latest": {
                "transcript": latest_t,
                "created_at": latest.get("created_at"),
            },
            "delta": {
                "word_delta":     len(latest_t.split()) - len(first_t.split()),
                "sentence_delta": _sentence_count(latest_t) - _sentence_count(first_t),
            },
        })

    items.sort(key=lambda x: x["latest"]["created_at"] or "", reverse=True)
    return items


@app.get("/api/diagnosis/timeline")
@limiter.limit("20/minute")
async def diagnosis_timeline(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Returns (Pro):
    - first_session: earliest created_at
    - total_sessions: count of all practice_records
    - topics_seen: distinct topics
    - weakness_timeline: per weakness_tag, list of {week, count}
    - weakness_first_seen: per tag, first created_at
    - weakness_last_seen: per tag, last created_at
    - weakness_counts: per tag, total count
    - recent_trend: last 14 days vs prior 14 days per tag (delta)
    - capped: False

    Returns (Free, server-side strip):
    - first_session, total_sessions, topics_seen (kept)
    - weakness_counts: only the single top tag
    - capped: True
    Timeline / first_seen / last_seen / recent_trend are NOT included
    so the upgrade nudge in the frontend has a clear gap to point at.
    """
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    is_pro = get_user_pro_status(user_id)
    try:
        resp = (
            supabase_admin.table("practice_records")
            .select("created_at, weakness_tag, topic")
            .eq("user_id", user_id)
            .order("created_at", desc=False)
            .execute()
        )
    except Exception:
        logger.exception("diagnosis_timeline query failed", extra={"user_id": user_id})
        raise HTTPException(status_code=503, detail="Failed to load timeline")

    rows = resp.data or []
    if not rows:
        empty_base = {
            "first_session": None,
            "total_sessions": 0,
            "topics_seen": [],
            "weakness_counts": {},
            "capped": not is_pro,
        }
        if is_pro:
            empty_base.update({
                "weakness_timeline": {},
                "weakness_first_seen": {},
                "weakness_last_seen": {},
                "recent_trend": {},
            })
        return empty_base

    from collections import defaultdict

    first_session = rows[0]["created_at"]
    total_sessions = len(rows)
    topics_seen = list({r["topic"] for r in rows if r.get("topic")})

    tag_rows: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        tag = (r.get("weakness_tag") or "").strip()
        if tag:
            tag_rows[tag].append(r["created_at"])

    weakness_first_seen = {tag: dates[0] for tag, dates in tag_rows.items()}
    weakness_last_seen = {tag: dates[-1] for tag, dates in tag_rows.items()}
    weakness_counts = {tag: len(dates) for tag, dates in tag_rows.items()}

    weakness_timeline = {}
    for tag, dates in tag_rows.items():
        buckets: dict[str, int] = defaultdict(int)
        for d in dates:
            dt = datetime.fromisoformat(d.replace("Z", "+00:00"))
            week = dt.strftime("%Y-W%W")
            buckets[week] += 1
        weakness_timeline[tag] = [
            {"week": w, "count": c} for w, c in sorted(buckets.items())
        ]

    now = datetime.now(timezone.utc)
    cutoff_recent = now - timedelta(days=14)
    cutoff_prior = now - timedelta(days=28)
    recent_trend = {}
    for tag, dates in tag_rows.items():
        recent = sum(
            1 for d in dates
            if datetime.fromisoformat(d.replace("Z", "+00:00")) >= cutoff_recent
        )
        prior = sum(
            1 for d in dates
            if cutoff_prior <= datetime.fromisoformat(d.replace("Z", "+00:00")) < cutoff_recent
        )
        recent_trend[tag] = {"recent": recent, "prior": prior, "delta": recent - prior}

    if is_pro:
        return {
            "first_session": first_session,
            "total_sessions": total_sessions,
            "topics_seen": topics_seen,
            "weakness_timeline": weakness_timeline,
            "weakness_first_seen": weakness_first_seen,
            "weakness_last_seen": weakness_last_seen,
            "weakness_counts": weakness_counts,
            "recent_trend": recent_trend,
            "capped": False,
        }

    # Free user: strip everything except the top-1 weakness tag.
    top_only = {}
    if weakness_counts:
        top_tag, top_count = max(weakness_counts.items(), key=lambda x: x[1])
        top_only = {top_tag: top_count}
    return {
        "first_session": first_session,
        "total_sessions": total_sessions,
        "topics_seen": topics_seen,
        "weakness_counts": top_only,
        "capped": True,
    }


@app.get("/api/practice-records/weakness-summary")
@limiter.limit("10/minute")
async def practice_records_weakness_summary(
    request: Request,
    authorization: Optional[str] = Header(None),
    last_n: int = Query(10, ge=1, le=50),
):
    """
    Aggregate the caller's last `last_n` practice_records by weakness_tag.
    Returns { total_practices, weakness_distribution: [{tag, label, count}] }
    sorted by count desc with tag alphabetical tie-break.

    Filter: weakness_tag IS NOT NULL AND != ''. Resolved status is NOT
    filtered — diagnosis must show the full picture (4/24 red line).

    Empty result → {"total_practices": 0, "weakness_distribution": []}.
    """
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        # Filter weakness_tag IS NOT NULL at the DB layer; the empty-string
        # filter happens in Python (PostgREST `.neq("col", "")` works but
        # the helper sticks to filters used elsewhere in this file).
        response = (
            supabase_admin.table("practice_records")
            .select("weakness_tag, created_at")
            .eq("user_id", user_id)
            .not_.is_("weakness_tag", "null")
            .order("created_at", desc=True)
            .limit(last_n)
            .execute()
        )
    except Exception:
        logger.exception(
            "weakness-summary query failed", extra={"user_id": user_id}
        )
        raise HTTPException(
            status_code=503,
            detail="Failed to load weakness summary, please try again",
        )

    rows = response.data or []
    # TODO[pro-gate]: pass real is_pro once free/pro split lands.
    return _build_weakness_summary(rows, is_pro=True)


@app.get("/api/questions/bank")
@limiter.limit("30/minute")
async def get_question_bank(request: Request):
    """Return all Part 1 questions for client-side bank (no auth required)."""
    try:
        resp = supabase_admin.table("questions") \
            .select("text, topic, part") \
            .eq("part", 1) \
            .execute()
        questions = [
            {"question": r["text"], "topic": r["topic"]}
            for r in (resp.data or [])
        ]
        return {"questions": questions}
    except Exception as exc:
        logger.exception("Failed to fetch question bank")
        raise HTTPException(status_code=500, detail="Failed to fetch question bank")


@app.get("/api/questions/next")
@limiter.limit("10/minute")
async def next_question(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Return one part=2 question for the caller, biased away from anything they've
    practiced in the last QUESTION_EXCLUSION_DAYS days. Selection is purely
    diversity-driven (no weakness_tag-based recommendation):

      Tier 1 — not recently practiced AND on a topic the user has never touched
      Tier 2 — not recently practiced (any topic)
      Tier 3 — pure random across the whole pool (theoretical fallback only)

    503 if the questions pool is empty; generic 500 on unexpected failure.
    """
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        cutoff = (
            datetime.now(timezone.utc) - timedelta(days=QUESTION_EXCLUSION_DAYS)
        ).isoformat()
        recent_resp = (
            supabase_admin.table("practice_records")
            .select("question")
            .eq("user_id", user_id)
            .gte("created_at", cutoff)
            .execute()
        )
        recent_practiced_set = {
            (row.get("question") or "").strip()
            for row in (recent_resp.data or [])
            if (row.get("question") or "").strip()
        }

        topics_resp = (
            supabase_admin.table("practice_records")
            .select("topic")
            .eq("user_id", user_id)
            .execute()
        )
        historical_topics_set = {
            (row.get("topic") or "").strip()
            for row in (topics_resp.data or [])
            if (row.get("topic") or "").strip()
        }

        questions_resp = (
            supabase_admin.table("questions")
            .select("id, text, topic, part")
            .eq("part", 1)
            .execute()
        )
        questions = questions_resp.data or []
        if not questions:
            raise HTTPException(
                status_code=503, detail="Question pool not initialized"
            )

        not_recent = [
            q for q in questions
            if (q.get("text") or "").strip() not in recent_practiced_set
        ]
        tier1 = [
            q for q in not_recent
            if (q.get("topic") or "").strip() not in historical_topics_set
        ]

        if tier1:
            chosen = random.choice(tier1)
        elif not_recent:
            chosen = random.choice(not_recent)
        else:
            chosen = random.choice(questions)

        previous_transcript = None
        previous_practiced_at = None
        try:
            prev_resp = (
                supabase_admin.table("practice_records")
                .select("user_transcript, created_at")
                .eq("user_id", user_id)
                .eq("question", (chosen.get("text") or "").strip())
                .not_.is_("user_transcript", "null")
                .order("created_at", desc=True)
                .limit(1)
                .execute()
            )
            prev_rows = prev_resp.data or []
            if prev_rows and (prev_rows[0].get("user_transcript") or "").strip():
                previous_transcript = prev_rows[0]["user_transcript"].strip()
                previous_practiced_at = prev_rows[0].get("created_at")
        except Exception:
            logger.exception(
                "previous_transcript lookup failed (non-fatal)",
                extra={"user_id": user_id},
            )

        return {
            "id":                    chosen.get("id"),
            "text":                  chosen.get("text"),
            "topic":                 chosen.get("topic"),
            "part":                  chosen.get("part"),
            "previous_transcript":   previous_transcript,
            "previous_practiced_at": previous_practiced_at,
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "next-question selection failed", extra={"user_id": user_id}
        )
        raise HTTPException(status_code=500, detail="Failed to load next question")


def score_to_band_estimate(score: int) -> float:
    """Map drill_score (0-100) to approximate IELTS band for persona bucketing."""
    if score < 30:
        return 4.0
    elif score < 50:
        return 4.5
    elif score < 70:
        return 5.5
    elif score < 90:
        return 6.5
    return 7.5


def get_persona(user_band: Optional[float]) -> str:
    """
    Persona label from user_band. Stable thresholds, never drifts.
    None (new user) defaults to A.
    """
    if user_band is None or user_band < 4.5:
        return "A"
    elif user_band < 5.5:
        return "B"
    elif user_band < 6.5:
        return "C"
    return "D"


def _get_user_band(user_id: str) -> Optional[float]:
    """Read profiles.user_band. Returns None on any error (safe default → Persona A)."""
    if supabase_admin is None:
        return None
    try:
        resp = (
            supabase_admin.table("profiles")
            .select("user_band")
            .eq("id", user_id)
            .single()
            .execute()
        )
        val = (resp.data or {}).get("user_band")
        return float(val) if val is not None else None
    except Exception:
        logger.exception("_get_user_band failed", extra={"user_id": user_id})
        return None


def update_user_band(user_id: str, new_estimate: float) -> float:
    """
    Weighted moving average: existing 80%, new 20%.
    If user_band is null, sets directly. Fails safe — caller catches exceptions.
    """
    if supabase_admin is None:
        return new_estimate
    resp = (
        supabase_admin.table("profiles")
        .select("user_band")
        .eq("id", user_id)
        .single()
        .execute()
    )
    current = (resp.data or {}).get("user_band")
    updated = (
        round(float(current) * 0.8 + new_estimate * 0.2, 2)
        if current is not None
        else new_estimate
    )
    supabase_admin.table("profiles").update({
        "user_band":        updated,
        "band_updated_at":  datetime.now(timezone.utc).isoformat(),
    }).eq("id", user_id).execute()
    return updated


def get_user_pro_status(user_id: str) -> bool:
    """
    Return True if the user is Pro by ANY mechanism — paid (LemonSqueezy
    webhook → profiles.is_pro) or granted (admin → profiles.is_pro_grant).
    Backed by the is_user_pro() Postgres helper so the OR logic lives in
    one place.

    Fails safe: any error returns False so free quota enforcement is
    never accidentally bypassed.
    """
    if supabase_admin is None:
        return False
    try:
        resp = supabase_admin.rpc("is_user_pro", {"user_id": user_id}).execute()
        return bool(resp.data)
    except Exception:
        logger.exception("get_user_pro_status failed", extra={"user_id": user_id})
        return False


async def is_user_pro(user_id: str) -> bool:
    """
    Async parallel of get_user_pro_status(). Backed by the same RPC.
    Use this from new async code paths; existing callers can stay on
    get_user_pro_status() until refactored.

    Fails safe: any error returns False.
    """
    if supabase_admin is None:
        return False
    try:
        resp = supabase_admin.rpc("is_user_pro", {"user_id": user_id}).execute()
        return bool(resp.data)
    except Exception:
        logger.exception("is_user_pro failed", extra={"user_id": user_id})
        return False


def _drill_quota_state(user_id: str) -> tuple[int, Optional[str]]:
    """
    Count how many drill_usage rows the user has created within the rolling
    DRILL_QUOTA_WINDOW_DAYS window, and return (drill_count, quota_resets_at).

    quota_resets_at = oldest_in_window.created_at + window_days, formatted
    as ISO 8601. That timestamp is when the oldest in-window drill falls off
    the rolling count, freeing up one slot. Returns None when drill_count == 0.

    Failure is bubbled up to the caller — quota MUST be authoritative; we
    never silently zero-out the count.
    """
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    cutoff_dt = datetime.now(timezone.utc) - timedelta(days=DRILL_QUOTA_WINDOW_DAYS)
    cutoff_iso = cutoff_dt.isoformat()
    resp = (
        supabase_admin.table("drill_usage")
        .select("created_at")
        .eq("user_id", user_id)
        .gte("created_at", cutoff_iso)
        .order("created_at", desc=False)
        .execute()
    )
    rows = resp.data or []
    drill_count = len(rows)
    if drill_count == 0:
        return 0, None
    oldest_iso = (rows[0].get("created_at") or "").strip()
    if not oldest_iso:
        return drill_count, None
    # Postgres returns ISO timestamps with timezone offset; fromisoformat
    # handles both "+00:00" and "Z" forms. Defensive on parse failure.
    try:
        oldest_dt = datetime.fromisoformat(oldest_iso.replace("Z", "+00:00"))
    except ValueError:
        return drill_count, None
    resets_dt = oldest_dt + timedelta(days=DRILL_QUOTA_WINDOW_DAYS)
    return drill_count, resets_dt.isoformat()


@app.get("/api/drill/check_quota")
@limiter.limit("20/minute")
async def check_drill_quota(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Tell the client how many drills they have left this rolling 7-day window.
    Pro users (profiles.is_pro=true) have unlimited drills; should_upgrade
    stays false regardless of drill_count.
    """
    user_id = verify_token(authorization)
    try:
        drill_count, quota_resets_at = _drill_quota_state(user_id)
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "check_drill_quota state lookup failed", extra={"user_id": user_id}
        )
        raise HTTPException(status_code=500, detail="Failed to load drill quota")
    is_pro = get_user_pro_status(user_id)
    remaining = max(0, FREE_DRILL_QUOTA - drill_count)
    should_upgrade = (remaining <= 0) and (not is_pro)
    return {
        "drill_count":      drill_count,
        "free_quota":       FREE_DRILL_QUOTA,
        "remaining":        remaining,
        "is_pro":           is_pro,
        "should_upgrade":   should_upgrade,
        "quota_resets_at":  quota_resets_at,
        "window_days":      DRILL_QUOTA_WINDOW_DAYS,
    }


def _resolve_optional_user_id(authorization: Optional[str]) -> str:
    """
    Tracking endpoints accept anonymous and authenticated callers. Returns
    the Supabase user_id when a valid bearer token is supplied; otherwise
    returns the literal "anonymous". Never raises — bad/expired tokens are
    treated as anonymous so that tracking never blocks the user flow.
    """
    if not authorization or not authorization.startswith("Bearer "):
        return "anonymous"
    if supabase_admin is None:
        return "anonymous"
    token = authorization.split(" ", 1)[1].strip()
    if not token:
        return "anonymous"
    try:
        resp = supabase_admin.auth.get_user(token)
    except Exception:
        return "anonymous"
    user = getattr(resp, "user", None)
    user_id = getattr(user, "id", None) if user else None
    return user_id or "anonymous"


@app.post("/api/track/upgrade_page_view")
@limiter.limit("60/minute")
async def track_upgrade_page_view(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Lightweight page-view ping from /upgrade. Logs only — no persistence
    until we wire actual analytics. Auth is optional; anonymous viewers
    are logged with user_id="anonymous".
    """
    user_id = _resolve_optional_user_id(authorization)
    logger.info(
        "[UPGRADE_PAGE_VIEW] user_id=%s timestamp=%s",
        user_id,
        datetime.now(timezone.utc).isoformat(),
    )
    return {"ok": True}


@app.post("/api/track/upgrade_interest")
@limiter.limit("30/minute")
async def track_upgrade_interest(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Email collection ping from /upgrade. Logs only — no persistence yet.
    The endpoint contract is stable so a future Sprint 2D / 2E can swap
    the body for a Stripe / mailing-list integration without churning
    the frontend.
    """
    user_id = _resolve_optional_user_id(authorization)
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}
    email = (body.get("email") or "").strip()
    price_signal = str(body.get("price_signal") or "").strip()
    client_ts = (body.get("timestamp") or "").strip()
    logger.info(
        "[UPGRADE_INTEREST] user_id=%s price_signal=%r email=%r client_ts=%s server_ts=%s",
        user_id,
        price_signal,
        email,
        client_ts,
        datetime.now(timezone.utc).isoformat(),
    )
    return {"ok": True}


@app.patch("/api/practice-records/{record_id}/resolve", status_code=204)
@limiter.limit("60/minute")
async def resolve_practice_record(
    request: Request,
    record_id: str,
    authorization: Optional[str] = Header(None),
):
    """
    Flip a single practice_record to resolved=true.

    - 400 if record_id isn't a valid UUID (defence against garbage paths).
    - 404 if no such record exists.
    - 403 if the record exists but belongs to someone else.
    - 204 on success (empty body).
    """
    user_id = verify_token(authorization)
    try:
        uuid.UUID(record_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid record id")

    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    # Fetch first so we can distinguish 403 (wrong owner) from 404 (no row).
    # A single update with composite filter would collapse both into "0 rows
    # affected", which leaks less but also tells the client less.
    try:
        fetch = (
            supabase_admin.table("practice_records")
            .select("user_id, resolved")
            .eq("id", record_id)
            .limit(1)
            .execute()
        )
    except Exception as exc:
        logger.exception("resolve fetch failed", extra={"record_id": record_id})
        raise HTTPException(status_code=500, detail="Failed to load record") from exc

    rows = fetch.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="Record not found")
    if rows[0].get("user_id") != user_id:
        raise HTTPException(status_code=403, detail="Forbidden")

    # Already resolved is idempotent — still return 204, no point writing again.
    if rows[0].get("resolved") is True:
        return Response(status_code=204)

    try:
        supabase_admin.table("practice_records").update(
            {"resolved": True}
        ).eq("id", record_id).execute()
    except Exception as exc:
        logger.exception("resolve update failed", extra={"record_id": record_id})
        raise HTTPException(status_code=500, detail="Failed to resolve record") from exc

    return Response(status_code=204)


# ─── Vocabulary system ────────────────────────────────────────────────────────
# vocabulary_items is a public-read catalog; user_vocabulary holds the SRS
# state per user; vocabulary_review_logs is the audit trail. RLS handles
# tenancy at the DB layer — these endpoints additionally use service-role
# client with explicit user_id filtering for parity with the rest of the
# /api/* surface.

def _vocab_item_select() -> str:
    """Columns we send to the client for both /items and joined queries."""
    return (
        "id, word, part_of_speech, zh_meaning, difficulty_level, ielts_band_level, "
        "topic, tags, simple_definition_en, common_chunk, speaking_sentence, "
        "common_mistake, better_than, usage_note_zh, created_at"
    )


@app.get("/api/vocabulary/items")
@limiter.limit("30/minute")
async def vocabulary_items_list(
    request: Request,
    topic: Optional[str] = None,
    level: Optional[str] = None,
    search: Optional[str] = None,
):
    """
    Public-ish catalog. Optional filters: topic, level (ielts_band_level),
    search (substring match on word). Auth not required — RLS allows
    anyone to SELECT from vocabulary_items.
    """
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    topic = (topic or "").strip() or None
    level = (level or "").strip() or None
    search = (search or "").strip() or None
    try:
        query = supabase_admin.table("vocabulary_items").select(_vocab_item_select())
        if topic:
            query = query.eq("topic", topic)
        if level:
            query = query.eq("ielts_band_level", level)
        if search:
            # PostgREST ilike: case-insensitive LIKE; wildcards on both sides.
            query = query.ilike("word", f"%{search}%")
        resp = query.order("word", desc=False).execute()
        return {"items": resp.data or []}
    except HTTPException:
        raise
    except Exception:
        logger.exception("vocabulary_items_list failed")
        raise HTTPException(status_code=503, detail="Failed to load vocabulary items")


@app.post("/api/vocabulary/my")
@limiter.limit("30/minute")
async def vocabulary_my_add(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Add a vocabulary_item to the caller's collection. Idempotent: if the
    user already has the item, returns the existing row. Otherwise inserts
    a fresh row with default SRS state (level 0, due immediately).

    Body: { "vocabulary_item_id": "<uuid>", "source": "manual_added" | ... }
    """
    user_id = verify_token(authorization)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    item_id = (body.get("vocabulary_item_id") or "").strip()
    if not item_id:
        raise HTTPException(status_code=400, detail="vocabulary_item_id required")
    try:
        uuid.UUID(item_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid vocabulary_item_id format")
    source = (body.get("source") or "manual_added").strip()
    source_practice_record_id = body.get("source_practice_record_id")

    select_cols = f"*, vocabulary_items({_vocab_item_select()})"

    try:
        # Existence check before insert — supabase-py upsert helpers don't
        # round-trip the joined item cleanly, so we do explicit check+insert.
        existing = (
            supabase_admin.table("user_vocabulary")
            .select(select_cols)
            .eq("user_id", user_id)
            .eq("vocabulary_item_id", item_id)
            .maybe_single()
            .execute()
        )
        if existing and existing.data:
            return existing.data

        # Pro gate: free users may save up to 30 vocabulary items total.
        # Idempotent re-adds (handled by the existing-check above) do NOT
        # consume quota; only genuinely new inserts. Pro skip the check
        # via get_user_pro_status — same fail-safe helper used by
        # /api/history and /api/diagnosis/timeline.
        if not get_user_pro_status(user_id):
            count_resp = (
                supabase_admin.table("user_vocabulary")
                .select("id", count="exact")
                .eq("user_id", user_id)
                .limit(1)
                .execute()
            )
            total = count_resp.count or 0
            if total >= 30:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "vocab_limit_reached",
                        "limit": 30,
                        "message": (
                            "Free users may save up to 30 words. "
                            "Upgrade to Pro for unlimited vocabulary."
                        ),
                    },
                )

        # Verify the catalog item actually exists — fk would catch this on
        # insert, but a 404 is more useful than a Postgres error string.
        item_check = (
            supabase_admin.table("vocabulary_items")
            .select("id")
            .eq("id", item_id)
            .maybe_single()
            .execute()
        )
        if not (item_check and item_check.data):
            raise HTTPException(status_code=404, detail="Vocabulary item not found")

        payload: dict = {
            "user_id": user_id,
            "vocabulary_item_id": item_id,
            "source": source,
        }
        if source_practice_record_id:
            try:
                uuid.UUID(str(source_practice_record_id))
                payload["source_practice_record_id"] = str(source_practice_record_id)
            except ValueError:
                pass  # silently drop bad practice id rather than 400

        inserted = supabase_admin.table("user_vocabulary").insert(payload).execute()
        if not inserted.data:
            raise HTTPException(status_code=500, detail="Failed to add vocabulary item")
        new_id = inserted.data[0]["id"]
        # Re-fetch with the join for a consistent return shape.
        full = (
            supabase_admin.table("user_vocabulary")
            .select(select_cols)
            .eq("id", new_id)
            .single()
            .execute()
        )
        return full.data
    except HTTPException:
        raise
    except Exception:
        logger.exception("vocabulary_my_add failed", extra={"user_id": user_id})
        raise HTTPException(status_code=503, detail="Failed to add vocabulary item")


@app.get("/api/vocabulary/my")
@limiter.limit("30/minute")
async def vocabulary_my_list(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """List all vocabulary items in the caller's collection, with the catalog data joined."""
    user_id = verify_token(authorization)
    select_cols = f"*, vocabulary_items({_vocab_item_select()})"
    try:
        resp = (
            supabase_admin.table("user_vocabulary")
            .select(select_cols)
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .execute()
        )
        return {"items": resp.data or []}
    except Exception:
        logger.exception("vocabulary_my_list failed", extra={"user_id": user_id})
        raise HTTPException(status_code=503, detail="Failed to load vocabulary collection")


@app.get("/api/vocabulary/review/today")
@limiter.limit("30/minute")
async def vocabulary_review_today(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Return up to 10 user_vocabulary rows due for review (next_review_at <= now)."""
    user_id = verify_token(authorization)
    select_cols = f"*, vocabulary_items({_vocab_item_select()})"
    now_iso = datetime.now(timezone.utc).isoformat()
    try:
        resp = (
            supabase_admin.table("user_vocabulary")
            .select(select_cols)
            .eq("user_id", user_id)
            .lte("next_review_at", now_iso)
            .order("next_review_at", desc=False)
            .limit(10)
            .execute()
        )
        return {"items": resp.data or []}
    except Exception:
        logger.exception("vocabulary_review_today failed", extra={"user_id": user_id})
        raise HTTPException(status_code=503, detail="Failed to load review queue")


@app.post("/api/vocabulary/review")
@limiter.limit("60/minute")
async def vocabulary_review_submit(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Submit one flashcard review result. Updates SRS state, increments
    review/correct/wrong counters, sets last_reviewed_at + next_review_at,
    and writes a row to vocabulary_review_logs.

    Body: { "user_vocabulary_id": "<uuid>", "result": "again|hard|good|easy",
            "review_type": "flashcard" }
    """
    user_id = verify_token(authorization)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    uv_id = (body.get("user_vocabulary_id") or "").strip()
    result = (body.get("result") or "").strip()
    review_type = (body.get("review_type") or "flashcard").strip() or "flashcard"

    if not uv_id:
        raise HTTPException(status_code=400, detail="user_vocabulary_id required")
    try:
        uuid.UUID(uv_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user_vocabulary_id format")
    if result not in VOCAB_VALID_RESULTS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid result; must be one of {sorted(VOCAB_VALID_RESULTS)}",
        )

    try:
        existing = (
            supabase_admin.table("user_vocabulary")
            .select("*")
            .eq("id", uv_id)
            .eq("user_id", user_id)
            .maybe_single()
            .execute()
        )
        if not (existing and existing.data):
            raise HTTPException(status_code=404, detail="Vocabulary record not found")

        row = existing.data
        prev_level = int(row.get("srs_level") or 0)
        review_count = int(row.get("review_count") or 0)
        correct_count = int(row.get("correct_count") or 0)
        wrong_count = int(row.get("wrong_count") or 0)

        # SRS transitions per spec.
        if result == "again":
            new_level = max(prev_level - 1, 0)
            days = 1
            wrong_count += 1
        elif result == "hard":
            new_level = prev_level
            days = 2
        elif result == "good":
            new_level = min(prev_level + 1, 5)
            days = VOCAB_SRS_SCHEDULE.get(new_level, 0)
            correct_count += 1
        else:  # easy
            new_level = min(prev_level + 2, 5)
            days = VOCAB_SRS_SCHEDULE.get(new_level, 0)
            correct_count += 1

        now = datetime.now(timezone.utc)
        next_review_at = (now + timedelta(days=days)).isoformat()
        now_iso = now.isoformat()

        update_payload = {
            "srs_level": new_level,
            "review_count": review_count + 1,
            "correct_count": correct_count,
            "wrong_count": wrong_count,
            "last_reviewed_at": now_iso,
            "next_review_at": next_review_at,
            "status": "reviewing" if new_level > 0 else "new",
        }
        updated = (
            supabase_admin.table("user_vocabulary")
            .update(update_payload)
            .eq("id", uv_id)
            .eq("user_id", user_id)
            .execute()
        )
        if not updated.data:
            raise HTTPException(status_code=500, detail="Failed to update review state")

        # Audit log — failure here should not roll back the SRS update,
        # but it should surface in logs so we can spot drift.
        try:
            supabase_admin.table("vocabulary_review_logs").insert({
                "user_id": user_id,
                "user_vocabulary_id": uv_id,
                "vocabulary_item_id": row["vocabulary_item_id"],
                "review_type": review_type,
                "result": result,
                "previous_level": prev_level,
                "new_level": new_level,
            }).execute()
        except Exception:
            logger.exception(
                "[vocab/review] log insert failed (state already saved)",
                extra={"user_id": user_id, "user_vocabulary_id": uv_id},
            )

        return updated.data[0]
    except HTTPException:
        raise
    except Exception:
        logger.exception("vocabulary_review_submit failed", extra={"user_id": user_id})
        raise HTTPException(status_code=503, detail="Failed to submit review")


@app.post("/api/vocabulary/generate")
@limiter.limit("3/minute")
async def vocabulary_generate(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    DB-first vocab fetch with LLM fallback.
      1. Pull caller's top weakness_tag + most-recent topic from practice_records.
      2. Query vocabulary_items by topic.
      3. If < 10 hits: ask Groq for the rest, persist, return combined set.
      4. Returns {items, generated_count, weakness_tag, topic}.

    Fail-open on Groq: returns whatever we have from DB with generated_count=0
    so the user still sees something useful when the LLM is down.
    """
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    # 1. Top weakness_tag + most recent topic from practice_records.
    try:
        recs_resp = (
            supabase_admin.table("practice_records")
            .select("weakness_tag, topic, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )
    except Exception:
        logger.exception("vocabulary_generate practice_records query failed")
        raise HTTPException(status_code=503, detail="Failed to load practice history")

    rows = recs_resp.data or []
    if not rows:
        raise HTTPException(status_code=404, detail="No practice records yet — practice a few times first")

    tag_counts: Counter = Counter(
        (r.get("weakness_tag") or "").strip()
        for r in rows
        if (r.get("weakness_tag") or "").strip()
    )
    weakness_tag = tag_counts.most_common(1)[0][0] if tag_counts else ""
    topic = ""
    for r in rows:
        t = (r.get("topic") or "").strip()
        if t:
            topic = t
            break

    if not topic:
        raise HTTPException(status_code=404, detail="No topic in practice records — practice a few times first")

    # 2. DB lookup by topic. Topic in vocab catalog is lowercase short keys
    # (people/place/work/...); practice topic is human text (Free Time / Hometown).
    # Map common practice topics to their catalog equivalents; unknown topics
    # fall through to lowercased raw value (still tried, may return zero rows
    # → LLM fallback path takes over).
    TOPIC_MAP = {
        'free time':   'hobby',
        'hobbies':     'hobby',
        'hobby':       'hobby',
        'shopping':    'shopping',
        'travel':      'travel',
        'travelling':  'travel',
        'work':        'work',
        'job':         'work',
        'study':       'study',
        'education':   'study',
        'technology':  'technology',
        'tech':        'technology',
        'people':      'people',
        'family':      'people',
        'friends':     'people',
        'place':       'place',
        'hometown':    'place',
        'city':        'place',
        'environment': 'place',
        'experience':  'experience',
        'memory':      'experience',
        'event':       'experience',
        'emotion':     'emotion',
        'feelings':    'emotion',
        'health':      'emotion',
        'nature':      'place',
        'sport':       'hobby',
        'sports':      'hobby',
        'music':       'hobby',
        'food':        'experience',
        'cooking':     'experience',
    }
    raw_topic = topic.lower().strip()
    norm_topic = TOPIC_MAP.get(raw_topic, raw_topic)
    db_items: list[dict] = []
    try:
        db_resp = (
            supabase_admin.table("vocabulary_items")
            .select(_vocab_item_select())
            .eq("topic", norm_topic)
            .limit(10)
            .execute()
        )
        db_items = db_resp.data or []
    except Exception:
        logger.exception("vocabulary_generate db lookup failed")
        # Continue with empty db_items; LLM may still succeed.

    if len(db_items) >= 10:
        return {
            "items": db_items[:10],
            "generated_count": 0,
            "weakness_tag": weakness_tag,
            "topic": topic,
            "mapped_topic": norm_topic,
        }

    needed = 10 - len(db_items)
    existing_words = [it.get("word") for it in db_items if it.get("word")]

    # 3. Groq generation. Fail-open: any error returns DB items unchanged.
    system_prompt = (
        "You are an IELTS Speaking vocabulary expert. "
        "Generate vocabulary items as JSON only. No explanation."
    )
    user_prompt = f"""Generate {needed} IELTS Speaking vocabulary items for topic "{norm_topic}" targeting weakness "{weakness_tag}".

Return a JSON array. Each item must have exactly these fields:
- word (string)
- part_of_speech (string: adjective/verb/noun/adverb/phrase)
- zh_meaning (string, Traditional Chinese)
- difficulty_level (string: B1/B2/C1)
- ielts_band_level (string: 5.5/6.0/6.5/7.0)
- topic (string, use: {norm_topic})
- common_chunk (string, 3-6 words)
- speaking_sentence (string, natural IELTS Speaking sentence, no apostrophes)
- usage_note_zh (string, Traditional Chinese, when/how to use this word)
- tags (array of strings, include "{weakness_tag}")

Rules:
- Words must be genuinely useful for IELTS Speaking Band 5.5-7.0
- speaking_sentence must be a complete natural sentence without apostrophes
- Do not repeat words already in this list: {existing_words}
- Return ONLY the JSON array, no markdown, no explanation
"""

    generated_items: list[dict] = []
    try:
        completion = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.4,
            max_tokens=2000,
        )
        raw = (completion.choices[0].message.content or "").strip()
        # LLM sometimes wraps JSON in ```json ... ```. Strip fences.
        if raw.startswith("```"):
            raw = re.sub(r"^```(?:json)?\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        parsed = json.loads(raw)
        if not isinstance(parsed, list):
            raise ValueError("LLM did not return a JSON array")

        existing_lower = {(w or "").lower() for w in existing_words}
        for item in parsed:
            if not isinstance(item, dict):
                continue
            word = (item.get("word") or "").strip()
            zh = (item.get("zh_meaning") or "").strip()
            chunk = (item.get("common_chunk") or "").strip()
            if not (word and zh and chunk):
                continue
            if word.lower() in existing_lower:
                continue
            existing_lower.add(word.lower())
            tags = item.get("tags")
            if not isinstance(tags, list):
                tags = [weakness_tag] if weakness_tag else []
            elif weakness_tag and weakness_tag not in tags:
                tags = [*tags, weakness_tag]
            row = {
                "word": word,
                "part_of_speech": (item.get("part_of_speech") or "").strip() or None,
                "zh_meaning": zh,
                "difficulty_level": (item.get("difficulty_level") or "").strip() or None,
                "ielts_band_level": (item.get("ielts_band_level") or "").strip() or None,
                "topic": (item.get("topic") or norm_topic).strip().lower() or norm_topic,
                "tags": tags,
                "common_chunk": chunk,
                "speaking_sentence": (item.get("speaking_sentence") or "").strip() or None,
                "usage_note_zh": (item.get("usage_note_zh") or "").strip() or None,
            }
            generated_items.append(row)
            if len(generated_items) >= needed:
                break
    except Exception:
        logger.exception("vocabulary_generate Groq generation failed (fail-open)")
        # Soft-fail: return DB items only.

    inserted_items: list[dict] = []
    if generated_items:
        try:
            ins = supabase_admin.table("vocabulary_items").insert(generated_items).execute()
            inserted_items = ins.data or []
        except Exception:
            logger.exception("vocabulary_generate insert failed (using in-memory items)")
            inserted_items = generated_items  # surface to client even if persist failed

    combined = (db_items + inserted_items)[:10]
    return {
        "items": combined,
        "generated_count": len(inserted_items),
        "weakness_tag": weakness_tag,
        "topic": topic,
        "mapped_topic": norm_topic,
    }


@app.get("/admin/recent")
@limiter.limit("30/minute")
async def admin_recent(
    request: Request,
    authorization: Optional[str] = Header(None),
    mode: Optional[str] = None,
    weakness_tag: Optional[str] = None,
    drill_tag: Optional[str] = None,
    limit: int = 20,
):
    try:
        verify_admin(authorization)
        # Coerce empty-string query params to None — `?mode=` shouldn't
        # become .eq("mode", "") which silently returns nothing.
        mode = (mode or "").strip() or None
        weakness_tag = (weakness_tag or "").strip() or None
        drill_tag = (drill_tag or "").strip() or None
        limit = min(max(int(limit), 1), 100)

        query = (
            supabase_admin.table("practice_records")
            .select(
                "id, user_id, created_at, mode, weakness_tag, drill_tag, "
                "topic, question, user_transcript, coach_response, evidence, drill_score"
            )
            .order("created_at", desc=True)
            .limit(limit)
        )
        if mode:
            query = query.eq("mode", mode)
        if weakness_tag:
            query = query.eq("weakness_tag", weakness_tag)
        if drill_tag:
            query = query.eq("drill_tag", drill_tag)

        response = query.execute()
        rows = response.data or []

        # Email enrichment — kept server-side so the frontend can show
        # email without coordinating across endpoints. Cache per request
        # since the same user often appears on multiple recent rows.
        email_cache: dict[str, str] = {}
        records = []
        for record in rows:
            uid = record.get("user_id") or ""
            if uid and uid not in email_cache:
                try:
                    user_response = supabase_admin.auth.admin.get_user_by_id(uid)
                    user = getattr(user_response, "user", None)
                    email_cache[uid] = getattr(user, "email", None) or ""
                except Exception:
                    email_cache[uid] = ""
            record_with_email = dict(record)
            record_with_email["email"] = email_cache.get(uid, "")
            records.append(record_with_email)

        return {"records": records, "total": len(records)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("admin recent endpoint failed")
        raise HTTPException(status_code=500, detail="Failed to load admin data") from exc


@app.get("/admin/pro_breakdown")
@limiter.limit("30/minute")
async def admin_pro_breakdown(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Returns the four Pro counts: total_pro_effective, paying_users,
    granted_users, both_paid_and_granted. Defaults to all-zeros if the
    RPC returns nothing (e.g. brand-new install).
    """
    try:
        verify_admin(authorization)
        defaults = {
            "total_pro_effective":   0,
            "paying_users":          0,
            "granted_users":         0,
            "both_paid_and_granted": 0,
        }
        try:
            br = supabase_admin.rpc("get_admin_pro_breakdown").execute()
        except Exception:
            logger.exception("[admin/pro_breakdown] rpc failed")
            return defaults
        if not br.data:
            return defaults
        row = br.data[0] if isinstance(br.data, list) else br.data
        return {
            "total_pro_effective":   row.get("total_pro_effective")   or 0,
            "paying_users":          row.get("paying_users")          or 0,
            "granted_users":         row.get("granted_users")         or 0,
            "both_paid_and_granted": row.get("both_paid_and_granted") or 0,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("admin pro_breakdown endpoint failed")
        raise HTTPException(status_code=500, detail="Failed to load pro breakdown") from exc


@app.patch("/admin/user/{user_id}/pro", deprecated=True)
@limiter.limit("30/minute")
async def admin_set_pro_legacy(
    user_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Removed in favor of /admin/user/{user_id}/pro_grant.
    Kept as a 410 stub so any cached client surfaces a clear error
    instead of a confusing 405 Method Not Allowed.
    """
    raise HTTPException(
        status_code=410,
        detail=(
            "This endpoint is deprecated. Use PATCH "
            "/admin/user/{user_id}/pro_grant with body "
            '{"granted": bool, "reason": str}.'
        ),
    )


@app.patch("/admin/user/{user_id}/pro_grant")
@limiter.limit("30/minute")
async def admin_set_pro_grant(
    user_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Toggle profiles.is_pro_grant — the admin-controlled half of Pro
    status. The webhook-controlled is_pro column is NEVER touched here,
    so a manual grant survives a subscription cancellation.

    Body: {
        "granted":    bool,
        "reason":     str|null,
        "expires_at": ISO 8601 datetime str|null   # null = permanent
    }
    Reason is REQUIRED when granted=true (audit trail).
    expires_at is optional; if absent or null on grant, the grant is permanent.
    On revoke, expires_at is always cleared.
    """
    try:
        admin_id = verify_admin(authorization)
        try:
            uuid.UUID(user_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid user_id format")

        try:
            body = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON")

        granted = bool(body.get("granted", False))
        reason = (body.get("reason") or "").strip() or None
        if granted and not reason:
            raise HTTPException(
                status_code=400,
                detail="A reason is required when granting Pro (audit trail)",
            )

        # Parse + validate expires_at. Accept ISO 8601 with or without timezone;
        # store as UTC-aware. Reject garbage so the DB stays clean.
        expires_at_iso: Optional[str] = None
        if granted:
            raw_expires = body.get("expires_at")
            if raw_expires is not None and str(raw_expires).strip() != "":
                try:
                    expires_dt = datetime.fromisoformat(str(raw_expires).replace("Z", "+00:00"))
                    if expires_dt.tzinfo is None:
                        expires_dt = expires_dt.replace(tzinfo=timezone.utc)
                    expires_at_iso = expires_dt.astimezone(timezone.utc).isoformat()
                except (ValueError, TypeError):
                    raise HTTPException(
                        status_code=422,
                        detail="expires_at must be a valid ISO 8601 datetime string or null",
                    )

        # Resolve admin email for the audit column.
        try:
            admin_resp = supabase_admin.auth.admin.get_user_by_id(admin_id)
            admin_user = getattr(admin_resp, "user", None)
            admin_email = (getattr(admin_user, "email", None) or admin_id)
        except Exception:
            admin_email = admin_id

        now_iso = datetime.now(timezone.utc).isoformat()
        update_data = {
            "is_pro_grant":         granted,
            "pro_grant_reason":     reason         if granted else None,
            "pro_grant_at":         now_iso        if granted else None,
            "pro_grant_by":         admin_email    if granted else None,
            "pro_grant_expires_at": expires_at_iso if granted else None,
            "updated_at":           now_iso,
        }

        upd = supabase_admin.table("profiles") \
            .update(update_data) \
            .eq("id", user_id) \
            .execute()
        if not upd.data:
            logger.error("[admin/pro_grant] no rows updated for %s", user_id)
            raise HTTPException(status_code=500, detail="Profile update returned no rows")

        logger.warning(
            "[admin/pro_grant] %s set granted=%s on user %s (reason=%r, expires_at=%r)",
            admin_email, granted, user_id, reason, expires_at_iso,
        )
        return {
            "status": "ok",
            "user_id": user_id,
            "is_pro_grant": granted,
            "pro_grant_reason": reason,
            "pro_grant_expires_at": expires_at_iso,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("admin set_pro_grant endpoint failed")
        raise HTTPException(status_code=500, detail="Failed to update grant status") from exc


@app.delete("/admin/user/{user_id}")
@limiter.limit("10/minute")
async def admin_delete_user(
    user_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Hard-delete a user. Restricted to users with zero practice/drill activity
    to prevent accidental nuking of real users — the UI hides the button, but
    we re-check on the server.
    """
    try:
        admin_id = verify_admin(authorization)
        try:
            uuid.UUID(user_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid user_id format")

        # Server-side safety rail: refuse if the user has any activity.
        practice = (
            supabase_admin.table("practice_records")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        drill = (
            supabase_admin.table("drill_usage")
            .select("id", count="exact")
            .eq("user_id", user_id)
            .limit(1)
            .execute()
        )
        if (practice.count or 0) > 0 or (drill.count or 0) > 0:
            raise HTTPException(
                status_code=409,
                detail="User has activity records — refusing to delete",
            )

        # Delete child rows first, then profile, then auth.users.
        # upgrade_intent.user_id may not exist on every row (anonymous waitlist
        # entries) — the eq filter just no-ops in that case.
        for table in ("drill_usage", "practice_records", "upgrade_intent"):
            try:
                supabase_admin.table(table).delete().eq("user_id", user_id).execute()
            except Exception:
                logger.exception("[admin/delete] failed cleaning %s for %s", table, user_id)

        try:
            supabase_admin.table("profiles").delete().eq("id", user_id).execute()
        except Exception:
            logger.exception("[admin/delete] failed deleting profile for %s", user_id)

        try:
            supabase_admin.auth.admin.delete_user(user_id)
        except Exception as exc:
            logger.exception("[admin/delete] auth.users delete failed for %s", user_id)
            raise HTTPException(status_code=500, detail="Auth user delete failed") from exc

        logger.warning(
            "[admin/delete] %s deleted user %s (no prior activity)", admin_id, user_id
        )
        return {"status": "ok", "user_id": user_id, "deleted": True}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("admin delete_user endpoint failed")
        raise HTTPException(status_code=500, detail="Failed to delete user") from exc


@app.get("/admin/waitlist")
@limiter.limit("30/minute")
async def admin_waitlist(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    try:
        verify_admin(authorization)
        waitlist_resp = (
            supabase_admin.table("upgrade_intent")
            .select("email, reserved_price, reserved_at, source")
            .order("reserved_at", desc=True)
            .execute()
        )
        users_resp = supabase_admin.rpc("get_admin_users_full").execute()

        # Index users by lowercased email — waitlist emails come from
        # external sources (forms/LemonSqueezy) and may differ in case.
        users_by_email: dict[str, dict] = {}
        for u in (users_resp.data or []):
            ue = (u.get("email") or "").strip().lower()
            if ue:
                users_by_email[ue] = u

        enriched: list[dict] = []
        for w in (waitlist_resp.data or []):
            we = (w.get("email") or "").strip().lower()
            u = users_by_email.get(we, {})
            enriched.append({
                **w,
                "practice_count":      u.get("practice_count") or 0,
                "practice_count_7d":   u.get("practice_count_7d") or 0,
                "last_practice":       u.get("last_practice"),
                "main_weakness_tag":   u.get("main_weakness_tag"),
                "conversion_score":    u.get("conversion_score") or 0,
            })

        enriched.sort(key=lambda x: x.get("conversion_score") or 0, reverse=True)
        return {"waitlist": enriched, "total": len(enriched)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("admin waitlist endpoint failed")
        raise HTTPException(status_code=500, detail="Failed to load waitlist") from exc


@app.get("/admin/dashboard")
@limiter.limit("30/minute")
async def admin_dashboard(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Headline metrics for the admin dashboard.

    DAU/WAU/MAU are deduplicated client-side from .data, which PostgREST
    caps at 1000 rows per response. Fine at current scale; revisit with
    a server-side count_distinct RPC when any window crosses ~1000 rows.
    """
    try:
        verify_admin(authorization)

        now = datetime.now(timezone.utc)
        today_start = now.replace(hour=0, minute=0, second=0, microsecond=0).isoformat()
        yesterday_start = (
            now.replace(hour=0, minute=0, second=0, microsecond=0) - timedelta(days=1)
        ).isoformat()
        day3_start = (now - timedelta(days=3)).isoformat()
        day7_start = (now - timedelta(days=7)).isoformat()
        day30_start = (now - timedelta(days=30)).isoformat()
        hour24_start = (now - timedelta(hours=24)).isoformat()

        # Active-user windows
        dau = (
            supabase_admin.table("practice_records")
            .select("user_id", count="exact")
            .gte("created_at", today_start)
            .execute()
        )
        wau = (
            supabase_admin.table("practice_records")
            .select("user_id", count="exact")
            .gte("created_at", day7_start)
            .execute()
        )
        mau = (
            supabase_admin.table("practice_records")
            .select("user_id", count="exact")
            .gte("created_at", day30_start)
            .execute()
        )
        yesterday = (
            supabase_admin.table("practice_records")
            .select("user_id", count="exact")
            .gte("created_at", yesterday_start)
            .lt("created_at", today_start)
            .execute()
        )

        # Practice mode breakdown today
        practice_today = (
            supabase_admin.table("practice_records")
            .select("id, mode", count="exact")
            .gte("created_at", today_start)
            .execute()
        )
        practice_today_rows = practice_today.data or []
        drill_today = sum(1 for r in practice_today_rows if r.get("mode") == "drill")
        normal_today = sum(1 for r in practice_today_rows if r.get("mode") != "drill")

        # Waitlist
        waitlist_total = (
            supabase_admin.table("upgrade_intent")
            .select("id", count="exact")
            .execute()
        )
        waitlist_24h = (
            supabase_admin.table("upgrade_intent")
            .select("id", count="exact")
            .gte("reserved_at", hour24_start)
            .execute()
        )

        # Churn cohort: users with >= 3 practices whose last_practice is older than N days
        all_users = supabase_admin.rpc("get_admin_user_activity").execute()
        users_data = all_users.data or []

        churn_3d = sum(
            1 for u in users_data
            if (u.get("practice_count") or 0) >= 3
            and u.get("last_practice")
            and u["last_practice"] < day3_start
        )
        churn_7d = sum(
            1 for u in users_data
            if (u.get("practice_count") or 0) >= 3
            and u.get("last_practice")
            and u["last_practice"] < day7_start
        )
        never_returned = sum(
            1 for u in users_data if (u.get("practice_count") or 0) == 1
        )

        dau_unique = len({r["user_id"] for r in (dau.data or []) if r.get("user_id")})
        wau_unique = len({r["user_id"] for r in (wau.data or []) if r.get("user_id")})
        mau_unique = len({r["user_id"] for r in (mau.data or []) if r.get("user_id")})
        yesterday_unique = len({r["user_id"] for r in (yesterday.data or []) if r.get("user_id")})

        total_practices_today = practice_today.count or 0
        avg_practice = (
            round(total_practices_today / dau_unique, 1) if dau_unique > 0 else 0
        )

        # Pro breakdown — paying vs granted vs both. Defensive against the
        # RPC not yet existing (during the brief migration window).
        pro = {
            "total_pro_effective":   0,
            "paying_users":          0,
            "granted_users":         0,
            "both_paid_and_granted": 0,
        }
        try:
            br = supabase_admin.rpc("get_admin_pro_breakdown").execute()
            if br.data:
                row = br.data[0] if isinstance(br.data, list) else br.data
                pro.update({
                    "total_pro_effective":   row.get("total_pro_effective") or 0,
                    "paying_users":          row.get("paying_users") or 0,
                    "granted_users":         row.get("granted_users") or 0,
                    "both_paid_and_granted": row.get("both_paid_and_granted") or 0,
                })
        except Exception:
            logger.exception("[admin/dashboard] get_admin_pro_breakdown rpc failed")

        return {
            "dau": dau_unique,
            "dau_yesterday": yesterday_unique,
            "wau": wau_unique,
            "mau": mau_unique,
            "practice_count_today": total_practices_today,
            "avg_practice_per_user": avg_practice,
            "drill_today": drill_today,
            "normal_today": normal_today,
            "waitlist_count": waitlist_total.count or 0,
            "waitlist_24h": waitlist_24h.count or 0,
            "churn_3d": churn_3d,
            "churn_7d": churn_7d,
            "never_returned": never_returned,
            **pro,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("admin dashboard endpoint failed")
        raise HTTPException(status_code=500, detail="Failed to load dashboard") from exc


@app.get("/admin/activity")
@limiter.limit("30/minute")
async def admin_activity(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    try:
        verify_admin(authorization)
        response = supabase_admin.rpc("get_admin_user_activity").execute()
        return {"users": response.data or []}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("admin activity endpoint failed")
        raise HTTPException(status_code=500, detail="Failed to load activity") from exc


@app.get("/admin/users")
@limiter.limit("30/minute")
async def admin_users(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    try:
        verify_admin(authorization)
        result = supabase_admin.rpc("get_admin_users_full").execute()
        rows = result.data or []

        # Merge per-user quality_grade counts. We do this in Python rather
        # than touching the RPC so the migration stays one ALTER TABLE.
        # Single fetch; bucketed by (user_id, grade). Failure here is
        # non-fatal — the admin list still loads without the counts.
        quality_by_user: dict[str, dict[str, int]] = {}
        try:
            qresp = (
                supabase_admin.table("practice_records")
                .select("user_id, quality_grade")
                .execute()
            )
            for r in qresp.data or []:
                uid = r.get("user_id")
                grade = (r.get("quality_grade") or "unknown") or "unknown"
                if not uid:
                    continue
                bucket = quality_by_user.setdefault(uid, {})
                bucket[grade] = bucket.get(grade, 0) + 1
        except Exception:
            logger.exception("admin_users quality aggregation failed")

        for row in rows:
            uid = row.get("user_id")
            counts = quality_by_user.get(uid, {})
            row["quality_counts"] = {
                "valid":   counts.get("valid", 0),
                "partial": counts.get("partial", 0),
                "invalid": counts.get("invalid", 0),
                "unknown": counts.get("unknown", 0),
            }

        return {"users": rows, "total": len(rows)}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("admin users endpoint failed")
        raise HTTPException(status_code=500, detail="Failed to load admin users") from exc


@app.post("/api/admin/reclassify")
@limiter.limit("3/minute")
async def admin_reclassify(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    One-shot batch: classify up to 200 practice_records where
    quality_grade IS NULL, then return how many got updated and how many
    still need work. Admin-only (gated by ADMIN_EMAILS via verify_admin).

    Repeat until {remaining: 0}. Capped at 200 per call so we don't
    blow the request budget on a single hit.
    """
    verify_admin(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        resp = (
            supabase_admin.table("practice_records")
            .select("id, user_transcript, coach_response, weakness_tag")
            .is_("quality_grade", "null")
            .limit(200)
            .execute()
        )
    except Exception as exc:
        logger.exception("admin_reclassify select failed")
        raise HTTPException(status_code=503, detail="Failed to load rows") from exc

    rows = resp.data or []
    updated = 0
    failed = 0
    for row in rows:
        try:
            quality = classify_quality(
                row.get("user_transcript") or "",
                row.get("weakness_tag") or "",
            )
            supabase_admin.table("practice_records").update({
                "quality_grade":  quality["grade"],
                "quality_reason": quality["reason"],
            }).eq("id", row["id"]).execute()
            updated += 1
        except Exception:
            logger.exception("reclassify row failed", extra={"record_id": row.get("id")})
            failed += 1

    # remaining: caller should re-poll until this is 0. We don't do a
    # second count() because it'd race with the next batch insert anyway.
    remaining_estimate = max(0, len(rows) - updated)
    return {
        "updated": updated,
        "failed": failed,
        "batch_size": len(rows),
        "remaining": remaining_estimate,
    }


@app.get("/api/admin/student_brief/{target_user_id}")
@limiter.limit("10/minute")
async def admin_student_brief(
    target_user_id: str,
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    One-paragraph coach brief for a student, generated from their last 30
    valid practice records. Output is Traditional Chinese, formatted as
    現況判斷 / 最關鍵問題 / 給學生的建議 — copy-paste ready for messaging.
    Admin-only.
    """
    verify_admin(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        resp = (
            supabase_admin.table("practice_records")
            .select("created_at, topic, question, user_transcript, weakness_tag, quality_grade")
            .eq("user_id", target_user_id)
            .eq("quality_grade", "valid")
            .order("created_at", desc=False)
            .limit(30)
            .execute()
        )
    except Exception as exc:
        logger.exception("admin_student_brief query failed", extra={"target_user_id": target_user_id})
        raise HTTPException(status_code=503, detail="Failed to load records") from exc

    rows = resp.data or []
    if not rows:
        return {"brief": "No valid practice records found for this student."}

    records_text = "\n\n".join([
        f"[{(r.get('created_at') or '')[:10]}] Topic: {r.get('topic','')} | Q: {r.get('question','')}\n"
        f"Said: {(r.get('user_transcript') or '')[:300]}\n"
        f"Weakness: {r.get('weakness_tag','')}"
        for r in rows
    ])

    try:
        response = anthropic_client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=800,
            system=(
                "You are an expert IELTS Speaking coach reviewing a student's practice history. "
                "Be direct, specific, and actionable. Write in Traditional Chinese. "
                "Format: 1) 現況判斷（2句話） 2) 最關鍵問題（1個，具體證據） 3) 給學生的建議（可直接發給學生的文字）"
            ),
            messages=[{"role": "user", "content": f"Student's valid practice records:\n\n{records_text}"}],
        )
    except Exception as exc:
        logger.exception("admin_student_brief Claude call failed")
        raise HTTPException(status_code=503, detail="LLM call failed") from exc

    return {"brief": response.content[0].text.strip()}


@app.get("/admin/user/{user_id}")
@limiter.limit("30/minute")
async def admin_user_records(
    request: Request,
    user_id: str,
    authorization: Optional[str] = Header(None),
):
    try:
        verify_admin(authorization)
        try:
            uuid.UUID(user_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid user_id format")

        email = None
        try:
            user_response = supabase_admin.auth.admin.get_user_by_id(user_id)
            user = getattr(user_response, "user", None)
            email = getattr(user, "email", None)
        except Exception:
            email = None

        response = (
            supabase_admin.table("practice_records")
            .select(
                "id, mode, topic, question, user_transcript, coach_response, "
                "better_expression, better_expression_zh, next_question, "
                "weakness_tag, memory_snapshot, created_at, "
                "quality_grade, quality_reason"
            )
            .eq("user_id", user_id)
            .order("created_at", desc=False)
            .execute()
        )

        records = []
        for index, record in enumerate(response.data or [], start=1):
            record_with_sequence = dict(record)
            record_with_sequence["sequence"] = index
            records.append(record_with_sequence)

        return {
            "user_id": user_id,
            "email": email,
            "total_records": len(records),
            "records": records,
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("admin user records endpoint failed", extra={"target_user_id": user_id})
        raise HTTPException(status_code=500, detail="Failed to load admin data") from exc


def _generate_user_diagnosis(user_id: str, is_pro: bool = False) -> dict:
    """
    Build a diagnosis for one user from their practice records. Used by
    both POST /admin/user/{id}/diagnosis (admin) and POST /api/diagnosis/me
    (student). Caller is responsible for auth — this trusts the user_id.

    Returns the response dict. Raises HTTPException(503) if Groq fails
    after exhausted retries; HTTPException(400) on invalid user_id.

    is_pro is currently a no-op flag — see TODO below.
    """
    try:
        uuid.UUID(user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user_id format")

    # TODO[pro-gate]: free 限 last 10 practice, pro 全部.
    # When the gate ships, prepend `.limit(10)` for !is_pro.
    response = (
        supabase_admin.table("practice_records")
        .select("question, user_transcript, coach_response, weakness_tag, created_at")
        .eq("user_id", user_id)
        .order("created_at", desc=False)
        .execute()
    )

    records = response.data or []
    now_iso = datetime.now(timezone.utc).isoformat()

    if len(records) == 0:
        return {
            "user_id": user_id,
            "total_records": 0,
            "practice_count": 0,
            "generated_at": now_iso,
            "diagnosis_markdown": (
                "Thou hast yet to record thy first practice. Begin, and "
                "thy patterns shall reveal themselves."
            ),
        }

    # Pull one real sentence from the user's practice to ground the
    # diagnosis. Logic: top weakness tag by count → most recent record
    # matching that tag with a non-empty transcript → first 150 chars.
    # If none qualifies, omit the line entirely (no hallucination).
    example_sentence: Optional[str] = None
    tag_counts: Counter = Counter(
        (r.get("weakness_tag") or "").strip()
        for r in records
        if (r.get("weakness_tag") or "").strip()
    )
    if tag_counts:
        top_tag = tag_counts.most_common(1)[0][0]
        for r in reversed(records):  # records sorted ASC → reverse for newest-first
            if (r.get("weakness_tag") or "").strip() != top_tag:
                continue
            transcript = (r.get("user_transcript") or "").strip()
            if not transcript:
                continue
            example_sentence = transcript[:150]
            break

    system_prompt, user_prompt = build_diagnosis_prompt(
        records, example_sentence=example_sentence
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    # 3 attempts, exponential backoff 1s → 2s. Mirrors run_groq() policy
    # but plain-text completion (not JSON mode), so we own the loop.
    max_attempts = 3
    last_error: Optional[Exception] = None
    diagnosis_text: Optional[str] = None
    for attempt in range(max_attempts):
        try:
            system_content = next(
                (m["content"] for m in messages if m["role"] == "system"), ""
            )
            user_messages = [m for m in messages if m["role"] != "system"]
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=2048,
                system=system_content,
                messages=user_messages,
            )
            diagnosis_text = (response.content[0].text or "").strip()
            if diagnosis_text:
                break
            last_error = ValueError("empty diagnosis from Groq")
        except Exception as exc:
            last_error = exc
            logger.warning(
                "diagnosis groq attempt %s/%s failed: %s",
                attempt + 1, max_attempts, exc,
            )
        if attempt < max_attempts - 1:
            time.sleep(2 ** attempt)

    if not diagnosis_text:
        logger.error(
            "diagnosis groq exhausted retries for %s: %s", user_id, last_error
        )
        raise HTTPException(
            status_code=503,
            detail="Diagnosis service unavailable, please try again",
        )

    # Strip code fences just in case the model ignored the no-fence rule.
    raw = diagnosis_text.strip()
    if raw.startswith("```"):
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

    # Extract JSON object if prose wraps it
    brace_start = raw.find("{")
    brace_end = raw.rfind("}")
    if brace_start > 0 or (brace_start == 0 and brace_end < len(raw) - 1):
        raw = raw[brace_start:brace_end + 1]

    fmt = "raw"
    parsed_data = None
    try:
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError("LLM returned non-dict JSON")
        if not isinstance(parsed.get("weaknesses"), list):
            raise ValueError("missing weaknesses array")
        fmt = "structured"
        parsed_data = parsed
    except (json.JSONDecodeError, ValueError, AttributeError) as exc:
        logger.exception("diagnosis JSON parse failed, raw[:200]=%s", raw[:200])

    return {
        "user_id": user_id,
        "total_records": len(records),
        "practice_count": len(records),
        "generated_at": now_iso,
        "format": fmt,
        "data": parsed_data,
        "raw": raw,
        # diagnosis_markdown kept for the admin tool which still renders
        # the raw payload through marked(); for structured output it'll
        # show the JSON until admin.html learns the new shape.
        "diagnosis_markdown": raw,
    }


@app.post("/admin/user/{user_id}/diagnosis")
@limiter.limit("10/minute")
async def admin_user_diagnosis(
    request: Request,
    user_id: str,
    authorization: Optional[str] = Header(None),
):
    try:
        verify_admin(authorization)
        # Admin sees full history.
        return _generate_user_diagnosis(user_id, is_pro=True)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("admin diagnosis endpoint failed", extra={"target_user_id": user_id})
        raise HTTPException(status_code=500, detail=f"Diagnosis failed: {str(exc)}") from exc


async def _refresh_diagnosis_cache(user_id: str):
    try:
        result = await asyncio.to_thread(_generate_user_diagnosis, user_id, True)
        current_count_resp = supabase_admin.table("practice_records").select("id", count="exact").eq("user_id", user_id).execute()
        current_count = current_count_resp.count or 0
        supabase_admin.table("diagnosis_cache").upsert({
            "user_id": user_id,
            "content": result.get("raw") or "",
            "practice_count": current_count,
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }).execute()
        logger.info(f"diagnosis cache refreshed for {user_id}")
    except Exception as e:
        logger.warning(f"_refresh_diagnosis_cache failed: {e}")


@app.post("/api/diagnosis/me")
@limiter.limit("10/minute")
async def my_diagnosis(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Student-facing diagnosis. user_id ALWAYS comes from JWT — body is
    ignored. Anyone passing target_user_id in body would have it dropped.

    Cached by diagnosis_cache (user_id PK). Cache hit when stored
    practice_count matches the current row count; any new practice
    invalidates by count mismatch. Admin endpoint above stays uncached
    so QA can re-run on demand.
    """
    try:
        user_id = verify_token(authorization)
        if supabase_admin is None:
            raise HTTPException(status_code=503, detail="Database not configured")

        # Always get current count for staleness check
        current_count_resp = supabase_admin.table("practice_records").select("id", count="exact").eq("user_id", user_id).execute()
        current_count = current_count_resp.count or 0

        # Check cache
        cache_resp = supabase_admin.table("diagnosis_cache").select("content,practice_count,updated_at").eq("user_id", user_id).limit(1).execute()
        cache = cache_resp.data[0] if cache_resp.data else None

        if cache and cache.get("content"):
            # Return immediately — stale or fresh
            is_stale = cache.get("practice_count") != current_count
            if is_stale:
                _bg_tasks = getattr(asyncio.get_event_loop(), '_blabby_bg_tasks', set())
                asyncio.get_event_loop()._blabby_bg_tasks = _bg_tasks
                task = asyncio.create_task(_refresh_diagnosis_cache(user_id))
                _bg_tasks.add(task)
                task.add_done_callback(_bg_tasks.discard)
            # Parse content same way as before
            content = cache["content"]
            try:
                start = content.find("{")
                end = content.rfind("}") + 1
                if start != -1 and end > start:
                    parsed = json.loads(content[start:end])
                    return {
                        "format": "structured",
                        "data": parsed,
                        "source": "cache",
                        "cached": True,
                        "diagnosis_markdown": parsed.get("summary", ""),
                    }
            except Exception:
                pass
            return {
                "format": "raw",
                "raw": content,
                "source": "cache",
                "cached": True,
                "diagnosis_markdown": content,
            }

        # No cache — first time, must wait
        result = await asyncio.to_thread(_generate_user_diagnosis, user_id, True)
        # Save to cache
        try:
            supabase_admin.table("diagnosis_cache").upsert({
                "user_id": user_id,
                "content": result.get("raw") or "",
                "practice_count": current_count,
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
        except Exception as e:
            logger.warning(f"diagnosis cache save failed: {e}")
        return result
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("my_diagnosis endpoint failed")
        raise HTTPException(status_code=500, detail=f"Diagnosis failed: {str(exc)}") from exc


# ---------------------------------------------------------------------------
# Part 2 topic cards
# ---------------------------------------------------------------------------

_part2_topics: list[dict] = []


def _load_part2_topics() -> list[dict]:
    global _part2_topics
    if _part2_topics:
        return _part2_topics
    data_path = Path(__file__).parent / "data" / "ielts_part2_topics.json"
    with open(data_path, encoding="utf-8") as f:
        _part2_topics = json.load(f)
    return _part2_topics


@app.get("/part2/topics")
async def get_part2_topic(category: Optional[str] = None):
    topics = _load_part2_topics()
    pool = [t for t in topics if t["category"] == category] if category else topics
    if not pool:
        raise HTTPException(status_code=404, detail="No topics for that category")
    return random.choice(pool)


# ---------------------------------------------------------------------------
# Part 2 evaluation
# ---------------------------------------------------------------------------

def _build_part2_scoring_prompt(topic_title: str, bullets: list[str], notes: Optional[str] = None) -> str:
    bullets_text = "\n".join(f"- {b}" for b in bullets)
    notes_block = ""
    if notes and notes.strip():
        notes_block = (
            "\n\n--- STUDENT'S PREPARATION NOTES ---\n"
            f"{notes.strip()}\n\n"
            "These notes are the candidate's plan. After your strengths and improvements, "
            "you MUST populate the top-level \"notes_analysis\" field in the JSON output with:\n"
            "  - A concrete comparison of what was written in the notes versus what was actually said.\n"
            "  - Point out specific items from the notes that were NOT mentioned in the speech.\n"
            "  - Point out things in the speech that were NOT in the notes (improvised content).\n"
            "  - Be specific: quote actual words from the notes and the transcript.\n"
            "  - Maximum 2 sentences. Direct, diagnostic tone in Traditional Chinese (繁體中文).\n"
            "Scoring rule: Do NOT use notes to inflate band scores — bands are based on the spoken transcript only."
        )
    # Schema fragment — notes_analysis is required when notes_block is non-empty,
    # null when the student submitted no notes (also reflected in the JSON sample below).
    notes_schema_line = (
        '  "notes_analysis": "<Traditional Chinese 繁體中文 — ≤2-sentence diagnostic comparison of notes vs speech>"'
        if notes and notes.strip()
        else '  "notes_analysis": null'
    )
    return f"""You are a trained IELTS Speaking examiner. Score the Part 2 monologue below against all four official criteria. Return ONLY a valid JSON object — no markdown fences, no commentary before or after.

CUE CARD given to candidate:
Topic: {topic_title}
You should say:
{bullets_text}{notes_block}

--- OFFICIAL IELTS BAND DESCRIPTOR ANCHORS ---

Fluency & Coherence
  Band 5: Hesitations and repetitions frequently interrupt flow; limited use of cohesive devices
  Band 6: Some repetition or self-correction; uses basic connectives (and, but, so, because)
  Band 7: Speaks at length without noticeable effort; good use of discourse markers and cohesive devices
  Band 8+: Speaks fluently with only occasional hesitation; ideas are logically sequenced throughout

Lexical Resource
  Band 5: Uses limited vocabulary; errors in word choice noticeably impede communication
  Band 6: Uses adequate vocabulary for familiar topics; some errors in word choice or collocation
  Band 7: Uses less common and idiomatic vocabulary; some awareness of style; occasional inaccuracies
  Band 8+: Uses a wide vocabulary naturally and flexibly; errors are rare and do not impede communication

Grammatical Range & Accuracy
  Band 5: Produces short simple sentences; errors are frequent and sometimes cause difficulty for the listener
  Band 6: Mix of simple and complex structures; errors in complex forms are common; meaning is rarely lost
  Band 7: Uses a variety of complex structures frequently; some errors but they rarely reduce communication
  Band 8+: Wide range of structures; majority of sentences are error-free; errors are minor slips

Pronunciation
  Band 5: Limited control of pronunciation features; L1 accent frequently interferes with intelligibility
  Band 6: Generally intelligible despite accent; some features of pronunciation affect clarity at times
  Band 7: Easy to understand throughout; uses features of connected speech; accent does not impede
  Band 8+: Uses a wide range of pronunciation features; is easy to understand; accent adds character

--- ANTI-INFLATION RULES (mandatory, apply before assigning bands) ---

1. Count filler words/phrases in the transcript: "um", "uh", "er", "like" (as filler), "you know" (as filler). If total > 3 occurrences, Fluency & Coherence MUST NOT exceed 6.5.
2. Count distinct vocabulary items above CEFR B1 level. If fewer than 5 such items appear, Lexical Resource MUST NOT exceed 5.5.
3. Count total words in the transcript. If word count < 100, ALL four bands MUST NOT exceed 6.0.
4. No criterion may receive a band above 7.0 UNLESS its description field contains a direct quote from the transcript as evidence.
5. When uncertain between two adjacent bands, assign the LOWER band (round down).

--- JSON OUTPUT FORMAT ---

Return exactly this structure. Every field is required:
{{
  "band_score": <float — mean of four bands, rounded to nearest 0.5>,
  "criteria": [
    {{
      "name": "Fluency & Coherence",
      "band": <float, 0.5 increment, 4.0–9.0>,
      "description": "<one sentence citing concrete transcript evidence>",
      "improvement": "<one actionable fix — reference specific words or patterns from the transcript>"
    }},
    {{
      "name": "Lexical Resource",
      "band": <float>,
      "description": "<one sentence citing concrete transcript evidence>",
      "improvement": "<one actionable fix>"
    }},
    {{
      "name": "Grammatical Range & Accuracy",
      "band": <float>,
      "description": "<one sentence citing concrete transcript evidence>",
      "improvement": "<one actionable fix>"
    }},
    {{
      "name": "Pronunciation",
      "band": <float — infer from vocabulary precision, self-corrections, and transcription artefacts>,
      "description": "<one sentence citing concrete transcript evidence>",
      "improvement": "<one actionable fix>"
    }}
  ],
  "strengths": ["<Traditional Chinese 繁體中文 — specific observed behaviour with transcript evidence>", "<Traditional Chinese 繁體中文 — specific observed behaviour>"],
  "improvements": ["<Traditional Chinese 繁體中文 — actionable language fix targeting a clear pattern>", "<Traditional Chinese 繁體中文 — actionable language fix>"],
{notes_schema_line}
}}

Computation rule: band_score = round(mean([fc, lr, gra, pron]) * 2) / 2
Tone rule: factual, clinical — describe what was measured, not how the candidate felt.
Language rule: the top-level "strengths", "improvements", and (when non-null) "notes_analysis" MUST be written in Traditional Chinese (繁體中文, zh-TW) — NOT Simplified Chinese, NOT English. English transcript quotes may be embedded inside the Chinese text as evidence (e.g. 引用「I think it's...」). All other fields (band_score, criteria[].name, criteria[].band, criteria[].description, criteria[].improvement) remain in English.

--- BAND SCORE CALIBRATION RULES (non-negotiable, override the computation rule when in conflict) ---

1. Assess honestly first — transcribe what was actually said. Do not assume intent or fill in what the candidate "probably meant".
2. Overall band_score must be within ±1.0 of the LOWEST individual criterion score. If applying this clamp produces a different value than the mean computation, the clamp wins.
3. If the candidate speaks fewer than 30 seconds (estimate from transcript length: ~50 words at natural pace), overall band_score cannot exceed 4.0.
4. If major grammatical errors (tense, subject-verb agreement, basic word order) appear in nearly every sentence, the Grammatical Range & Accuracy criterion cannot exceed 4.5.
5. Do not inflate scores to encourage the candidate — a band 4.0 response must receive 4.0, not 5.5. Encouragement is not your job.
6. All criterion bands and the overall band_score MUST use 0.5 increments only (e.g. 4.0, 4.5, 5.0). Never 4.3, 5.7, or any other fraction.

Accuracy over encouragement. These rules are non-negotiable."""


def _persist_part2(
    user_id: str,
    topic_title: str,
    transcript: str,
    result: Optional[dict],
    notes: Optional[str] = None,
) -> None:
    if supabase_admin is None:
        return
    try:
        supabase_admin.table("practice_records").insert({
            "user_id":          user_id,
            "topic":            topic_title,
            "question":         topic_title,
            "user_transcript":  transcript,
            "coach_response":   json.dumps(result) if result is not None else None,
            "notes":            (notes or "").strip() or None,
            "mode":             "part2",
        }).execute()
    except Exception:
        logger.exception("part2 practice_record insert failed", extra={"user_id": user_id})


@app.post("/api/debug/rec-log")
@limiter.limit("20/minute")
async def debug_rec_log(request: Request):
    """臨時觀測 endpoint:收前端錄音回報(mime/ext/size/ua/status/error)純寫 log。

    iPad 使用者不會開 console,這是遠端確認 iOS 實際錄音格式、驗證
    跨平台修復是否生效的唯一管道。確認修復後可整支移除。
    """
    try:
        payload = await request.json()
    except Exception:
        payload = {"_parse_error": True}
    logger.info("[REC-LOG] %s", json.dumps(payload, ensure_ascii=False)[:1000])
    # 同步寫進 Supabase rec_log:沒有 Render log 存取權的人(含開發者本機工具)才查得到
    # iPad 使用者真實的錄音格式/大小。純觀測,任何失敗(表不存在、Supabase 異常)一律吞掉,
    # 絕不影響回傳 — 確認 iOS 修復後可連同此 endpoint 一起移除。
    if supabase_admin is not None and isinstance(payload, dict):
        try:
            supabase_admin.table("rec_log").insert({
                "part":          payload.get("part"),
                "mime":          payload.get("mime"),
                "recorder_mime": payload.get("recorderMime"),
                "ext":           payload.get("ext"),
                "size":          payload.get("size"),
                "chunk_count":   payload.get("chunkCount"),
                "forced_mp4":    payload.get("forcedMp4"),
                "status":        None if payload.get("status") is None else str(payload.get("status")),
                "error":         payload.get("error"),
                "ua":            payload.get("ua"),
            }).execute()
        except Exception:
            logger.exception("rec_log insert failed (non-fatal)")
    return {"ok": True}


@app.post("/part2/evaluate")
@limiter.limit("5/minute")
async def part2_evaluate(
    request: Request,
    audio: UploadFile = File(...),
    topic_title: str = Form(...),
    bullet_points: str = Form(...),
    notes: str = Form(""),
    authorization: Optional[str] = Header(None),
):
    user_id = verify_token(authorization)

    try:
        bullets: list[str] = json.loads(bullet_points)
    except (json.JSONDecodeError, TypeError):
        raise HTTPException(status_code=422, detail="bullet_points must be a JSON array string")

    notes_clean: Optional[str] = (notes or "").strip() or None

    # Step 1: transcribe via Groq Whisper
    audio_bytes = await audio.read()
    # iOS Safari 錄出 audio/mp4(.m4a),Groq 靠副檔名選容器解碼器,
    # 副檔名必須跟著前端實際格式走,與 /process 的邏輯一致。
    ext = os.path.splitext(audio.filename or "")[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        tmp.write(audio_bytes)
        tmp_path = tmp.name
    try:
        with open(tmp_path, "rb") as f:
            transcript = groq_client.audio.transcriptions.create(
                model="whisper-large-v3-turbo",
                file=f,
                response_format="text",
            )
    except Exception as e:
        logger.error(
            "[/part2/evaluate] Groq transcription failed: user=%s filename=%s suffix=%s bytes=%d status=%s body=%s error=%s",
            user_id, audio.filename, ext, len(audio_bytes),
            getattr(e, "status_code", None), getattr(e, "body", None), e,
        )
        raise HTTPException(status_code=502, detail="Transcription failed, please try again.")
    finally:
        os.unlink(tmp_path)

    transcript = (transcript or "").strip()
    if not transcript:
        logger.warning(
            "[/part2/evaluate] Groq returned empty transcript: user=%s filename=%s suffix=%s bytes=%d",
            user_id, audio.filename, ext, len(audio_bytes),
        )
        raise HTTPException(status_code=422, detail="Could not transcribe audio — no speech detected")
    logger.info(
        "[/part2/evaluate] transcription ok: user=%s suffix=%s bytes=%d chars=%d",
        user_id, ext, len(audio_bytes), len(transcript),
    )

    # Step 2: Claude scoring (prep notes inform diagnosis only — see prompt rules)
    try:
        system_prompt = _build_part2_scoring_prompt(topic_title, bullets, notes_clean)
        result = run_claude([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": transcript},
        ])
    except Exception:
        logger.exception("part2 claude scoring failed", extra={"user_id": user_id})
        _persist_part2(user_id, topic_title, transcript, None, notes_clean)
        return {"transcript": transcript, "notes": notes_clean, "scoring_failed": True}

    # Step 3: persist to practice_records
    _persist_part2(user_id, topic_title, transcript, result, notes_clean)

    return {**result, "transcript": transcript, "notes": notes_clean}


# =============================================================================
# Reading module — Sprint Reading-1
# =============================================================================
#
# Schema lives in supabase/migrations/20260518_reading_module.sql:
#   - public.reading_passages   (id, title, body, difficulty_band, topic,
#                                source, word_count, created_at)
#   - public.reading_questions  (id, passage_id, question_type, question_text,
#                                options jsonb, correct_answer, explanation,
#                                evidence_quote, order_idx, created_at)
#   - public.reading_attempts   (id, user_id, passage_id, started_at,
#                                submitted_at, score, total, band_estimate,
#                                status)
#   - public.reading_answers    (id, attempt_id, question_id, user_answer,
#                                is_correct, answered_at)
#   - profiles.user_band_reading (numeric(3,1) nullable)
#
# Quota: 1 new attempt per UTC calendar day for free users. Enforced inline in
# Python (no daily_usage table exists; Speaking uses the same inline pattern).

FREE_READING_DAILY_QUOTA = 1
READING_TOTAL_QUESTIONS = 9
READING_PASSAGE_MAX_TOKENS = 6000
READING_QUESTIONS_MAX_TOKENS = 4000
READING_LLM_RETRIES = 2  # so 1 initial + 2 retries = 3 total attempts

# IELTS Reading raw-to-band mapping for the 9-question Blabby format. Scales
# proportionally from the standard 40-question Academic Reading band table.
_READING_BAND_BY_SCORE: dict[int, float] = {
    9: 9.0, 8: 8.5, 7: 7.5, 6: 6.5,
    5: 6.0, 4: 5.5, 3: 5.0, 2: 4.5,
    1: 4.0, 0: 4.0,
}


def _reading_band_from_score(score: int) -> float:
    clamped = max(0, min(READING_TOTAL_QUESTIONS, int(score)))
    return _READING_BAND_BY_SCORE[clamped]


def _extract_json_object(raw: str) -> str:
    """
    Extract the first top-level JSON object from an LLM response string.

    Handles three known failure modes of structured-output LLM calls:
      1. Markdown code fence wrappers: ```json ... ``` or ``` ... ```
         (with optional language tag and leading/trailing prose)
      2. Prose preamble/postamble: "Here is the passage:\n{...}\nHope this helps!"
      3. Trailing extra data: "{...}\n\nNote: ..." or stray characters after
         the closing brace, which cause json.JSONDecodeError("Extra data").

    Strategy:
      - Locate the first '{' in the string.
      - Walk forward tracking brace depth, ignoring braces inside string
        literals (handles escaped quotes).
      - Return the substring from the first '{' to its matching '}'.
      - Markdown fence wrappers (```json ... ```) and prose preambles are
        handled transparently — '```' is not '{', so they're skipped.

    Returns the extracted JSON substring (no parsing — caller invokes
    json.loads). If no '{' is found, returns the original input stripped;
    json.loads will then raise a meaningful error.
    """
    s = raw.strip()
    # Locate the first '{' anywhere in the string. The brace-counting loop
    # below transparently handles markdown fence wrappers, prose preambles,
    # and any other non-JSON text surrounding the object — '```' is not '{',
    # so the scanner walks past fences naturally.
    start = s.find("{")
    if start == -1:
        return s

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(s)):
        ch = s[i]
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return s[start:i + 1]
    # Unbalanced — return from first '{' to end, let json.loads raise.
    return s[start:]


def _run_claude_json(system_prompt: str, user_msg: str, max_tokens: int) -> dict:
    """
    Thin Claude wrapper for Reading-module structured-output calls.

    Mirrors run_claude()'s parsing rules (strip ```json fences, require dict),
    but lets the caller set max_tokens — passage generation needs more than
    the 2048-token cap baked into run_claude(). Does NOT loop on JSON errors;
    the calling endpoint owns retry policy because retries here are coupled
    with validator failures, which run_claude() can't see.

    Hardened against three LLM output failure modes via _extract_json_object:
    code-fenced output with prose, prose preamble/postamble, and trailing
    extra data after the closing brace.
    """
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    raw_content = response.content[0].text or ""
    extracted = _extract_json_object(raw_content)
    try:
        parsed = json.loads(extracted)
    except json.JSONDecodeError:
        # Log enough to diagnose without dumping a full passage to logs.
        head = raw_content[:200].replace("\n", "\\n")
        tail = raw_content[-200:].replace("\n", "\\n") if len(raw_content) > 200 else ""
        logger.error(
            "_run_claude_json parse failure | raw_len=%d | head=%r | tail=%r",
            len(raw_content), head, tail,
        )
        raise
    if not isinstance(parsed, dict):
        raise ValueError("Claude returned non-object JSON")
    return parsed


def _extract_vocab_targets_haiku(passage_text: str, difficulty_band: float) -> list[str]:
    """
    Extract 6-10 band-appropriate vocab targets from a passage using
    claude-haiku for cost/latency. Returns lowercased lemmas.

    Non-fatal: any failure returns [] and is logged as a warning. Caller
    must accept that vocab_targets may be empty (user can still read the
    passage but cannot tap difficult words).

    Cost: ~$0.0025/call. Latency: ~1s typical.
    """
    from reading_prompts import build_vocab_targets_prompt_haiku

    prompt = build_vocab_targets_prompt_haiku(passage_text, difficulty_band)
    try:
        response = anthropic_client.messages.create(
            # Dated alias — undated 'claude-haiku-4-5' was producing empty
            # responses in production (JSONDecodeError 'Expecting value at
            # char 0' downstream). Anthropic's session-start hook lists
            # this as the canonical Haiku 4.5 identifier.
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=prompt,
            messages=[{"role": "user", "content": "Output the JSON array now."}],
        )
        raw = (response.content[0].text or "").strip()

        # Defensive: empty response can occur from model misconfiguration,
        # safety filtering, or rate-limit edge cases. Bail before json.loads
        # so we don't surface "Expecting value at char 0" — return [] silently
        # (vocab is non-fatal; passage still renders, just without tappable words).
        if not raw:
            logger.warning(
                "haiku vocab_targets returned empty response; "
                "passage will have no tappable words",
                extra={"passage_chars": len(passage_text)},
            )
            return []

        extracted = _extract_json_object(raw) if raw.startswith("{") else raw
        # The prompt asks for a bare array, but defensive: handle either.
        if extracted.startswith("{"):
            # Unexpected — fall back to find first [ to first ]
            lb = extracted.find("[")
            rb = extracted.rfind("]")
            if lb == -1 or rb == -1 or rb <= lb:
                raise ValueError("Haiku returned object, no array found")
            extracted = extracted[lb:rb + 1]
        parsed = json.loads(extracted)
        if not isinstance(parsed, list):
            raise ValueError("Haiku vocab_targets not a list")
        cleaned = [
            w.strip().lower() for w in parsed
            if isinstance(w, str) and w.strip()
        ]
        # Dedupe preserving order, cap at 10.
        seen = set()
        out = []
        for w in cleaned:
            if w not in seen:
                seen.add(w)
                out.append(w)
            if len(out) >= 10:
                break
        if len(out) < 6:
            logger.warning(
                "haiku vocab_targets returned only %d items (need >=6)",
                len(out),
            )
        return out
    except json.JSONDecodeError:
        # Log raw response head/tail so future regressions can be diagnosed
        # without re-running. Mirrors _run_claude_json's hardening pattern.
        # raw is guaranteed bound here: JSONDecodeError can only originate
        # from json.loads(extracted), which is preceded by raw = ... in
        # the same try block.
        head = raw[:200].replace("\n", "\\n")
        tail = raw[-200:].replace("\n", "\\n") if len(raw) > 200 else ""
        logger.warning(
            "haiku vocab_targets parse failure | raw_len=%d | head=%r | tail=%r",
            len(raw), head, tail,
            exc_info=True,
        )
        return []
    except Exception:
        logger.warning(
            "haiku vocab_targets extraction failed; passage will have no "
            "tappable words",
            exc_info=True,
        )
        return []


def _generate_questions_for_passage(
    passage_text: str,
    difficulty_band: float,
    user_band_reading: float,
) -> dict:
    """
    Generate IELTS reading questions + vocab_targets for a given passage.

    Shared between /reading/passage/generate (blocking, full passage+questions)
    and /reading/questions/generate (new, called after streaming passage).

    Returns the full questions_data dict: {"questions": [...], "vocab_targets": [...]}.
    The caller decides which fields to use:
      - blocking endpoint uses both (questions persisted, vocab_targets
        written onto reading_passages row)
      - new questions endpoint uses only "questions" — vocab_targets for
        streamed passages comes from _extract_vocab_targets_haiku, not here

    Raises HTTPException(500) on retry exhaustion. Caller does NOT catch;
    the framework returns the 500 to the client.

    Retry behaviour mirrors the prior inline loop: READING_LLM_RETRIES + 1
    attempts, validator-driven retry on each failure.
    """
    questions_prompt = build_questions_prompt(
        passage_text, difficulty_band, user_band_reading,
    )
    questions_data: Optional[dict] = None
    last_question_reason: Optional[str] = None

    for attempt in range(READING_LLM_RETRIES + 1):
        try:
            candidate = _run_claude_json(
                questions_prompt,
                passage_text,
                READING_QUESTIONS_MAX_TOKENS,
            )
        except Exception:
            logger.exception(
                "reading questions LLM call failed",
                extra={"attempt": attempt},
            )
            last_question_reason = "llm_call_failed"
            continue
        ok, reason = validate_questions(candidate, passage_text)
        if ok:
            questions_data = candidate
            break
        last_question_reason = reason
        logger.warning(
            "reading questions validator rejected output",
            extra={"attempt": attempt, "reason": reason},
        )

    if questions_data is None:
        logger.error(
            "reading questions generation failed after retries",
            extra={"reason": last_question_reason},
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "questions_generation_failed",
                "reason": last_question_reason or "unknown",
            },
        )

    return questions_data


def _get_user_band_reading(user_id: str) -> Optional[float]:
    """Read profiles.user_band_reading. Returns None on any error."""
    if supabase_admin is None:
        return None
    try:
        resp = (
            supabase_admin.table("profiles")
            .select("user_band_reading")
            .eq("id", user_id)
            .single()
            .execute()
        )
        val = (resp.data or {}).get("user_band_reading")
        return float(val) if val is not None else None
    except Exception:
        logger.exception("_get_user_band_reading failed", extra={"user_id": user_id})
        return None


def _update_user_band_reading(user_id: str, new_estimate: float) -> float:
    """
    Weighted moving average: existing 80%, new 20%. If user_band_reading is
    null, sets directly. Mirrors update_user_band() for the Speaking band.
    Fails safe — caller catches exceptions.
    """
    if supabase_admin is None:
        return new_estimate
    resp = (
        supabase_admin.table("profiles")
        .select("user_band_reading")
        .eq("id", user_id)
        .single()
        .execute()
    )
    current = (resp.data or {}).get("user_band_reading")
    updated = (
        round(float(current) * 0.8 + new_estimate * 0.2, 2)
        if current is not None
        else float(new_estimate)
    )
    supabase_admin.table("profiles").update({
        "user_band_reading":       updated,
        "reading_band_updated_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", user_id).execute()
    return updated


def _reading_daily_count(user_id: str) -> int:
    """
    Count this user's non-abandoned reading_attempts since UTC midnight today.
    Used by both /reading/passage/generate (pre-emptive guard) and
    /reading/attempt/start (authoritative quota gate).
    """
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    today_start = datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).isoformat()
    resp = (
        supabase_admin.table("reading_attempts")
        .select("id")
        .eq("user_id", user_id)
        .gte("started_at", today_start)
        .neq("status", "abandoned")
        .execute()
    )
    return len(resp.data or [])


def _reading_quota_block_response() -> HTTPException:
    """
    Paywall error shape — identical to the Speaking module's drill quota
    response at main.py:1659 so the frontend can use one handler.
    """
    return HTTPException(
        status_code=403,
        detail={
            "error": "quota_exceeded",
            "redirect": "/upgrade",
        },
    )


def _enforce_reading_quota(user_id: str) -> None:
    """Raise the paywall HTTPException if the free quota is consumed."""
    try:
        daily = _reading_daily_count(user_id)
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "reading quota lookup failed; failing closed",
            extra={"user_id": user_id},
        )
        raise HTTPException(status_code=503, detail="Failed to verify reading quota")
    if daily >= FREE_READING_DAILY_QUOTA and not get_user_pro_status(user_id):
        logger.info(
            "[READING_QUOTA_BLOCKED] user_id=%s daily=%d",
            user_id, daily,
        )
        raise _reading_quota_block_response()


@app.get("/reading/quota")
@limiter.limit("30/minute")
async def reading_quota(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Display-only quota probe for the Reading page. Mirrors Speaking's
    /api/drill/check_quota shape so the frontend can render an allowance
    line and gate the "Summon" button without relying on a 403 round-trip.

    The authoritative quota gate is still on /reading/passage/generate and
    /reading/attempt/start — this endpoint must not be trusted by the
    server for enforcement.
    """
    user_id = verify_token(authorization)
    try:
        used_today = _reading_daily_count(user_id)
    except HTTPException:
        raise
    except Exception:
        logger.exception("reading_quota lookup failed", extra={"user_id": user_id})
        raise HTTPException(status_code=500, detail="Failed to load quota")
    is_pro = get_user_pro_status(user_id)
    # For Pro users, limit is null per spec (unlimited).
    limit = None if is_pro else FREE_READING_DAILY_QUOTA
    remaining = None if is_pro else max(0, FREE_READING_DAILY_QUOTA - used_today)
    should_upgrade = (not is_pro) and (used_today >= FREE_READING_DAILY_QUOTA)
    return {
        "used_today":     used_today,
        "limit":          limit,
        "remaining":      remaining,
        "is_pro":         is_pro,
        "should_upgrade": should_upgrade,
    }


@app.post("/reading/attempt/abandon")
@limiter.limit("30/minute")
async def reading_attempt_abandon(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Mark an in_progress attempt as abandoned. Verifies ownership; only
    in_progress attempts can be abandoned (submitted attempts return 409).

    Abandoned attempts do NOT count toward the daily quota — _reading_daily_count
    filters them out. This is the difference between client-side leave (the row
    stays in_progress and counts) and explicit abandon (free).
    """
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}
    attempt_id = body.get("attempt_id")
    if not isinstance(attempt_id, str) or not attempt_id.strip():
        raise HTTPException(status_code=400, detail="attempt_id required")

    try:
        attempt_resp = (
            supabase_admin.table("reading_attempts")
            .select("id, user_id, status")
            .eq("id", attempt_id)
            .single()
            .execute()
        )
    except Exception:
        logger.exception(
            "reading abandon lookup failed",
            extra={"user_id": user_id, "attempt_id": attempt_id},
        )
        raise HTTPException(status_code=404, detail="Attempt not found")
    attempt = attempt_resp.data or {}
    if attempt.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="Attempt not found")
    if attempt.get("status") != "in_progress":
        raise HTTPException(status_code=409, detail="Attempt is not in_progress")

    try:
        supabase_admin.table("reading_attempts").update({
            "status": "abandoned",
        }).eq("id", attempt_id).execute()
    except Exception:
        logger.exception(
            "reading abandon update failed",
            extra={"user_id": user_id, "attempt_id": attempt_id},
        )
        raise HTTPException(status_code=500, detail="Failed to abandon attempt")
    logger.info(
        "[READING_ATTEMPT_ABANDONED] user_id=%s attempt_id=%s",
        user_id, attempt_id,
    )
    return {"status": "abandoned", "attempt_id": attempt_id}


# In-memory LRU cache for /vocab/lookup. Keyed by lowercased word only —
# context-aware definitions would balloon the cache, and the v1 popover
# uses the word alone for display. The cache is process-local and resets on
# every deploy. functools.lru_cache wraps the LLM call; the FastAPI handler
# below normalises input and dispatches.
import functools as _vocab_functools


@_vocab_functools.lru_cache(maxsize=500)
def _vocab_definition_cached(word: str) -> dict:
    """
    Lookup a definition + example via Claude. Caches up to 500 unique words
    per process. Raises on LLM failure — the caller wraps in HTTPException.
    """
    system_prompt = (
        "You are a concise English dictionary. Given a single word, return "
        "STRICT JSON with three keys and nothing else: "
        '{"definition": str, "example": str, "part_of_speech": str}. '
        "Constraints: definition must be at most 25 words and use formal "
        "British English; example is one short natural sentence using the "
        "word; part_of_speech is one of 'noun', 'verb', 'adjective', "
        "'adverb', 'preposition', 'conjunction', 'pronoun', 'determiner', "
        "or 'other'. No preamble, no markdown fences."
    )
    user_msg = f"Word: {word}"
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=300,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    content = (response.content[0].text or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("LLM returned non-object JSON")
    defn = (parsed.get("definition") or "").strip()
    example = (parsed.get("example") or "").strip()
    pos = (parsed.get("part_of_speech") or "").strip().lower()
    if not defn or not example:
        raise ValueError("LLM omitted required fields")
    return {
        "definition":     defn,
        "example":        example,
        "part_of_speech": pos,
    }


@app.post("/vocab/lookup")
@limiter.limit("30/minute")
async def vocab_lookup(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Concise dictionary lookup for a single word, used by the Reading
    passage's word popover. Authenticated to discourage scraping; results
    are cached process-locally (LRU 500). The optional context_sentence
    is accepted for forward-compat but not used in the cache key — v1
    definitions are context-free.

    Body: {"word": str, "context_sentence": str | None}
    Returns: {"word", "definition", "example", "part_of_speech", "cached"}
    """
    verify_token(authorization)
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    raw = body.get("word")
    if not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="word required")
    word = re.sub(r"[^A-Za-z\-']", "", raw).strip().lower()
    if not word or len(word) > 60:
        raise HTTPException(status_code=400, detail="word invalid")

    info = _vocab_definition_cached.cache_info()
    hits_before = info.hits
    try:
        result = _vocab_definition_cached(word)
    except (json.JSONDecodeError, ValueError):
        logger.exception("vocab_lookup LLM output invalid", extra={"word": word})
        raise HTTPException(status_code=502, detail="Definition unavailable")
    except Exception:
        logger.exception("vocab_lookup failed", extra={"word": word})
        raise HTTPException(status_code=503, detail="Definition unavailable")
    cached = _vocab_definition_cached.cache_info().hits > hits_before
    return {"word": word, "cached": cached, **result}


import hashlib as _vocab_hashlib


def _zh_context_hash(context_sentence: Optional[str]) -> str:
    """
    Stable cache key fragment derived from the surrounding context. Only the
    first 80 lowercased characters are hashed — fine-grained context-aware
    translation isn't a v1 goal; this exists so that genuinely different
    contexts don't collide while keeping the cache key space bounded.
    Returns "_" when there is no context.
    """
    if not isinstance(context_sentence, str) or not context_sentence.strip():
        return "_"
    head = context_sentence.strip().lower()[:80]
    return _vocab_hashlib.sha1(head.encode("utf-8")).hexdigest()[:12]


@_vocab_functools.lru_cache(maxsize=500)
def _translate_to_zh(word: str, context_hash: str, english_definition: str) -> dict:
    """
    Translate a single English word to Traditional Chinese via Claude.

    context_hash is the cache-key fragment from _zh_context_hash(); it isn't
    used inside the LLM prompt directly (we pass the raw context separately
    when the endpoint calls this), so the hash exists purely for cache-key
    stability. english_definition is passed through to the prompt to anchor
    the translation when the bare word would be ambiguous.

    Raises on LLM/JSON failure — the calling endpoint wraps in HTTPException.
    """
    # context_hash is captured by the closure for cache-key purposes; we
    # rebuild the human-readable context inside the endpoint, which calls
    # this function with the same hash. Including it as a parameter keeps
    # lru_cache aware of context variation without inflating prompt size.
    _ = context_hash  # silence linters; the value influences the cache key only

    system_prompt = (
        "You are a concise English→Traditional Chinese (繁體中文) "
        "dictionary. Return STRICT JSON with one key and nothing else: "
        '{"zh_meaning": "..."}. Constraints: zh_meaning must be at most '
        "15 Traditional Chinese characters, in dictionary style (a short "
        "noun phrase or one-or-two-word gloss), not a full sentence. Do "
        "not include the English word, parentheses, pinyin, or simplified "
        "characters. No preamble, no markdown fences."
    )
    user_msg = (
        f"Word: {word}\n"
        f"English definition: {english_definition}"
    )
    response = anthropic_client.messages.create(
        model="claude-sonnet-4-6",
        max_tokens=120,
        system=system_prompt,
        messages=[{"role": "user", "content": user_msg}],
    )
    content = (response.content[0].text or "").strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    parsed = json.loads(content)
    if not isinstance(parsed, dict):
        raise ValueError("LLM returned non-object JSON")
    zh = (parsed.get("zh_meaning") or "").strip()
    if not zh:
        raise ValueError("LLM omitted zh_meaning")
    # Clip to the same 15-character limit the prompt enforces, in case the
    # model overshoots.
    return {"zh_meaning": zh[:15]}


@app.post("/vocab/translate_zh")
@limiter.limit("30/minute")
async def vocab_translate_zh(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Translate a single English word to Traditional Chinese. Used by the
    Reading page's two-step save flow so newly-saved words land in the
    vocabulary catalog with a real zh_meaning rather than an empty string.

    Body: {"word": str, "context_sentence": str | None, "english_definition": str | None}
    Returns: {"word", "zh_meaning", "cached"}
    """
    verify_token(authorization)
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    raw = body.get("word")
    if not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="word required")
    word = re.sub(r"[^A-Za-z\-']", "", raw).strip().lower()
    if not word or len(word) > 60:
        raise HTTPException(status_code=400, detail="word invalid")
    context_sentence = body.get("context_sentence")
    english_definition = body.get("english_definition")
    if not isinstance(english_definition, str):
        english_definition = ""
    english_definition = english_definition.strip()[:300]
    context_hash = _zh_context_hash(context_sentence)

    info = _translate_to_zh.cache_info()
    hits_before = info.hits
    try:
        result = _translate_to_zh(word, context_hash, english_definition)
    except (json.JSONDecodeError, ValueError):
        logger.exception("vocab_translate_zh LLM output invalid", extra={"word": word})
        raise HTTPException(status_code=502, detail="Translation unavailable")
    except Exception:
        logger.exception("vocab_translate_zh failed", extra={"word": word})
        raise HTTPException(status_code=503, detail="Translation unavailable")
    cached = _translate_to_zh.cache_info().hits > hits_before
    logger.info(
        "[VOCAB_TRANSLATE_ZH] word=%r cached=%s zh=%r",
        word, cached, result["zh_meaning"],
    )
    return {"word": word, "cached": cached, **result}


@app.post("/api/vocabulary/save_word")
@limiter.limit("30/minute")
async def vocabulary_save_word(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Free-form word save — used by the Reading passage's click-to-save flow.

    The existing /api/vocabulary/my endpoint requires a vocabulary_item_id
    (UUID into the curated catalog). Reading users click arbitrary words that
    may not be in the catalog, so we resolve the word lazily here:
      1. Find vocabulary_items row WHERE word = <lowercased input>
      2. If absent, insert a sparse catalog row (word + optional zh_meaning)
      3. Upsert user_vocabulary linking to that catalog row

    The Reading frontend supplies zh_meaning via /vocab/translate_zh before
    calling this endpoint, so newly-inserted catalog rows now ship with a
    real translation. Speaking-tier callers that omit the field continue to
    work — the fallback is '' (matches the pre-enrichment behaviour).

    Existing catalog rows are NOT overwritten — the catalog is shared, and
    silently mutating another caller's translation is the wrong default.
    Enriching a previously-empty existing row is a deliberate follow-up.

    Body: {"word": str, "source": str, "zh_meaning": str | None}
    """
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    raw = body.get("word")
    if not isinstance(raw, str):
        raise HTTPException(status_code=400, detail="word required")
    # Strip everything that isn't a letter, hyphen, or apostrophe. Lowercase
    # for catalog idempotency (the catalog is treated as case-insensitive).
    word = re.sub(r"[^A-Za-z\-']", "", raw).strip().lower()
    if not word or len(word) > 60:
        raise HTTPException(status_code=400, detail="word invalid")
    source = (body.get("source") or "reading").strip() or "reading"

    # Optional zh_meaning — sent by Reading's two-step save flow. Clipped to
    # 30 characters to keep dictionary cards consistent with the curated
    # catalog's natural length.
    zh_raw = body.get("zh_meaning")
    if zh_raw is None:
        zh_meaning = ""
    elif isinstance(zh_raw, str):
        zh_meaning = zh_raw.strip()[:30]
    else:
        zh_meaning = ""

    try:
        existing_item = (
            supabase_admin.table("vocabulary_items")
            .select("id")
            .eq("word", word)
            .limit(1)
            .execute()
        )
        if existing_item.data:
            item_id = existing_item.data[0]["id"]
        else:
            inserted_item = (
                supabase_admin.table("vocabulary_items")
                .insert({"word": word, "zh_meaning": zh_meaning})
                .execute()
            )
            if not inserted_item.data:
                raise HTTPException(status_code=500, detail="Failed to create catalog entry")
            item_id = inserted_item.data[0]["id"]

        existing_uv = (
            supabase_admin.table("user_vocabulary")
            .select("id")
            .eq("user_id", user_id)
            .eq("vocabulary_item_id", item_id)
            .limit(1)
            .execute()
        )
        if existing_uv.data:
            logger.info(
                "[VOCAB_SAVE_WORD_DUP] user_id=%s word=%r source=%s",
                user_id, word, source,
            )
            return {
                "status": "exists",
                "vocabulary_item_id": item_id,
                "word": word,
            }

        inserted_uv = (
            supabase_admin.table("user_vocabulary")
            .insert({
                "user_id": user_id,
                "vocabulary_item_id": item_id,
                "source": source,
            })
            .execute()
        )
        if not inserted_uv.data:
            raise HTTPException(status_code=500, detail="Failed to save word")
        logger.info(
            "[VOCAB_SAVE_WORD] user_id=%s word=%r source=%s item_id=%s",
            user_id, word, source, item_id,
        )
        return {
            "status": "added",
            "vocabulary_item_id": item_id,
            "word": word,
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "vocabulary_save_word failed",
            extra={"user_id": user_id, "word": word},
        )
        raise HTTPException(status_code=503, detail="Failed to save word")


@app.post("/reading/passage/generate")
@limiter.limit("6/minute")
async def reading_generate_passage(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Generate a new IELTS Reading passage + 9 questions and persist them.
    Does not itself consume quota — /reading/attempt/start is the hard gate —
    but pre-emptively blocks free users who have already used today's slot, so
    we don't spend Claude tokens on a passage they cannot attempt.
    """
    user_id = verify_token(authorization)

    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}

    band_raw = body.get("difficulty_band")
    topic_raw = body.get("topic")
    # Single DB lookup for user_band_reading — used twice in this handler
    # (difficulty fallback when caller omits a band, and vocab_targets
    # calibration before questions generation). Reused via local variable
    # rather than a second SELECT.
    cached_user_band_reading = _get_user_band_reading(user_id)
    if band_raw is None:
        difficulty_band = (
            cached_user_band_reading
            if cached_user_band_reading is not None
            else 6.0
        )
    else:
        try:
            difficulty_band = float(band_raw)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="difficulty_band must be numeric")
        if difficulty_band < 4.0 or difficulty_band > 9.0:
            raise HTTPException(status_code=400, detail="difficulty_band must be 4.0–9.0")
    topic = topic_raw.strip() if isinstance(topic_raw, str) and topic_raw.strip() else None

    _enforce_reading_quota(user_id)

    # --- Passage generation with validator retries -------------------------
    passage_prompt = build_passage_prompt(difficulty_band, topic)
    passage_data: Optional[dict] = None
    last_passage_reason: Optional[str] = None
    for attempt in range(READING_LLM_RETRIES + 1):
        try:
            candidate = _run_claude_json(
                passage_prompt,
                "Generate the passage as specified.",
                READING_PASSAGE_MAX_TOKENS,
            )
        except Exception:
            logger.exception(
                "reading passage LLM call failed",
                extra={"user_id": user_id, "attempt": attempt},
            )
            last_passage_reason = "llm_call_failed"
            continue
        ok, reason = validate_passage(candidate)
        if ok:
            passage_data = candidate
            break
        last_passage_reason = reason
        logger.warning(
            "reading passage validator rejected output",
            extra={"user_id": user_id, "attempt": attempt, "reason": reason},
        )
    if passage_data is None:
        logger.error(
            "reading passage generation failed after retries",
            extra={"user_id": user_id, "reason": last_passage_reason},
        )
        raise HTTPException(
            status_code=500,
            detail={
                "error": "passage_generation_failed",
                "reason": last_passage_reason or "unknown",
            },
        )

    # --- Question generation via shared helper ----------------------------
    # vocab_targets are calibrated against the user's own band, not just the
    # requested passage difficulty. Falls back to difficulty_band (which itself
    # falls back to 6.0 above) for first-time Reading users with no prior band.
    # Reuses the cached_user_band_reading lookup from the top of this handler
    # so we don't issue a second SELECT on profiles per request.
    user_band_for_targets = (
        cached_user_band_reading
        if cached_user_band_reading is not None
        else difficulty_band
    )
    # Helper raises HTTPException(500) on retry exhaustion; no try/except here
    # since the orphan-passage rollback logic below only protects against the
    # questions-insert failure (DB-level), not LLM-generation failure (which
    # happens before any passage is persisted, so no rollback needed).
    questions_data = _generate_questions_for_passage(
        passage_data["body"], difficulty_band, user_band_for_targets,
    )

    # --- Persist ------------------------------------------------------------
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    # vocab_targets is generation-time-fixed: never recomputed when an
    # existing passage is fetched. Stored as JSONB on reading_passages so
    # the frontend can drive the dotted-underline affordance.
    vocab_targets_for_persist = questions_data.get("vocab_targets") or []
    try:
        passage_insert = supabase_admin.table("reading_passages").insert({
            "title":           passage_data["title"],
            "body":            passage_data["body"],
            "difficulty_band": difficulty_band,
            "topic":           passage_data.get("topic") or topic,
            "source":          "ai_generated",
            "word_count":      passage_data["word_count"],
            "vocab_targets":   vocab_targets_for_persist,
            "created_by":      user_id,
        }).execute()
        passage_id = (passage_insert.data or [{}])[0].get("id")
        if not passage_id:
            raise RuntimeError("passage insert returned no id")
    except Exception:
        logger.exception("reading passage insert failed", extra={"user_id": user_id})
        raise HTTPException(status_code=500, detail="Failed to persist passage")

    try:
        question_rows = [
            {
                "passage_id":     passage_id,
                "question_type":  q["question_type"],
                "question_text":  q["question_text"],
                "options":        q.get("options"),
                "correct_answer": q["correct_answer"],
                "explanation":    q["explanation"],
                "evidence_quote": q.get("evidence_quote"),
                "order_idx":      q["order_idx"],
            }
            for q in questions_data["questions"]
        ]
        questions_insert = (
            supabase_admin.table("reading_questions").insert(question_rows).execute()
        )
        inserted_questions = questions_insert.data or []
        if len(inserted_questions) != READING_TOTAL_QUESTIONS:
            raise RuntimeError(
                f"expected {READING_TOTAL_QUESTIONS} question rows, "
                f"got {len(inserted_questions)}"
            )
    except Exception:
        logger.exception("reading questions insert failed", extra={"user_id": user_id})
        # Roll back the orphaned passage so we don't leak rows.
        try:
            supabase_admin.table("reading_passages").delete().eq("id", passage_id).execute()
        except Exception:
            logger.exception(
                "rollback of orphan passage failed",
                extra={"user_id": user_id, "passage_id": passage_id},
            )
        raise HTTPException(status_code=500, detail="Failed to persist questions")

    logger.info(
        "[READING_PASSAGE_GENERATED] user_id=%s passage_id=%s band=%s topic=%r words=%s",
        user_id, passage_id, difficulty_band,
        passage_data.get("topic"), passage_data.get("word_count"),
    )

    client_questions = sorted(
        [
            {
                "id":            row["id"],
                "question_type": row["question_type"],
                "question_text": row["question_text"],
                "options":       row.get("options"),
                "order_idx":     row["order_idx"],
            }
            for row in inserted_questions
        ],
        key=lambda r: r["order_idx"],
    )

    return {
        "passage_id":    passage_id,
        "title":         passage_data["title"],
        "body":          passage_data["body"],
        "vocab_targets": vocab_targets_for_persist,
        "questions":     client_questions,
    }


@app.post("/reading/passage/generate_stream")
@limiter.limit("6/minute")
async def reading_generate_passage_stream(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Streaming variant of /reading/passage/generate.

    Emits SSE events:
      passage_chunk    — incremental text deltas as Claude streams the body
      passage_complete — sent once when streaming ends; carries word count
      metadata         — passage_id, title, topic, band, vocab_targets
      done             — terminal success marker
      error            — terminal failure marker
                         code ∈ {VALIDATION_FAILED, LLM_ERROR, INTERNAL}

    Note: Haiku vocab extraction failure is intentionally non-fatal —
    it logs a warning and emits vocab_targets=[]. No error event is sent.

    Order: passage_chunk* -> passage_complete -> metadata -> done
       or: (any point)    -> error             -> [connection closes]

    Side effects only happen AFTER successful validation:
      - DB insert into reading_passages
      - Quota decrement (via _enforce_reading_quota, called pre-stream
        but reversible? -- NO. See note below.)

    Quota note:
      _enforce_reading_quota raises if user is over quota. We call it
      BEFORE starting the stream so users hit a clean 403 modal instead
      of getting a half-streamed passage that fails at the end. This
      means quota IS consumed on stream start. If the stream fails
      mid-flight (validator reject, LLM error), the user has effectively
      "lost" one quota slot for that day. This matches the blocking
      endpoint's behaviour: any /reading/passage/generate call counts
      against quota regardless of outcome. Future work could refund on
      failure but is out of scope.

    Note: questions generation is NOT covered by this endpoint. Client
    must call the blocking /reading/passage/generate (or a future
    questions-stream endpoint) using the returned passage_id. This
    endpoint streams the passage body only.
    """
    from reading_prompts import build_passage_prompt_plaintext

    user_id = verify_token(authorization)

    # Parse body (same shape as blocking endpoint)
    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}

    # Aligns with blocking endpoint /reading/passage/generate, which uses
    # "difficulty_band" — keeps frontend payload shape identical across
    # streaming and blocking variants.
    band_raw = body.get("difficulty_band")
    topic_raw = body.get("topic")

    cached_user_band_reading = _get_user_band_reading(user_id)
    if band_raw is None:
        difficulty_band = (
            cached_user_band_reading
            if cached_user_band_reading is not None
            else 6.0
        )
    else:
        try:
            difficulty_band = float(band_raw)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="invalid band")
        if difficulty_band < 4.0 or difficulty_band > 9.0:
            raise HTTPException(status_code=400, detail="band out of range")

    topic = (
        topic_raw.strip()
        if isinstance(topic_raw, str) and topic_raw.strip()
        else None
    )

    # Pre-stream quota check (clean 403 if blocked)
    _enforce_reading_quota(user_id)

    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    passage_prompt = build_passage_prompt_plaintext(difficulty_band, topic)

    async def event_stream():
        """SSE generator. Each yield is one SSE frame."""

        def sse(event: str, data: dict) -> str:
            return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"

        collected_chunks: list[str] = []

        try:
            # Stream from Anthropic. The sync client's messages.stream() is a
            # sync context manager yielding text deltas synchronously; safe
            # inside an async generator as long as no await sits in the
            # text_stream loop.
            with anthropic_client.messages.stream(
                model="claude-sonnet-4-6",
                max_tokens=READING_PASSAGE_MAX_TOKENS,
                system=passage_prompt,
                messages=[
                    {"role": "user", "content": "Generate the passage now."}
                ],
            ) as stream:
                for text in stream.text_stream:
                    if not text:
                        continue
                    collected_chunks.append(text)
                    yield sse("passage_chunk", {"text": text})

            passage_text = "".join(collected_chunks).strip()

            if not passage_text:
                yield sse(
                    "error",
                    {"code": "LLM_ERROR", "message": "empty passage from LLM"},
                )
                return

            word_count = len(passage_text.split())
            yield sse("passage_complete", {"words": word_count})

            # Derive title from first sentence (cheap, deterministic).
            # Done before validate so we can pass it into the validator's
            # required {title, body, topic, word_count} shape.
            first_period = passage_text.find(".")
            if first_period != -1 and first_period <= 120:
                # First sentence is reasonable length — use it whole.
                title = passage_text[: first_period + 1].strip()
            else:
                # No period in first 120 chars (rare; usually LLM verbosity).
                # Truncate at word boundary near 80 chars + ellipsis.
                truncated = passage_text[:80].rsplit(" ", 1)[0].strip()
                title = truncated + "…"

            # Validator expects a dict matching the blocking-endpoint schema.
            passage_data = {
                "title":      title,
                "body":       passage_text,
                "topic":      topic or "general",
                "word_count": word_count,
            }
            ok, reason = validate_passage(passage_data)
            if not ok:
                logger.warning(
                    "streaming passage validator rejected | "
                    "band=%s topic=%s reason=%s",
                    difficulty_band, topic, reason,
                )
                yield sse(
                    "error",
                    {
                        "code": "VALIDATION_FAILED",
                        "message": "passage failed quality checks",
                    },
                )
                return

            # Haiku vocab extraction (blocking, non-fatal — returns [] on fail)
            vocab_targets = _extract_vocab_targets_haiku(
                passage_text, difficulty_band
            )

            # DB insert (only after validate passes). Persist with
            # created_by for ownership: the new /reading/questions/generate
            # endpoint relies on this to enforce "you can only generate
            # questions for passages you created"; without it, knowing a
            # passage_id would let any authenticated user burn Sonnet
            # tokens on somebody else's passage.
            try:
                passage_insert = supabase_admin.table("reading_passages").insert({
                    "title":           title,
                    "body":            passage_text,
                    "difficulty_band": difficulty_band,
                    "topic":           topic or "general",
                    "source":          "ai_generated",
                    "word_count":      word_count,
                    "vocab_targets":   vocab_targets,
                    "created_by":      user_id,
                }).execute()
            except Exception:
                logger.exception(
                    "streaming passage insert failed",
                    extra={"user_id": user_id},
                )
                yield sse(
                    "error",
                    {"code": "INTERNAL", "message": "passage persist failed"},
                )
                return

            passage_id = (passage_insert.data or [{}])[0].get("id")
            if not passage_id:
                yield sse(
                    "error",
                    {"code": "INTERNAL", "message": "passage persist returned no id"},
                )
                return

            logger.info(
                "[READING_PASSAGE_STREAMED] user_id=%s passage_id=%s "
                "band=%s topic=%r words=%d vocab=%d",
                user_id, passage_id, difficulty_band, topic,
                word_count, len(vocab_targets),
            )

            yield sse("metadata", {
                "passage_id":      passage_id,
                "title":           title,
                "topic":           topic or "general",
                "difficulty_band": difficulty_band,
                "vocab_targets":   vocab_targets,
                "word_count":      word_count,
            })

            yield sse("done", {})

        except HTTPException:
            raise
        except Exception:
            logger.exception("streaming passage failed unexpectedly")
            yield sse(
                "error",
                {"code": "INTERNAL", "message": "unexpected stream failure"},
            )

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/reading/questions/generate")
@limiter.limit("6/minute")
async def reading_generate_questions(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Generate the 9-question battery for an existing passage.

    Designed to be called by the frontend AFTER /reading/passage/generate_stream
    has emitted its `metadata` event with the passage_id. Lets the client read
    the passage while questions are being generated in parallel.

    Ownership: questions can only be generated for passages where
    reading_passages.created_by == current user. Mismatches return 404 (not
    403) to avoid leaking the existence of other users' passages.

    Idempotent: if the passage already has READING_TOTAL_QUESTIONS rows in
    reading_questions, returns them without re-invoking the LLM. This
    protects against double-click / network-retry causing duplicate insert.

    Quota: NOT enforced here. /reading/passage/generate_stream consumes the
    quota slot when it begins. Calling this endpoint after a successful
    stream is part of the same logical "summon" and must not be gated.
    """
    user_id = verify_token(authorization)

    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}

    passage_id = body.get("passage_id")
    if not isinstance(passage_id, str) or not passage_id.strip():
        raise HTTPException(status_code=400, detail="passage_id required")
    passage_id = passage_id.strip()

    # --- Ownership + existence check (single query) -----------------------
    # Select the passage; 404 covers both "doesn't exist" and "not yours".
    try:
        passage_row = (
            supabase_admin.table("reading_passages")
            .select("id, body, difficulty_band, topic, created_by")
            .eq("id", passage_id)
            .limit(1)
            .execute()
        )
    except Exception:
        logger.exception(
            "reading questions passage lookup failed",
            extra={"user_id": user_id, "passage_id": passage_id},
        )
        raise HTTPException(status_code=500, detail="passage lookup failed")

    rows = passage_row.data or []
    if not rows or rows[0].get("created_by") != user_id:
        # Either passage doesn't exist or belongs to someone else. Same
        # response for both cases — don't leak passage existence.
        raise HTTPException(status_code=404, detail="passage not found")
    passage = rows[0]

    # --- Idempotency check ------------------------------------------------
    # If questions already exist for this passage (e.g. user double-clicked,
    # network retried, or this is a deliberate refresh), return what's there
    # instead of re-generating.
    try:
        existing_questions_resp = (
            supabase_admin.table("reading_questions")
            .select(
                "id, question_type, question_text, options, "
                "correct_answer, explanation, evidence_quote, order_idx"
            )
            .eq("passage_id", passage_id)
            .order("order_idx")
            .execute()
        )
    except Exception:
        logger.exception(
            "reading questions idempotency lookup failed",
            extra={"user_id": user_id, "passage_id": passage_id},
        )
        raise HTTPException(status_code=500, detail="questions lookup failed")

    existing = existing_questions_resp.data or []
    if len(existing) == READING_TOTAL_QUESTIONS:
        logger.info(
            "[READING_QUESTIONS_IDEMPOTENT] user_id=%s passage_id=%s",
            user_id, passage_id,
        )
        client_questions = sorted(existing, key=lambda r: r["order_idx"])
        return {"questions": client_questions}
    if 0 < len(existing) < READING_TOTAL_QUESTIONS:
        # Schema-level invariant violated. Don't try to "fix" by inserting
        # missing rows; the existing partial set may have valid attempt
        # answers attached. Surface loudly.
        logger.error(
            "reading_questions partial state detected",
            extra={
                "user_id": user_id,
                "passage_id": passage_id,
                "existing_count": len(existing),
                "expected": READING_TOTAL_QUESTIONS,
            },
        )
        raise HTTPException(
            status_code=500,
            detail="questions in partial state; contact support",
        )

    # --- Generate ---------------------------------------------------------
    cached_user_band_reading = _get_user_band_reading(user_id)
    user_band_for_targets = (
        cached_user_band_reading
        if cached_user_band_reading is not None
        else float(passage["difficulty_band"])
    )

    questions_data = _generate_questions_for_passage(
        passage["body"],
        float(passage["difficulty_band"]),
        user_band_for_targets,
    )

    # --- Insert -----------------------------------------------------------
    try:
        question_rows = [
            {
                "passage_id":     passage_id,
                "question_type":  q["question_type"],
                "question_text":  q["question_text"],
                "options":        q.get("options"),
                "correct_answer": q["correct_answer"],
                "explanation":    q["explanation"],
                "evidence_quote": q.get("evidence_quote"),
                "order_idx":      q["order_idx"],
            }
            for q in questions_data["questions"]
        ]
        insert_resp = (
            supabase_admin.table("reading_questions")
            .insert(question_rows)
            .execute()
        )
        inserted = insert_resp.data or []
        if len(inserted) != READING_TOTAL_QUESTIONS:
            raise RuntimeError(
                f"expected {READING_TOTAL_QUESTIONS} question rows, "
                f"got {len(inserted)}"
            )
    except Exception:
        logger.exception(
            "reading questions insert failed (questions endpoint)",
            extra={"user_id": user_id, "passage_id": passage_id},
        )
        # We do NOT roll back the passage here. Unlike the blocking endpoint
        # (which created the passage moments earlier), the passage in this
        # endpoint was created by a separate /reading/passage/generate_stream
        # call. Rolling it back would break stream semantics and could erase
        # a passage the user is actively reading. Surface the error and let
        # the user retry the questions call.
        raise HTTPException(status_code=500, detail="Failed to persist questions")

    logger.info(
        "[READING_QUESTIONS_GENERATED] user_id=%s passage_id=%s questions=%d",
        user_id, passage_id, len(inserted),
    )

    client_questions = sorted(inserted, key=lambda r: r["order_idx"])
    return {"questions": client_questions}


@app.post("/reading/attempt/start")
@limiter.limit("12/minute")
async def reading_start_attempt(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Open a reading_attempts row in `in_progress` state. This is the hard
    quota-consuming action. If the user already has an in_progress attempt for
    the same passage, return that one instead of creating a duplicate.
    """
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}
    passage_id = body.get("passage_id")
    if not isinstance(passage_id, str) or not passage_id.strip():
        raise HTTPException(status_code=400, detail="passage_id required")

    try:
        existing = (
            supabase_admin.table("reading_attempts")
            .select("id, started_at")
            .eq("user_id", user_id)
            .eq("passage_id", passage_id)
            .eq("status", "in_progress")
            .limit(1)
            .execute()
        )
    except Exception:
        logger.exception(
            "reading existing-attempt lookup failed",
            extra={"user_id": user_id, "passage_id": passage_id},
        )
        raise HTTPException(status_code=500, detail="Failed to load attempt state")

    if existing.data:
        row = existing.data[0]
        return {"attempt_id": row["id"], "started_at": row["started_at"]}

    _enforce_reading_quota(user_id)

    try:
        insert = supabase_admin.table("reading_attempts").insert({
            "user_id":    user_id,
            "passage_id": passage_id,
            "status":     "in_progress",
        }).execute()
        row = (insert.data or [{}])[0]
        attempt_id = row.get("id")
        started_at = row.get("started_at")
        if not attempt_id:
            raise RuntimeError("attempt insert returned no id")
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "reading attempt insert failed",
            extra={"user_id": user_id, "passage_id": passage_id},
        )
        raise HTTPException(status_code=500, detail="Failed to start attempt")

    logger.info(
        "[READING_ATTEMPT_STARTED] user_id=%s attempt_id=%s passage_id=%s",
        user_id, attempt_id, passage_id,
    )

    return {"attempt_id": attempt_id, "started_at": started_at}


def _reading_is_correct(user_answer: Optional[str], correct_answer: str) -> bool:
    if user_answer is None:
        return False
    return (user_answer or "").strip().lower() == (correct_answer or "").strip().lower()


@app.post("/reading/attempt/submit")
@limiter.limit("12/minute")
async def reading_submit_attempt(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Score a reading attempt, persist answers, update the user's reading band,
    and reveal the correct answers + explanations.
    """
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        body = await request.json()
        if not isinstance(body, dict):
            body = {}
    except Exception:
        body = {}
    attempt_id = body.get("attempt_id")
    answers_payload = body.get("answers")
    if not isinstance(attempt_id, str) or not attempt_id.strip():
        raise HTTPException(status_code=400, detail="attempt_id required")
    if not isinstance(answers_payload, list):
        raise HTTPException(status_code=400, detail="answers must be a list")

    # 1. Verify attempt ownership + status
    try:
        attempt_resp = (
            supabase_admin.table("reading_attempts")
            .select("id, user_id, passage_id, status")
            .eq("id", attempt_id)
            .single()
            .execute()
        )
    except Exception:
        logger.exception(
            "reading attempt lookup failed",
            extra={"user_id": user_id, "attempt_id": attempt_id},
        )
        raise HTTPException(status_code=404, detail="Attempt not found")
    attempt = attempt_resp.data or {}
    if attempt.get("user_id") != user_id:
        raise HTTPException(status_code=404, detail="Attempt not found")
    if attempt.get("status") != "in_progress":
        raise HTTPException(status_code=409, detail="Attempt is not in_progress")
    passage_id = attempt.get("passage_id")

    # 2. Fetch questions for this passage (incl. correct_answer)
    try:
        questions_resp = (
            supabase_admin.table("reading_questions")
            .select("id, correct_answer, explanation, evidence_quote, order_idx")
            .eq("passage_id", passage_id)
            .order("order_idx")
            .execute()
        )
    except Exception:
        logger.exception(
            "reading questions lookup failed",
            extra={"user_id": user_id, "attempt_id": attempt_id},
        )
        raise HTTPException(status_code=500, detail="Failed to load questions")
    questions = questions_resp.data or []
    if not questions:
        raise HTTPException(status_code=500, detail="Passage has no questions")
    questions_by_id = {q["id"]: q for q in questions}

    # 3. Normalise the submitted answers, compute is_correct
    submitted_by_qid: dict[str, Optional[str]] = {}
    for entry in answers_payload:
        if not isinstance(entry, dict):
            continue
        qid = entry.get("question_id")
        ua = entry.get("user_answer")
        if isinstance(qid, str) and qid in questions_by_id:
            submitted_by_qid[qid] = ua if isinstance(ua, str) else None

    answer_rows = []
    score = 0
    results = []
    for q in questions:
        qid = q["id"]
        user_answer = submitted_by_qid.get(qid)
        is_correct = _reading_is_correct(user_answer, q["correct_answer"])
        if is_correct:
            score += 1
        answer_rows.append({
            "attempt_id":  attempt_id,
            "question_id": qid,
            "user_answer": user_answer,
            "is_correct":  is_correct,
        })
        results.append({
            "question_id":    qid,
            "user_answer":    user_answer,
            "correct_answer": q["correct_answer"],
            "is_correct":     is_correct,
            "explanation":    q["explanation"],
            "evidence_quote": q.get("evidence_quote"),
        })

    total = len(questions)
    band_estimate = _reading_band_from_score(score)

    # 4. Persist answers (idempotency: the table has UNIQUE(attempt_id,
    #    question_id) so a retried submit can't double-insert; we treat
    #    re-submit as 409 above, so a clean insert is expected here).
    try:
        supabase_admin.table("reading_answers").insert(answer_rows).execute()
    except Exception:
        logger.exception(
            "reading answers insert failed",
            extra={"user_id": user_id, "attempt_id": attempt_id},
        )
        raise HTTPException(status_code=500, detail="Failed to save answers")

    # 5. Mark the attempt submitted
    try:
        supabase_admin.table("reading_attempts").update({
            "submitted_at":  datetime.now(timezone.utc).isoformat(),
            "status":        "submitted",
            "score":         score,
            "total":         total,
            "band_estimate": band_estimate,
        }).eq("id", attempt_id).execute()
    except Exception:
        logger.exception(
            "reading attempt update failed",
            extra={"user_id": user_id, "attempt_id": attempt_id},
        )
        raise HTTPException(status_code=500, detail="Failed to finalise attempt")

    # 6. Update user_band_reading — non-fatal on failure (mirrors the
    #    Speaking-module posture; band is a derived stat, not core state).
    try:
        _update_user_band_reading(user_id, band_estimate)
    except Exception:
        logger.exception(
            "update_user_band_reading failed (non-fatal)",
            extra={"user_id": user_id, "attempt_id": attempt_id},
        )

    logger.info(
        "[READING_ATTEMPT_SUBMITTED] user_id=%s attempt_id=%s score=%d/%d band=%s",
        user_id, attempt_id, score, total, band_estimate,
    )

    return {
        "score":         score,
        "total":         total,
        "band_estimate": band_estimate,
        "results":       results,
    }


@app.get("/reading/attempt/{attempt_id}")
@limiter.limit("30/minute")
async def reading_get_attempt(
    request: Request,
    attempt_id: str,
    authorization: Optional[str] = Header(None),
):
    """Re-fetch the full reveal for a previously-submitted attempt."""
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        attempt_resp = (
            supabase_admin.table("reading_attempts")
            .select("id, user_id, passage_id, status, score, total, "
                    "band_estimate, started_at, submitted_at")
            .eq("id", attempt_id)
            .single()
            .execute()
        )
    except Exception:
        logger.exception(
            "reading attempt detail lookup failed",
            extra={"user_id": user_id, "attempt_id": attempt_id},
        )
        raise HTTPException(status_code=404, detail="Attempt not found")
    attempt = attempt_resp.data or {}
    if attempt.get("user_id") != user_id or attempt.get("status") != "submitted":
        raise HTTPException(status_code=404, detail="Attempt not found")

    try:
        passage_resp = (
            supabase_admin.table("reading_passages")
            .select("title, body")
            .eq("id", attempt["passage_id"])
            .single()
            .execute()
        )
        questions_resp = (
            supabase_admin.table("reading_questions")
            .select("id, question_type, question_text, options, "
                    "correct_answer, explanation, evidence_quote, order_idx")
            .eq("passage_id", attempt["passage_id"])
            .order("order_idx")
            .execute()
        )
        answers_resp = (
            supabase_admin.table("reading_answers")
            .select("question_id, user_answer, is_correct")
            .eq("attempt_id", attempt_id)
            .execute()
        )
    except Exception:
        logger.exception(
            "reading attempt detail load failed",
            extra={"user_id": user_id, "attempt_id": attempt_id},
        )
        raise HTTPException(status_code=500, detail="Failed to load attempt detail")

    answers_by_qid = {a["question_id"]: a for a in (answers_resp.data or [])}
    results = []
    for q in (questions_resp.data or []):
        a = answers_by_qid.get(q["id"], {})
        results.append({
            "question_id":    q["id"],
            "question_type":  q["question_type"],
            "question_text":  q["question_text"],
            "options":        q.get("options"),
            "order_idx":      q["order_idx"],
            "user_answer":    a.get("user_answer"),
            "correct_answer": q["correct_answer"],
            "is_correct":     a.get("is_correct"),
            "explanation":    q["explanation"],
            "evidence_quote": q.get("evidence_quote"),
        })

    passage = passage_resp.data or {}
    return {
        "attempt_id":    attempt["id"],
        "passage_title": passage.get("title"),
        "passage_body":  passage.get("body"),
        "started_at":    attempt.get("started_at"),
        "submitted_at":  attempt.get("submitted_at"),
        "score":         attempt.get("score"),
        "total":         attempt.get("total"),
        "band_estimate": attempt.get("band_estimate"),
        "results":       results,
    }


@app.get("/reading/history")
@limiter.limit("30/minute")
async def reading_history(
    request: Request,
    limit: int = Query(20, ge=1, le=100),
    authorization: Optional[str] = Header(None),
):
    """Caller's most recent submitted attempts, newest first."""
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        attempts_resp = (
            supabase_admin.table("reading_attempts")
            .select("id, passage_id, score, total, band_estimate, submitted_at")
            .eq("user_id", user_id)
            .eq("status", "submitted")
            .order("submitted_at", desc=True)
            .limit(limit)
            .execute()
        )
    except Exception:
        logger.exception("reading history lookup failed", extra={"user_id": user_id})
        raise HTTPException(status_code=500, detail="Failed to load history")
    attempts = attempts_resp.data or []
    if not attempts:
        return {"attempts": []}

    passage_ids = list({a["passage_id"] for a in attempts if a.get("passage_id")})
    title_by_passage: dict[str, str] = {}
    if passage_ids:
        try:
            passages_resp = (
                supabase_admin.table("reading_passages")
                .select("id, title")
                .in_("id", passage_ids)
                .execute()
            )
            for p in (passages_resp.data or []):
                title_by_passage[p["id"]] = p.get("title", "")
        except Exception:
            logger.exception(
                "reading history passage-title lookup failed",
                extra={"user_id": user_id},
            )
            # Non-fatal — return rows without titles rather than 500.

    return {
        "attempts": [
            {
                "attempt_id":    a["id"],
                "passage_title": title_by_passage.get(a.get("passage_id"), ""),
                "score":         a.get("score"),
                "total":         a.get("total"),
                "band_estimate": a.get("band_estimate"),
                "submitted_at":  a.get("submitted_at"),
            }
            for a in attempts
        ],
    }


@app.get("/debug/sse_test")
async def debug_sse_test():
    """
    SSE spike test endpoint. Pushes one chunk per second for 60 seconds.
    Used to validate Render free tier's long-lived SSE support before Sprint A1.
    DO NOT remove until streaming module is stable in production for 2 weeks.
    """
    async def event_generator():
        start = datetime.now(timezone.utc)
        for i in range(60):
            now = datetime.now(timezone.utc)
            elapsed_ms = int((now - start).total_seconds() * 1000)
            payload = (
                f'{{"tick": {i+1}, "elapsed_ms": {elapsed_ms}, '
                f'"ts": "{now.isoformat()}"}}'
            )
            yield f"data: {payload}\n\n"
            await asyncio.sleep(1.0)
        yield 'data: {"done": true}\n\n'

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable proxy buffering (nginx hint)
        },
    )


def _derive_chart_title(question_prompt: str) -> str:
    """Best-effort chart title from the question's first sentence, stripping the
    'The <chart> below shows ...' lead-in so it reads like a heading, not an
    instruction. Used only when the caller supplies no explicit title."""
    parts = re.split(r"(?<=[.?!])\s+", (question_prompt or "").strip())
    first = parts[0].strip() if parts else ""
    m = re.match(
        r"^the\s+.*?\b(?:show|shows|illustrate|illustrates|depict|depicts|give|gives|compare|compares|present|presents|provide|provides)\b\s+(.*)$",
        first,
        flags=re.IGNORECASE,
    )
    title = (m.group(1).strip() if m else first).rstrip(".").strip()
    if title:
        title = title[0].upper() + title[1:]
    return title or first


def _chart_data_labels(chart_description: str) -> list:
    """Row labels (first column, e.g. years/categories) of a pipe-delimited
    table, excluding the header. Returns [] when the description is not a
    parseable table, so the group-presence check is simply skipped there."""
    lines = [ln for ln in (chart_description or "").splitlines() if ln.strip()]
    pipe_lines = [ln for ln in lines if "|" in ln]
    if len(pipe_lines) < 2:
        return []
    labels = []
    for i, ln in enumerate(pipe_lines):
        if i == 0:
            continue  # header row
        first = ln.split("|")[0].strip()
        if first:
            labels.append(first)
    return labels


def _norm_text(s: str) -> str:
    """Lowercase, collapse whitespace, unify dash variants — tolerant matching
    against text the model rendered into the SVG."""
    s = re.sub(r"[‐-―]", "-", s or "")
    return re.sub(r"\s+", " ", s).strip().lower()


_CHART_DATA_LABEL_MIN_RATIO = 0.5  # gate 3: at least half the data groups must appear in the SVG
_CHART_TITLE_MIN_WORD_COVERAGE = 0.7  # gate 2 (soft): >=70% of title words must appear in the SVG


def _validate_chart_svg(svg: str, chart_title: str, data_labels: list) -> tuple:
    """Return (hard_ok, reason).

    HARD gates reject the SVG (→ retry / text fallback):
      - gate 1: structural completeness (<svg> ... </svg>)
      - gate 3: at least _CHART_DATA_LABEL_MIN_RATIO of the data-group labels
        appear in the SVG. Proportional, not all-or-nothing — grouped/multi-series
        bar charts legitimately omit some X-axis tick labels, so demanding every
        label wrongly killed valid charts. Dropping over half the groups is still
        a real BUG-2 regression and hard-fails.
    SOFT gate is logged but never rejects:
      - gate 2: title word-coverage. Counts how many title words survive into the
        SVG rather than requiring full-string containment (a title legitimately
        split across two <text> lines would always fail containment). Below
        _CHART_TITLE_MIN_WORD_COVERAGE → soft warning only; a mis-derived or
        genuinely rewritten title still never kills the chart.
    Every reason carries actual values so production logs can tell a real model
    error apart from an over-strict validator."""
    # Gate 1 — structure (HARD)
    has_open = "<svg" in (svg or "")
    has_close = "</svg>" in (svg or "")
    if not svg or not has_open or not has_close:
        return False, f"incomplete_svg: has_open={has_open}, has_close={has_close}, len={len(svg or '')}"

    norm_svg = _norm_text(svg)

    # Gate 3 — data groups (HARD, proportional)
    if data_labels:
        missing = [lbl for lbl in data_labels if _norm_text(lbl) not in norm_svg]
        present_ratio = (len(data_labels) - len(missing)) / len(data_labels)
        if present_ratio < _CHART_DATA_LABEL_MIN_RATIO:
            return False, f"insufficient_data_labels: {present_ratio:.0%} present, missing={missing!r} (expected={data_labels!r})"

    # Gate 2 — title word-coverage (SOFT: log only, never reject). Full-string
    # containment would always fail a title split across two <text> lines, so we
    # measure how many title words survive into the SVG instead.
    if chart_title:
        title_words = [w for w in _norm_text(chart_title).split() if len(w) >= 3]
        if title_words:
            missing_words = [w for w in title_words if w not in norm_svg]
            coverage = (len(title_words) - len(missing_words)) / len(title_words)
            if coverage < _CHART_TITLE_MIN_WORD_COVERAGE:
                logger.warning(
                    "chart_svg soft check: %s",
                    f"title_low_coverage: {coverage:.0%} words present, missing={missing_words!r}",
                )

    return True, ""


def generate_chart_svg(
    task1_subtype: str,
    chart_description: str,
    question_prompt: str,
    chart_title: Optional[str] = None,
) -> Optional[str]:
    """Render a coloured SVG chart for a Task 1 question.

    The question text and the chart are produced by two separate model calls, so
    the chart model is handed the verbatim title and the exact data and forbidden
    to invent either. The result is validated (complete tags, title reproduced,
    every data group present); on failure we retry once (max 2 attempts) and then
    return None so the caller falls back to the plain-text description.
    Enhancement only — never raises."""
    title = (chart_title or "").strip() or _derive_chart_title(question_prompt)
    data_labels = _chart_data_labels(chart_description)

    system_prompt = f"""You are an SVG data visualisation engine. Generate a complete, accurate, coloured SVG chart that looks like an official IELTS Academic Writing Task 1 chart.

CANVAS: viewBox="0 0 600 420", white background (#ffffff).
CHART AREA: x=70 to x=560, y=40 to y=340. Origin bottom-left of chart area. (If the title needs two lines, start the chart area at y=56 instead, so it never overlaps the title.)

COLOUR PALETTE (use in order for series 1, 2, 3, 4):
  #1A3550 (navy), #C9A84C (gold), #2D5016 (green), #6B1A1A (wine)

DATA INTEGRITY (mandatory):
- Use ONLY the data supplied in the user message. Never invent, add, omit, or alter any data point or value.
- The chart title MUST be exactly the title supplied in the user message. Do not paraphrase, summarise, or invent a new title. Reproduce it verbatim.
- Before drawing, emit one SVG comment listing every data point you will plot, e.g. <!-- data: 2000=45, 2005=52, 2010=61 -->
- Then draw exactly one visual element per listed data point. The number of bars / line points / pie slices MUST equal the number of data points. Never truncate.
- Every row label (year/category) from the data MUST appear as an axis tick label. No group may be missing.

SELF-CHECK before returning (and fix the SVG if any check fails):
- bar / point / slice count == number of data points
- every year/category label is present as a tick label
- the title text matches the supplied title verbatim

REQUIRED ELEMENTS (all mandatory):
1. White background: <rect width="600" height="420" fill="#ffffff"/>
2. Title: centered x=300, y=24, font-size="15", font-weight="bold", font-family="Georgia, serif", fill="#000000". The title must fit entirely within the canvas width (max x=590). If the title is long, either reduce its font-size (minimum 11) or split it across two lines using two <text> elements (the second line at y+16, i.e. y=40). Never let the title overflow the viewBox or get clipped. When the title wraps to two lines, begin the chart area at y=56 instead of y=40 (shift the Y-axis top, the top gridline, and the value-to-y mapping down accordingly) so the chart never overlaps the title.
3. Y-axis line: x1=70 y1=40 x2=70 y2=340, stroke="#333", stroke-width="1"
4. X-axis line: x1=70 y1=340 x2=560 y2=340, stroke="#333", stroke-width="1"
5. Y-axis label: rotated -90deg, centered on y-axis, font-size="11", font-family="Georgia, serif"
6. X-axis label: centered x=315, y=412, font-size="11", font-family="Georgia, serif"
7. Horizontal grid lines: at least 5 evenly spaced, stroke="#e0e0e0", stroke-width="0.5"
8. Y-axis tick labels: right-aligned at x=65, font-size="10", font-family="Georgia, serif"
9. X-axis tick labels: centered below axis, font-size="10", font-family="Georgia, serif"
10. ALL DATA plotted — never leave chart area empty
11. LEGEND: at y=360-415, horizontal row, line/swatch + label per series, font-size="11", font-family="Georgia, serif"

CHART TYPE RULES:
BAR CHARTS:
  - Filled bars, each series gets its colour from palette, opacity="0.85"
  - Thin black stroke: stroke="#333" stroke-width="0.5"
  - All bars equal width; divide the horizontal chart area evenly by the number of groups, then split each group evenly by the number of series
  - For grouped/multi-series bar charts, the X-axis MUST label every year/category group at least once (place the label under the centre of each group). You may omit minor gridline labels but never omit a data group's category label.
  - Add value labels above each bar, font-size="9"

LINE CHARTS:
  - stroke-width="2", each series uses palette colour
  - Circle markers r=4, filled with series colour
  - Connect all data points with straight lines

PIE CHARTS:
  - Each slice filled with palette colour
  - Percentage label inside each slice, font-size="10", fill="#fff"
  - Category label outside each slice

PROCESS DIAGRAMS:
  - Boxes: fill="#F5F0E8", stroke="#1A3550", stroke-width="1.5", rx="3"
  - Arrows: stroke="#1A3550", stroke-width="1.5", marker-end arrowhead
  - Step text inside boxes, font-size="11"

MAPS (before/after):
  - Simple floor plan style
  - Buildings/areas as rectangles with labels
  - "Before" left side, "After" right side with dividing line
  - Use fill colours from palette at opacity="0.3"

CALCULATION RULES:
- Map data values to y: y = 340 - ((value - min) / (max - min)) * 300
- Scale y-axis with 10% padding above max value
- Compute all positions mathematically from actual data — no placeholder coordinates
- Round all coordinates to integers

COMPACTNESS (the complete SVG must finish within the token budget):
Keep the SVG compact. Do not add decorative elements, gradients, drop shadows, or redundant markup. Round all coordinates to integers. The complete SVG including </svg> must fit well within the token budget — prioritise completeness over visual flourish. A complete simple chart beats a truncated elaborate one.

Return ONLY raw SVG. Start exactly with: <svg viewBox="0 0 600 420" xmlns="http://www.w3.org/2000/svg">
End with: </svg>
No markdown. No explanation. No preamble.

Chart type: {task1_subtype}"""
    user_message = (
        f"Chart type: {task1_subtype}\n"
        f"CHART TITLE (reproduce verbatim, do not change a single character):\n{title}\n\n"
        f"Original question (context only — do not use it as the title):\n{question_prompt}\n\n"
        f"Data — plot every single point listed, drop nothing:\n{chart_description}\n\n"
        "Generate the SVG now."
    )

    max_attempts = 2
    last_reason = "unknown"
    for attempt in range(max_attempts):
        try:
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=8000,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            content = (response.content[0].text or "").strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:svg|html|xml|json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
            content = content.strip()
            hard_ok, reason = _validate_chart_svg(content, title, data_labels)
            if hard_ok:
                logger.info(
                    "generate_chart_svg ok (attempt %d): title=%r, data_points=%d",
                    attempt + 1, title, len(data_labels),
                )
                return content
            last_reason = reason
            logger.warning(
                "generate_chart_svg validation failed (attempt %d/%d): %s",
                attempt + 1, max_attempts, reason,
            )
        except Exception as exc:
            last_reason = f"exception: {exc}"
            logger.warning(
                "generate_chart_svg call failed (attempt %d/%d)",
                attempt + 1, max_attempts, exc_info=True,
            )
    logger.warning(
        "generate_chart_svg giving up after %d attempts; last reason: %s",
        max_attempts, last_reason,
    )
    return None


TASK1_SUBTYPES = ["bar_chart", "line_graph", "pie_chart", "table", "process", "map"]


def _writing_question_prompt(subtype: str) -> str:
    """System prompt to generate one Task 1 question of the given chart subtype.
    Shared by live generation and pregeneration so both stay identical. Injects
    data-realism constraints; for bar/line charts it randomises the structure so
    the bank is not exclusively grouped multi-series charts."""
    readable = subtype.replace("_", " ")

    realism = (
        "DATA REALISM (mandatory):\n"
        "- All data values must be realistic and substantial. Avoid near-zero or zero "
        "values that would render as invisible bars. Each value should generally fall "
        "between 5 and 95 for percentages, unless the topic genuinely demands otherwise.\n"
        "- Values must show a plausible, gradual trend — no implausible jumps from "
        "near-zero to high values between adjacent periods.\n"
        "- Choose a plausible topic: education, environment, economy, health, or technology."
    )

    if subtype == "bar_chart":
        if random.choice(["single_series", "grouped"]) == "single_series":
            structure = (
                "STRUCTURE: a single data series across 4-6 categories or years. "
                "chart_description format: 'Category | Value' with one value column only."
            )
            example = '"Sector | Value\\nEducation | 62\\nHealth | 48\\nTransport | 35\\nHousing | 27"'
        else:
            structure = (
                "STRUCTURE: 2-3 data series across 4-6 categories (at most 3 series, at most 6 categories). "
                "chart_description format: 'Category | Series A | Series B [| Series C]'."
            )
            example = '"Year | Urban | Rural\\n2000 | 45 | 30\\n2005 | 52 | 38\\n2010 | 61 | 44"'
    elif subtype == "line_graph":
        if random.choice(["single_line", "multi_line"]) == "single_line":
            structure = (
                "STRUCTURE: a single line across 4-6 time periods. "
                "chart_description format: 'Period | Value' with one value column only."
            )
            example = '"Year | Value\\n2000 | 22\\n2005 | 34\\n2010 | 41\\n2015 | 58"'
        else:
            structure = (
                "STRUCTURE: 2-3 lines across 4-6 time periods (at most 3 series). "
                "chart_description format: 'Period | Series A | Series B [| Series C]'."
            )
            example = '"Year | Urban | Rural\\n2000 | 22 | 15\\n2005 | 34 | 29\\n2010 | 41 | 38\\n2015 | 58 | 47"'
    else:
        # pie_chart / table / process / map — original plain-text table format.
        structure = (
            "STRUCTURE: a readable plain-text table with real-looking data — a header "
            "row plus 4-6 data rows, columns separated by ' | '."
        )
        example = '"Year | Category A | Category B\\n2000 | 45 | 30\\n2005 | 52 | 28\\n2010 | 61 | 24"'

    return (
        "You are an IELTS examiner. Generate one authentic IELTS Academic Writing "
        f"Task 1 question. Chart type: {readable}.\n"
        f"{realism}\n"
        f"{structure}\n"
        "Return ONLY valid JSON with no preamble and no markdown fences: "
        '{"prompt": "The ' + readable + ' below shows...", "chart_description": ' + example + "}"
    )


def pregenerate_writing_questions(target_per_subtype: int = 5):
    """
    Ensure the writing_questions table has at least target_per_subtype
    pregenerated questions per Task 1 subtype, plus target_per_subtype * 3
    Task 2 questions total.
    Called by APScheduler on startup and nightly.
    Fails silently — never crashes the server.
    """
    if supabase_admin is None:
        logger.warning("pregenerate_writing_questions: supabase_admin not configured")
        return

    # --- Task 1 ---
    for subtype in TASK1_SUBTYPES:
        try:
            existing = (
                supabase_admin.table("writing_questions")
                .select("id", count="exact")
                .eq("task_type", "task1")
                .eq("task1_subtype", subtype)
                .eq("is_pregenerated", True)
                .execute()
            )
            current_count = existing.count or 0
            needed = target_per_subtype - current_count
            if needed <= 0:
                logger.info(f"pregenerate: task1/{subtype} has {current_count}, skipping")
                continue

            logger.info(f"pregenerate: generating {needed} task1/{subtype} questions")
            for _ in range(needed):
                try:
                    # Generate question text
                    system_prompt = _writing_question_prompt(subtype)
                    resp = anthropic_client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=800,
                        system=system_prompt,
                        messages=[{"role": "user", "content": "Generate the question now."}],
                    )
                    content = resp.content[0].text.strip()
                    if content.startswith("```"):
                        content = re.sub(r"^```(?:json)?\s*", "", content)
                        content = re.sub(r"\s*```$", "", content)
                    parsed = json.loads(content)
                    if "prompt" not in parsed:
                        raise ValueError("missing prompt key")

                    # Generate SVG
                    chart_svg = ""
                    if parsed.get("chart_description"):
                        chart_svg = generate_chart_svg(subtype, parsed["chart_description"], parsed["prompt"])

                    ins = supabase_admin.table("writing_questions").insert({
                        "task_type": "task1",
                        "task1_subtype": subtype,
                        "prompt": parsed["prompt"],
                        "chart_description": parsed.get("chart_description"),
                        "chart_svg": chart_svg if chart_svg else None,
                        "is_pregenerated": True,
                        "used_count": 0,
                    }).execute()
                    if not ins.data:
                        logger.error(f"pregenerate: insert failed for task1/{subtype}")
                    else:
                        logger.info(f"pregenerate: created task1/{subtype} id={ins.data[0]['id']}")
                except Exception:
                    logger.exception(f"pregenerate: failed one task1/{subtype} question")
        except Exception:
            logger.exception(f"pregenerate: outer loop failed for task1/{subtype}")

    # --- Task 2 ---
    try:
        existing_t2 = (
            supabase_admin.table("writing_questions")
            .select("id", count="exact")
            .eq("task_type", "task2")
            .eq("is_pregenerated", True)
            .execute()
        )
        current_t2 = existing_t2.count or 0
        needed_t2 = (target_per_subtype * 3) - current_t2
        if needed_t2 > 0:
            logger.info(f"pregenerate: generating {needed_t2} task2 questions")
            for _ in range(needed_t2):
                try:
                    resp = anthropic_client.messages.create(
                        model="claude-haiku-4-5-20251001",
                        max_tokens=800,
                        system="""You are an IELTS examiner. Generate one authentic IELTS Academic Writing Task 2 prompt suitable for Band 5-6 Taiwanese candidates. Choose one type: opinion, discussion, problem_solution, or advantages_disadvantages. Return ONLY valid JSON with no preamble and no markdown fences: {"prompt": "...", "essay_type": "opinion|discussion|problem_solution|advantages_disadvantages"}""",
                        messages=[{"role": "user", "content": "Generate the question now."}],
                    )
                    content = resp.content[0].text.strip()
                    if content.startswith("```"):
                        content = re.sub(r"^```(?:json)?\s*", "", content)
                        content = re.sub(r"\s*```$", "", content)
                    parsed = json.loads(content)
                    if "prompt" not in parsed:
                        raise ValueError("missing prompt key")
                    ins = supabase_admin.table("writing_questions").insert({
                        "task_type": "task2",
                        "prompt": parsed["prompt"],
                        "essay_type": parsed.get("essay_type"),
                        "is_pregenerated": True,
                        "used_count": 0,
                    }).execute()
                    if not ins.data:
                        logger.error("pregenerate: task2 insert failed")
                    else:
                        logger.info(f"pregenerate: created task2 id={ins.data[0]['id']}")
                except Exception:
                    logger.exception("pregenerate: failed one task2 question")
    except Exception:
        logger.exception("pregenerate: task2 outer loop failed")


# ─── Writing Module ───────────────────────────────────────────────────────────
@app.get("/api/writing/question")
@limiter.limit("20/minute")
async def writing_get_question(
    request: Request,
    task_type: str = Query(...),
    task1_subtype: Optional[str] = Query(None),
    authorization: Optional[str] = Header(None),
):
    """Generate one IELTS Writing question via Claude and cache it in
    writing_questions. Task 1 is Pro-only; Task 2 is open to all."""
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    if task_type not in ("task1", "task2"):
        raise HTTPException(status_code=422, detail="task_type must be 'task1' or 'task2'")

    valid_subtypes = ("bar_chart", "line_graph", "pie_chart", "table", "process", "map")
    if task_type == "task1":
        if not task1_subtype or task1_subtype not in valid_subtypes:
            raise HTTPException(
                status_code=422,
                detail="task1_subtype is required for Task 1 and must be one of: bar_chart, line_graph, pie_chart, table, process, map",
            )
        if not await is_user_pro(user_id):
            raise HTTPException(
                status_code=403,
                detail={"error": "pro_required", "message": "Task 1 practice is available to Pro members only."},
            )

    # ── Pool-first: serve a pregenerated question if one exists (no AI call) ──
    try:
        pool_query = (
            supabase_admin.table("writing_questions")
            .select("id, prompt, chart_description, chart_svg, essay_type, task1_subtype, used_count")
            .eq("task_type", task_type)
            .eq("is_pregenerated", True)
        )
        if task_type == "task1":
            pool_query = pool_query.eq("task1_subtype", task1_subtype)
        pool_resp = (
            pool_query.order("used_count", desc=False)
            .order("created_at", desc=False)
            .limit(1)
            .execute()
        )
        if pool_resp.data:
            row = pool_resp.data[0]
            try:
                supabase_admin.table("writing_questions").update(
                    {"used_count": (row.get("used_count") or 0) + 1}
                ).eq("id", row["id"]).execute()
            except Exception:
                logger.warning(
                    "writing pool used_count increment failed",
                    extra={"question_id": row["id"]},
                )
            logger.info(f"writing question served from pool: {row['id']}")
            return {
                "question_id": row["id"],
                "task_type": task_type,
                "task1_subtype": row.get("task1_subtype"),
                "prompt": row["prompt"],
                "chart_description": row.get("chart_description"),
                "chart_svg": row.get("chart_svg"),
                "essay_type": row.get("essay_type"),
            }
    except Exception:
        logger.exception("writing pool lookup failed; falling back to live generation")

    logger.warning(
        f"writing question pool miss: task_type={task_type} subtype={task1_subtype}, generating live"
    )

    if task_type == "task2":
        system_prompt = (
            "You are an IELTS examiner. Generate one authentic IELTS Academic Writing "
            "Task 2 prompt suitable for Band 5-6 Taiwanese candidates. Choose one type: "
            "opinion (To what extent do you agree or disagree?), discussion (Discuss both "
            "views and give your opinion.), problem_solution, or advantages_disadvantages. "
            "Return ONLY valid JSON with no preamble and no markdown fences: "
            '{"prompt": "...", "essay_type": "opinion|discussion|problem_solution|advantages_disadvantages"}'
        )
    else:
        system_prompt = _writing_question_prompt(task1_subtype)

    try:
        response = anthropic_client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=800,
            system=system_prompt,
            messages=[{"role": "user", "content": "Generate the question now."}],
        )
        content = (response.content[0].text or "").strip()
        if content.startswith("```"):
            content = re.sub(r"^```(?:json)?\s*", "", content)
            content = re.sub(r"\s*```$", "", content)
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        logger.exception("writing question generation parse failed", extra={"user_id": user_id})
        raise HTTPException(status_code=500, detail="Question generation failed: invalid response from AI") from exc

    if not isinstance(parsed, dict) or "prompt" not in parsed:
        raise HTTPException(status_code=500, detail="Question generation failed: invalid response from AI")

    chart_svg = ""
    if task_type == "task1" and parsed.get("chart_description"):
        chart_svg = generate_chart_svg(task1_subtype, parsed["chart_description"], parsed["prompt"])

    insert_data = {
        "task_type": task_type,
        "prompt": parsed["prompt"],
        "essay_type": parsed.get("essay_type"),
        "task1_subtype": task1_subtype,
        "chart_description": parsed.get("chart_description"),
        "chart_svg": chart_svg if chart_svg else None,
        "is_pregenerated": False,
        "used_count": 0,
    }
    ins = supabase_admin.table("writing_questions").insert(insert_data).execute()
    if not ins.data:
        raise HTTPException(status_code=500, detail="Question could not be stored")

    return {
        "question_id": ins.data[0]["id"],
        "task_type": task_type,
        "task1_subtype": task1_subtype,
        "prompt": parsed["prompt"],
        "chart_description": parsed.get("chart_description"),
        "chart_svg": chart_svg if chart_svg else None,
        "essay_type": parsed.get("essay_type"),
    }


@app.post("/api/writing/submit")
@limiter.limit("10/minute")
async def writing_submit(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Grade a submitted essay against the four IELTS criteria via Claude and
    persist the result. Free members are capped at 3 submissions per day."""
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    try:
        body = await request.json()
    except Exception as exc:
        raise HTTPException(status_code=422, detail="Request body must be valid JSON") from exc

    question_id = (body.get("question_id") or "").strip()
    if not question_id:
        raise HTTPException(status_code=422, detail="question_id is required")
    essay_text = (body.get("essay_text") or "").strip()
    if not essay_text:
        raise HTTPException(status_code=422, detail="Essay must not be empty")
    task_type = (body.get("task_type") or "").strip()
    if task_type not in ("task1", "task2"):
        raise HTTPException(status_code=422, detail="task_type must be 'task1' or 'task2'")
    is_retry = bool(body.get("is_retry", False))

    word_count = len(essay_text.split())
    if task_type == "task2" and word_count < 50:
        raise HTTPException(
            status_code=422,
            detail="Your essay is too brief to evaluate. Please write at least 50 words.",
        )
    if task_type == "task1" and word_count < 30:
        raise HTTPException(
            status_code=422,
            detail="Your response is too brief to evaluate. Please write at least 30 words.",
        )

    q_resp = (
        supabase_admin.table("writing_questions")
        .select("prompt, task_type, chart_description")
        .eq("id", question_id)
        .limit(1)
        .execute()
    )
    if not q_resp.data:
        raise HTTPException(status_code=404, detail="The specified question could not be found")
    question_row = q_resp.data[0]
    question_prompt = question_row["prompt"]

    if task_type == "task1" and not await is_user_pro(user_id):
        raise HTTPException(
            status_code=403,
            detail={"error": "pro_required", "message": "Task 1 practice is available to Pro members only."},
        )

    if not await is_user_pro(user_id):
        try:
            today_start = datetime.now(timezone.utc).replace(
                hour=0, minute=0, second=0, microsecond=0
            ).isoformat()
            quota_resp = (
                supabase_admin.table("writing_submissions")
                .select("id", count="exact")
                .eq("user_id", user_id)
                .gte("submitted_at", today_start)
                .limit(1)
                .execute()
            )
            daily_count = quota_resp.count or 0
            if daily_count >= 3:
                raise HTTPException(
                    status_code=403,
                    detail={
                        "error": "daily_quota_reached",
                        "limit": 3,
                        "message": "Free members may submit 3 writing tasks per day. Upgrade to Pro for unlimited practice.",
                    },
                )
        except HTTPException:
            raise
        except Exception:
            logger.warning(
                "writing daily quota check failed; failing open",
                extra={"user_id": user_id},
            )

    grading_system = (
        "You are a strict but constructive IELTS Writing examiner specialising in "
        "Band 5-6 Taiwanese learners. Grade the submitted essay against official IELTS "
        "band descriptors. Band range for this cohort: 4.0 to 7.0. Return ONLY valid "
        "JSON with no preamble and no markdown fences:\n"
        '{"band_ta": <float>, "feedback_ta": "<2-3 sentences diagnosing Task Achievement/Response>", '
        '"fix_ta": "<single most important fix for TA>", "band_cc": <float>, '
        '"feedback_cc": "<Coherence & Cohesion diagnosis>", "fix_cc": "<single fix>", '
        '"band_lr": <float>, "feedback_lr": "<Lexical Resource diagnosis>", "fix_lr": "<single fix>", '
        '"band_gra": <float>, "feedback_gra": "<Grammatical Range & Accuracy diagnosis>", "fix_gra": "<single fix>", '
        '"band_overall": <float>, "priority_fix": "<the single most impactful improvement this student must make — max 2 sentences>"}'
    )
    grading_user = f"Task type: {task_type}\nQuestion: {question_prompt}\n\nStudent essay:\n{essay_text}"

    grading = None
    grading_error = None
    for _attempt in range(2):
        try:
            response = anthropic_client.messages.create(
                model="claude-sonnet-4-6",
                max_tokens=1500,
                system=grading_system,
                messages=[{"role": "user", "content": grading_user}],
            )
            content = (response.content[0].text or "").strip()
            if content.startswith("```"):
                content = re.sub(r"^```(?:json)?\s*", "", content)
                content = re.sub(r"\s*```$", "", content)
            grading = json.loads(content)
            break
        except json.JSONDecodeError as exc:
            grading_error = exc
            continue

    if grading is None:
        logger.error(
            "writing grading parse failed after retry: %s",
            grading_error,
            extra={"user_id": user_id},
        )
        raise HTTPException(
            status_code=500,
            detail="Grading could not be completed. Pray attempt submission once more.",
        )

    if not isinstance(grading, dict) or "band_overall" not in grading or "priority_fix" not in grading:
        raise HTTPException(
            status_code=500,
            detail="Grading could not be completed. Pray attempt submission once more.",
        )

    sub_data = {
        "user_id": user_id,
        "question_id": question_id,
        "task_type": task_type,
        "essay_text": essay_text,
        "word_count": word_count,
        "is_retry": is_retry,
        "feedback_ta": grading.get("feedback_ta"),
        "feedback_cc": grading.get("feedback_cc"),
        "feedback_lr": grading.get("feedback_lr"),
        "feedback_gra": grading.get("feedback_gra"),
        "fix_ta": grading.get("fix_ta"),
        "fix_cc": grading.get("fix_cc"),
        "fix_lr": grading.get("fix_lr"),
        "fix_gra": grading.get("fix_gra"),
        "band_ta": grading.get("band_ta"),
        "band_cc": grading.get("band_cc"),
        "band_lr": grading.get("band_lr"),
        "band_gra": grading.get("band_gra"),
        "band_overall": grading.get("band_overall"),
        "priority_fix": grading.get("priority_fix"),
    }
    ins = supabase_admin.table("writing_submissions").insert(sub_data).execute()
    if not ins.data:
        raise HTTPException(status_code=500, detail="Submission could not be recorded")

    return {
        "submission_id": ins.data[0]["id"],
        "word_count": word_count,
        "band_overall": grading.get("band_overall"),
        "priority_fix": grading.get("priority_fix"),
        "criteria": {
            "task_achievement": {"band": grading.get("band_ta"), "feedback": grading.get("feedback_ta"), "fix": grading.get("fix_ta")},
            "coherence_cohesion": {"band": grading.get("band_cc"), "feedback": grading.get("feedback_cc"), "fix": grading.get("fix_cc")},
            "lexical_resource": {"band": grading.get("band_lr"), "feedback": grading.get("feedback_lr"), "fix": grading.get("fix_lr")},
            "grammatical_range": {"band": grading.get("band_gra"), "feedback": grading.get("feedback_gra"), "fix": grading.get("fix_gra")},
        },
    }


@app.get("/api/writing/history")
@limiter.limit("20/minute")
async def writing_history(
    request: Request,
    limit: int = Query(10, ge=1, le=20),
    authorization: Optional[str] = Header(None),
):
    """Recent writing submissions for the caller. Free members see at most 5."""
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    is_pro = await is_user_pro(user_id)
    effective_limit = limit if is_pro else min(limit, 5)

    try:
        resp = (
            supabase_admin.table("writing_submissions")
            .select("id, task_type, band_overall, priority_fix, submitted_at, word_count, question_id")
            .eq("user_id", user_id)
            .order("submitted_at", desc=True)
            .limit(effective_limit)
            .execute()
        )
    except Exception as exc:
        logger.exception("writing history query failed", extra={"user_id": user_id})
        raise HTTPException(status_code=503, detail="Failed to load writing history") from exc

    return {"submissions": resp.data or [], "is_pro": is_pro}


@app.get("/api/writing/submission/{submission_id}")
@limiter.limit("20/minute")
async def writing_submission_detail(
    request: Request,
    submission_id: str,
    authorization: Optional[str] = Header(None),
):
    """Full detail of one submission. Ownership failure returns 404 (not 403)
    so the existence of other users' submissions is never disclosed."""
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")

    resp = (
        supabase_admin.table("writing_submissions")
        .select("*")
        .eq("id", submission_id)
        .eq("user_id", user_id)
        .limit(1)
        .execute()
    )
    if not resp.data:
        raise HTTPException(status_code=404, detail="Submission not found")

    return {"submission": resp.data[0]}


@app.post("/api/admin/writing/pregen")
@limiter.limit("5/minute")
async def admin_trigger_pregen(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """Manually trigger writing question pregeneration. Admin only."""
    verify_admin(authorization)
    import threading
    t = threading.Thread(target=pregenerate_writing_questions, kwargs={"target_per_subtype": 5}, daemon=True)
    t.start()
    return {"status": "pregeneration started in background"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
