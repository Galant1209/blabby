#!/usr/bin/env python3
"""
Blabby P0.2 Memory Closure Smoke Test.

Verifies the Single Correction + Memory end-to-end behaviour by driving the
`/process` endpoint via the dev-only Whisper bypass. Each case hits the
real Supabase + Groq pipeline; only audio→text is skipped.

Run:
    python scripts/smoke_test_memory.py

Exit code:
    0 if all four cases pass, 1 otherwise.

Required env (via shell export or a `.env` file picked up by python-dotenv):
    SUPABASE_URL                  Supabase project URL
    SUPABASE_SERVICE_ROLE_KEY     Service-role key (bypass RLS for seed/cleanup)
    DEV_BYPASS_SECRET             Shared secret; must match the backend's env
    SMOKE_TEST_USER_JWT           Valid Supabase user JWT (maps to SMOKE_TEST_USER_ID)

Optional:
    SMOKE_TEST_BASE_URL           Default: http://localhost:8000
    SMOKE_TEST_USER_ID            Default: 00000000-0000-0000-0000-000000000001
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field

import requests
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# ─── Config ───────────────────────────────────────────────────────────────────

BASE_URL             = os.getenv("SMOKE_TEST_BASE_URL", "http://localhost:8000").rstrip("/")
TEST_USER_ID         = os.getenv("SMOKE_TEST_USER_ID", "00000000-0000-0000-0000-000000000001")
TEST_USER_JWT        = os.getenv("SMOKE_TEST_USER_JWT", "")
BYPASS_SECRET        = os.getenv("DEV_BYPASS_SECRET", "")
SUPABASE_URL         = os.getenv("SUPABASE_URL", "")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY", "")

# ─── Env validation (fail loudly and early) ───────────────────────────────────

_missing = []
if not TEST_USER_JWT:        _missing.append("SMOKE_TEST_USER_JWT")
if not BYPASS_SECRET:        _missing.append("DEV_BYPASS_SECRET")
if not SUPABASE_URL:         _missing.append("SUPABASE_URL")
if not SUPABASE_SERVICE_KEY: _missing.append("SUPABASE_SERVICE_ROLE_KEY")
if _missing:
    print("❌ Missing required env vars: " + ", ".join(_missing))
    print("   Set them via shell export or a .env file. See the docstring.")
    sys.exit(1)

supabase = create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)


# ─── Types ────────────────────────────────────────────────────────────────────

@dataclass
class TestResult:
    name: str
    passed: bool
    reason: str = ""
    ai_response: dict = field(default_factory=dict)


# ─── Helpers ──────────────────────────────────────────────────────────────────

def cleanup_test_user() -> None:
    """Remove every practice_records row for the test user."""
    supabase.table("practice_records").delete().eq("user_id", TEST_USER_ID).execute()


def seed_history(records: list[dict]) -> None:
    """
    Seed practice_records with fake history rows for the test user.
    Each record dict must carry at least `user_transcript`; topic / question
    / weakness_tag default to sensible values.
    """
    rows = []
    for r in records:
        rows.append({
            "user_id":              TEST_USER_ID,
            "topic":                r.get("topic", "General"),
            "question":             r.get("question", ""),
            "user_transcript":      r["user_transcript"],
            "coach_response":       "seeded",
            "better_expression":    "",
            "better_expression_zh": "",
            "next_question":        "",
            "weakness_tag":         r.get("weakness_tag", "weak_vocab"),
            "memory_snapshot":      {},
        })
    if rows:
        supabase.table("practice_records").insert(rows).execute()


def call_process(topic: str, question: str, user_transcript: str) -> dict:
    """
    Call /process with the dev bypass active. Returns the parsed JSON response.
    Uses `files=` with (None, value) tuples to force multipart encoding even
    without an audio file, so the backend's Form parsing is satisfied.
    """
    resp = requests.post(
        f"{BASE_URL}/process",
        headers={"Authorization": f"Bearer {TEST_USER_JWT}"},
        files={
            "level":             (None, "Band 5"),
            "topic":             (None, topic),
            "question":          (None, question),
            "history":           (None, "[]"),
            "text_override":     (None, user_transcript),
            "dev_bypass_secret": (None, BYPASS_SECRET),
        },
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()


# ─── Cases ────────────────────────────────────────────────────────────────────

def case_1_pure_off_topic() -> TestResult:
    cleanup_test_user()
    result = call_process(
        topic="Daily Routine",
        question="What time do you wake up in the morning?",
        user_transcript="My hometown is a small city in central Taiwan. The weather is warm.",
    )

    failures: list[str] = []
    if result.get("weakness_tag") != "off_topic":
        failures.append(f"weakness_tag expected 'off_topic', got {result.get('weakness_tag')!r}")
    if result.get("on_topic") is not False:
        failures.append(f"on_topic expected False, got {result.get('on_topic')!r}")

    coach_response = result.get("coach_response", "") or ""
    off_topic_keywords = ["沒有回答", "偏題", "沒回到題目", "沒回答題目"]
    if not any(k in coach_response for k in off_topic_keywords):
        failures.append(
            f"coach_response doesn't mention off-topic. "
            f"Expected one of {off_topic_keywords}. "
            f"coach_response={coach_response!r}"
        )

    return TestResult(
        name="Case 1: Pure off-topic",
        passed=not failures,
        reason="; ".join(failures) if failures else "all assertions passed",
        ai_response=result,
    )


def case_2_off_topic_plus_weak_word() -> TestResult:
    cleanup_test_user()
    result = call_process(
        topic="Environment",
        question="What kind of trash do you see in your community?",
        user_transcript="I think reading books makes me very happy and it is very good for my mind.",
    )

    failures: list[str] = []
    if result.get("weakness_tag") != "off_topic":
        failures.append(f"weakness_tag expected 'off_topic', got {result.get('weakness_tag')!r}")
    if result.get("on_topic") is not False:
        failures.append(f"on_topic expected False, got {result.get('on_topic')!r}")

    # coach_response joins single_issue + correction with '\n'; check only the
    # single_issue line (correction is allowed to mention 'weak word 下次再說').
    coach_response = result.get("coach_response", "") or ""
    single_issue_line = coach_response.split("\n", 1)[0] if coach_response else ""

    # THE RULE — must not be weakened. When on_topic is false, the single_issue
    # must not reference a weak word.
    WEAK_WORDS_TO_CHECK = ["very", "good", "interesting", "thing", "things", "stuff", "happy"]
    sil_lower = single_issue_line.lower()
    found_weak = [w for w in WEAK_WORDS_TO_CHECK if w in sil_lower]
    if found_weak:
        failures.append(
            f"SINGLE-POINT RULE VIOLATION: single_issue mentions weak words {found_weak}. "
            f"single_issue={single_issue_line!r}"
        )

    return TestResult(
        name="Case 2: Off-topic + weak word (CRITICAL)",
        passed=not failures,
        reason=(
            "; ".join(failures)
            if failures
            else "all assertions passed — single_issue clean of weak words"
        ),
        ai_response=result,
    )


def case_3_normal_weak_vocab() -> TestResult:
    cleanup_test_user()
    result = call_process(
        topic="Food",
        question="What is your favorite food?",
        user_transcript="My favorite food is beef noodles. It is very good and very interesting.",
    )

    failures: list[str] = []
    if result.get("weakness_tag") != "weak_vocab":
        failures.append(f"weakness_tag expected 'weak_vocab', got {result.get('weakness_tag')!r}")
    if result.get("on_topic") is not True:
        failures.append(f"on_topic expected True, got {result.get('on_topic')!r}")

    return TestResult(
        name="Case 3: Normal weak_vocab (on-topic)",
        passed=not failures,
        reason="; ".join(failures) if failures else "all assertions passed",
        ai_response=result,
    )


def case_4_memory_repeated_detection() -> TestResult:
    cleanup_test_user()
    seed_history([
        {
            "user_transcript": "The food was very good and very interesting.",
            "topic":           "Food",
            "question":        "What did you eat yesterday?",
        },
        {
            "user_transcript": "The movie is very good and the actors are very good.",
            "topic":           "Entertainment",
            "question":        "What movie did you watch recently?",
        },
        {
            "user_transcript": "I think the book is very good and very interesting.",
            "topic":           "Reading",
            "question":        "What book are you reading?",
        },
    ])
    time.sleep(0.5)  # Small grace period so Supabase has committed the seed.

    result = call_process(
        topic="Daily Routine",
        question="What do you usually do in the morning?",
        user_transcript="I always pet my cats because that's very good and very interesting.",
    )

    failures: list[str] = []
    coach_response = result.get("coach_response", "") or ""
    RECURRENCE_KEYWORDS = ["又", "再次", "重複", "舊問題", "第", "又用"]
    found_recurrence = [k for k in RECURRENCE_KEYWORDS if k in coach_response]

    if not found_recurrence:
        failures.append(
            f"MEMORY FAILURE: coach_response has no recurrence language. "
            f"Expected one of {RECURRENCE_KEYWORDS}. "
            f"coach_response={coach_response!r}"
        )

    return TestResult(
        name="Case 4: Memory repeated detection (CORE BLABBY VALUE)",
        passed=not failures,
        reason=(
            "; ".join(failures)
            if failures
            else f"recurrence language detected: {found_recurrence}"
        ),
        ai_response=result,
    )


# ─── Runner ───────────────────────────────────────────────────────────────────

def run_all() -> None:
    print("=" * 70)
    print("Blabby P0.2 Memory Closure Smoke Test")
    print(f"Base URL : {BASE_URL}")
    print(f"Test user: {TEST_USER_ID}")
    print("=" * 70)
    print()

    cases = [
        case_1_pure_off_topic,
        case_2_off_topic_plus_weak_word,
        case_3_normal_weak_vocab,
        case_4_memory_repeated_detection,
    ]

    results: list[TestResult] = []
    for case_fn in cases:
        print(f"Running: {case_fn.__name__} ...", flush=True)
        try:
            result = case_fn()
        except Exception as e:
            result = TestResult(
                name=case_fn.__name__,
                passed=False,
                reason=f"EXCEPTION: {type(e).__name__}: {e}",
            )
        status = "✅ PASS" if result.passed else "❌ FAIL"
        print(f"  {status}: {result.name}")
        print(f"  Reason : {result.reason}")
        if not result.passed and result.ai_response:
            print("  AI response (first 500 chars):")
            preview = json.dumps(result.ai_response, ensure_ascii=False, indent=2)[:500]
            print(f"  {preview}")
        print()
        results.append(result)

    # Leave the DB in a clean state regardless of outcomes.
    cleanup_test_user()

    print("=" * 70)
    passed = sum(1 for r in results if r.passed)
    total  = len(results)
    print(f"SUMMARY: {passed}/{total} passed")
    for r in results:
        mark = "✅" if r.passed else "❌"
        print(f"  {mark} {r.name}")
    print("=" * 70)

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    run_all()
