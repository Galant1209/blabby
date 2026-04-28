from fastapi import FastAPI, UploadFile, File, Request, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime, timedelta, timezone
from typing import Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
import os
import json
import random
import tempfile
import re
import time
import uuid
from collections import Counter

load_dotenv()

GOOGLE_TTS_API_KEY   = os.getenv("GOOGLE_TTS_API_KEY")
SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")
ADMIN_EMAILS         = {
    email.strip().lower()
    for email in os.getenv("ADMIN_EMAILS", "").split(",")
    if email.strip()
}

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    logging.basicConfig(level=logging.INFO)

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
        "system_injection": """This turn is a VOCABULARY PRECISION DRILL.

The user has been overusing safe, low-precision words like "very", "good", "nice", "interesting", "thing", "stuff". Your job this turn is to give feedback that makes them feel — viscerally — that their word choice just leveled up.

After the user speaks, you MUST do these things in your feedback:

1. COUNT the B2+ vocabulary items they used. List them by name. Be specific: "你這次用了 3 個進階詞:particularly, distinctive, overwhelming." Numbers carry more weight than adjectives.

2. NAME any safe-word slips. If they used "very X" or "good", quote it directly and offer ONE precise replacement. Format:
「`very tasty` → 試試 `mouth-watering`」
Give the replacement, not a vague "try stronger words".

3. PRAISE specifically when they replaced a "very + adj" with a single stronger word. This is the win condition. Make them feel it: "你把 'very tired' 換成 'exhausted' 這一招很乾淨."

4. END with ONE concrete next-move. Format:
「下次試試把 `it was good` 直接換成一個動詞.例如 `it stood out` 或 `it surprised me`.」

DO NOT write generic encouragements like "Great effort!" or "Keep practicing!" — they are forbidden in drill mode. Every sentence in your feedback must point to a specific word the user said or a specific word they should try next.

Output the drill_score with axis="vocab_precision_score". Score 0-100 based on density of B2+ vocabulary AND absence of forbidden safe-words. threshold_passed = (score >= 70)."""
    },
    "lack_detail": {
        "expected_axis": "detail_density_score",
        "system_injection": """This turn is a CONCRETE DETAIL DRILL.

The user has been giving abstract, generic answers — empty of scene, numbers, sensory texture. Your job this turn is to make them feel that their answer just gained physical weight.

After the user speaks, you MUST do these things in your feedback:

1. COUNT the concrete details they included. Categorize them:
   - 時間 (when, how long, what year)
   - 地點 (specific place names, not "somewhere")
   - 數字 (price, quantity, frequency)
   - 感官 (smell, sound, texture, color)
   - 人物 (named people, not "my friend")

Format: "你這次給了 5 個具體細節:2018 年(時間)、台中夜市(地點)、80 塊(數字)、肉桂的味道(感官)、阿嬤(人物)."

Numbers and named specifics carry more weight than adjectives.

2. NAME any abstract slips. If they said "it was nice" or "we had fun", quote it and ask for the missing dimension:
「`it was nice` ← 缺感官細節.當下你聞到什麼?聽到什麼?」
Give the missing dimension, not vague "add more details".

3. PRAISE specifically when they used multi-sensory description. This is the win condition. Make them feel it: "你說 `the smell of cinnamon hit me first` 這種寫法是滿分等級的,讀者立刻聞到."

4. END with ONE concrete next-move. Format:
「下次試試在第一句就丟一個數字或地名.例如不要說 `I went to a restaurant`,直接說 `Last Tuesday I went to 鼎泰豐 in 信義區`.」

DO NOT write generic encouragements like "Good job!" or "Add more details!" — they are forbidden in drill mode. Every sentence in your feedback must reference a specific moment in the user's answer or a specific dimension they should add next.

Output the drill_score with axis="detail_density_score". Score 0-100 based on count and variety of concrete details across the 5 dimensions (time/place/number/sense/person). threshold_passed = (score >= 70)."""
    },
}
WEAKNESS_SUMMARY_WINDOW = 20
QUESTION_EXCLUSION_DAYS = 3
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
            .select("user_transcript, topic, question, created_at, weakness_tag")
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
- progress_note 欄位留空。
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
    6. better_phrasing_en <= 30 characters (when non-empty)
    7. better_phrasing_zh <= 30 characters (when non-empty)
    8. next_task <= 40 characters
    9. When expected_drill_axis is set (drill mode), data["drill_score"] must
       exist as a dict with axis (str matching expected), score (int 0-100),
       feedback (non-empty str), threshold_passed (bool). When
       expected_drill_axis is None, drill_score is not checked — its presence
       or absence is ignored for non-drill turns.

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
    if better_phrasing_en and len(better_phrasing_en) > 30:
        return False, f"better_phrasing_en is {len(better_phrasing_en)} chars (max 30)"
    if better_phrasing_zh and len(better_phrasing_zh) > 30:
        return False, f"better_phrasing_zh is {len(better_phrasing_zh)} chars (max 30)"
    if len(next_task) > 40:
        return False, f"next_task is {len(next_task)} chars (max 40)"

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

    return True, ""


def build_diagnosis_prompt(records: list[dict]) -> tuple[str, str]:
    """
    Build system + user prompt for AI diagnosis.
    records: list of practice_records dicts, sorted by created_at ASC
    Returns (system_prompt, user_prompt)
    """
    history_text = ""
    for i, r in enumerate(records, 1):
        history_text += f"\n【第 {i} 筆】\n"
        history_text += f"題目：{r.get('question', '-')}\n"
        history_text += f"學生回答：{r.get('user_transcript', '-')}\n"
        history_text += f"Blabby 教練回饋：{r.get('coach_response', '-')}\n"
        history_text += f"---\n"

    system_prompt = """你是一位資深 IELTS 口說教練，專門診斷 Band 4-6 台灣學生的口說弱點。

你的任務是讀完這位學生所有的練習記錄，用繁體中文產出一份診斷報告。

報告格式（必須嚴格遵守）：

## 程度判斷
- 預估 Band（4-7）
- 主要依據（2-3 句）

## 前三大卡點（按嚴重度排序）
1. [卡點名稱，例如「詞彙貧乏」]
   - 具體證據（引用哪一筆的哪一句話）
   - 發生頻率（N/N 筆出現）
2. ...
3. ...

## 她在哪裡停住
- 她每次講話停止的模式（2-3 個觀察）
- 例如：「講完一個事實後就停」、「想到要舉例就卡」

## 下一步建議
- 下一題該用什麼類型的題目
- 為什麼（教學邏輯）

## 危險訊號
- 重複模式 / 套模板 / 應付式回答
- 沒有則寫「無」

規則：
- 不打分數（Band 只是教練參考，不是成績）
- 不給空泛建議（像「多練習」這種不要寫）
- 每個卡點要有具體 transcript 段落當證據
- 語氣直接，像對一個同事講話，不要客氣
- 整份報告控制在 500 字內，精簡
"""

    user_prompt = f"以下是學生的練習記錄：\n{history_text}\n\n請產出診斷報告。"

    return system_prompt, user_prompt


def build_system_prompt(
    topic: str = "General",
    question: str = "",
    memory_block: str = "",
    repeated_weak_words: Optional[list[str]] = None,
    drill_tag: Optional[str] = None,
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
3. 如果 memory_block 有 weak word 記錄，且這次的回答裡**看不到那個 weak word**，也**有具體細節出現** → 層級 B（具體肯定）；肯定那個具體細節，再處理其他問題
4. 其他情況 → 層級 A（預設）

【輸出規則 — schema enforcement】
- correction 必須是物件（object），不得是 array、list、或 array of objects。You MUST return exactly ONE correction. Not two. Not three. ONE.
- correction 物件包含四個欄位，全部必填，缺一個或留空字串都視為違規：
  - quoted: 從用戶原句直接引用的片段，讓學生看到自己講了什麼
  - why_it_hurts: 為什麼這個地方傷害表達；繁體中文；最多 60 個字（why_it_hurts must be under 60 Chinese characters. Count before responding.）
  - better_phrasing_en: 一個更好的講法（英文版本）；最多 30 個字（含字母與標點；better_phrasing_en must be under 30 characters.）
  - better_phrasing_zh: 上述英文版本的中文對照；最多 30 個中文字
  - next_task: 下一輪請學生試的具體任務；繁體中文；最多 40 個字（next_task must be under 40 Chinese characters.）
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
  "tag": "weak_vocab",
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

【JSON 回應格式，不得偏離】
{
  "correction": {
    "quoted": "從用戶原句直接引用的片段（不可省略、不可改寫）",
    "why_it_hurts": "為什麼這個地方傷害表達；繁中；最多 60 字",
    "better_phrasing_en": "一個更好的講法（英文版本）；最多 30 字；偏題時可為空字串",
    "better_phrasing_zh": "上述英文版本的中文對照；最多 30 中文字；偏題時可為空字串",
    "next_task": "下一輪請學生試的具體任務；繁中；最多 40 字"
  },
  "tag": "本次回答最主要的問題分類，只能從這五個值選一個：weak_vocab（用 very/good/interesting 等空泛詞）、safe_answer（回答太空泛）、lack_detail（缺乏細節）、grammar_minor（文法小錯）、off_topic（完全沒回答題目）。若同時有多個問題，選最嚴重的那一個；若是 off_topic 必定選 off_topic，優先於所有其他 tag。",
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
        base
        + memory_block
        + diagnosis_context_block
        + question_block
        + f"\n【本題主題】\n{topic}\n"
        + drill_block
    )


@app.post("/process")
@limiter.limit("20/minute")
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
    authorization: Optional[str] = Header(None),
):
    try:
        user_id = verify_token(authorization)

        # Drill-mode validation: gate before any expensive op (recent records
        # pull, audio download, Groq call). 422 is the conventional FastAPI
        # status for request-shape validation failures.
        is_drill_mode = (mode == "drill")
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
            if len(audio_bytes) > 25 * 1024 * 1024:
                raise HTTPException(status_code=413, detail="Audio file too large, please re-record")
            ext = os.path.splitext(audio.filename or "")[1] or ".webm"
            with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name
            try:
                with open(tmp_path, "rb") as f:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1", file=f, language="en"
                    )
            finally:
                os.unlink(tmp_path)
            user_text = transcript.text
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
                memory_block=memory_block + tier_b_override + intensity_block,
                repeated_weak_words=repeated_weak_words,
                drill_tag=drill_tag if is_drill_mode else None,
            )
        }]
        for msg in history_list[-10:]:
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})

        # Validation retry layer (2 attempts max): if the LLM returns a
        # structurally invalid correction (wrong shape, missing fields,
        # over char-limit), retry once. Independent of run_groq's
        # connection-level retries (which handle JSON parse / 5xx /
        # json_validate_failed inside the SDK call). After 2 validation
        # failures, surface a 503 to the client.
        max_validation_attempts = 2
        parsed: dict = {}
        is_valid = False
        last_validation_reason = ""
        for validation_attempt in range(max_validation_attempts):
            parsed = run_groq(messages)
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
        #                    next_task is parsed and held in scope for a future
        #                    UI surface, but not exposed to the current frontend.
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
        next_question        = ""

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

        # Server-side persistence (was previously done client-side).
        # Uses supabase_admin (service_role) to bypass RLS. Failures are logged
        # but do NOT fail the response — the student still gets their feedback
        # even if the write breaks. `persisted` surfaces the state for the
        # client (or test harness) to act on.
        persisted = False
        new_record_id: Optional[str] = None
        if supabase_admin is not None:
            try:
                insert_resp = supabase_admin.table("practice_records").insert({
                    "user_id":              user_id,
                    "topic":                topic or "",
                    "question":             question or "",
                    "user_transcript":      user_text or "",
                    "coach_response":       coach_response,
                    "better_expression":    better_expression,
                    "better_expression_zh": better_expression_zh,
                    "next_question":        next_question,
                    "weakness_tag":         weakness_tag,
                    "memory_snapshot":      memory_snapshot,
                }).execute()
                persisted = True
                # Capture the fresh row's id so the resolution lookup below
                # can cleanly exclude it (avoids relying on ordering races).
                rows = insert_resp.data or []
                if rows:
                    new_record_id = rows[0].get("id")
            except Exception:
                logger.exception(
                    "failed to insert practice_record",
                    extra={"user_id": user_id},
                )

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

        response_payload = {
            "text":                 user_text,
            "coach_response":       coach_response,
            "next_question":        next_question,
            "better_expression":    better_expression,
            "better_expression_zh": better_expression_zh,
            "on_topic":             on_topic,
            "weakness_tag":         weakness_tag,
            "memory_snapshot":      memory_snapshot,
            "progress_note":        progress_note,
            "persisted":            persisted,
        }
        # Drill mode adds drill_score; non-drill turns return identical shape
        # to the previous version (acceptance: existing /process callers
        # unaffected when mode != "drill").
        if is_drill_mode:
            drill_score_data = parsed.get("drill_score")
            if isinstance(drill_score_data, dict):
                response_payload["drill_score"] = {
                    "axis":              drill_score_data.get("axis"),
                    "score":             drill_score_data.get("score"),
                    "feedback":          (drill_score_data.get("feedback") or "").strip(),
                    "threshold_passed":  drill_score_data.get("threshold_passed"),
                }
        return response_payload
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(
            "process endpoint failed",
            extra={"user_id": user_id if "user_id" in locals() else None},
        )
        raise HTTPException(status_code=500, detail="Internal error, please try again") from e


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


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


@app.get("/api/practice-records/weakness-summary")
@limiter.limit("10/minute")
async def practice_records_weakness_summary(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    """
    Aggregate the caller's last WEAKNESS_SUMMARY_WINDOW practice_records by
    weakness_tag, returning total eligible rows + tag counts sorted by count
    desc (ties broken by the tag's most recent created_at).

    Below the 5-row threshold the response is shaped {"total": N, "tag_counts": []}
    so the client can decide rendering without a second round-trip.
    """
    user_id = verify_token(authorization)
    if supabase_admin is None:
        raise HTTPException(status_code=503, detail="Database not configured")
    try:
        response = (
            supabase_admin.table("practice_records")
            .select("id, weakness_tag, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=True)
            .limit(WEAKNESS_SUMMARY_WINDOW)
            .execute()
        )
        rows = response.data or []

        eligible = [
            row for row in rows
            if (row.get("weakness_tag") or "") in ALLOWED_WEAKNESS_TAGS
        ]
        total = len(eligible)
        if total < 5:
            return {"total": total, "tag_counts": []}

        counts: Counter = Counter()
        # ISO-8601 strings sort lexicographically, so the lex max is the most
        # recent timestamp — no need to parse datetimes for the tie-breaker.
        latest_seen: dict[str, str] = {}
        for row in eligible:
            tag = row["weakness_tag"]
            counts[tag] += 1
            ts = row.get("created_at") or ""
            if tag not in latest_seen or ts > latest_seen[tag]:
                latest_seen[tag] = ts

        ordered = sorted(
            counts.items(),
            key=lambda kv: (kv[1], latest_seen.get(kv[0], "")),
            reverse=True,
        )
        return {
            "total": total,
            "tag_counts": [{"tag": tag, "count": count} for tag, count in ordered],
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "weakness-summary query failed", extra={"user_id": user_id}
        )
        raise HTTPException(status_code=500, detail="Failed to load weakness summary")


@app.get("/api/questions/bank")
async def get_question_bank():
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

        return {
            "id": chosen.get("id"),
            "text": chosen.get("text"),
            "topic": chosen.get("topic"),
            "part": chosen.get("part"),
        }
    except HTTPException:
        raise
    except Exception:
        logger.exception(
            "next-question selection failed", extra={"user_id": user_id}
        )
        raise HTTPException(status_code=500, detail="Failed to load next question")


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


@app.get("/admin/recent")
@limiter.limit("30/minute")
async def admin_recent(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    try:
        verify_admin(authorization)
        email_cache: dict[str, str] = {}
        response = (
            supabase_admin.table("practice_records")
            .select("id, user_id, topic, question, user_transcript, coach_response, weakness_tag, created_at")
            .order("created_at", desc=True)
            .limit(50)
            .execute()
        )

        records = []
        for record in response.data or []:
            record_user_id = record.get("user_id") or ""
            if record_user_id not in email_cache:
                try:
                    user_response = supabase_admin.auth.admin.get_user_by_id(record_user_id)
                    user = getattr(user_response, "user", None)
                    email_cache[record_user_id] = getattr(user, "email", None) or "unknown"
                except Exception:
                    email_cache[record_user_id] = "unknown"

            record_with_email = dict(record)
            record_with_email["email"] = email_cache.get(record_user_id, "unknown")
            records.append(record_with_email)

        return {"records": records}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("admin recent endpoint failed")
        raise HTTPException(status_code=500, detail="Failed to load admin data") from exc


@app.get("/admin/users")
@limiter.limit("30/minute")
async def admin_users(
    request: Request,
    authorization: Optional[str] = Header(None),
):
    try:
        verify_admin(authorization)

        # PostgREST doesn't expose aggregate SQL via the Python client,
        # so pull the cheap columns and group in memory.
        response = (
            supabase_admin.table("practice_records")
            .select("user_id, created_at")
            .execute()
        )

        aggregates: dict[str, dict] = {}
        for row in response.data or []:
            uid = row.get("user_id")
            if not uid:
                continue
            created = row.get("created_at")
            slot = aggregates.get(uid)
            if slot is None:
                aggregates[uid] = {
                    "user_id": uid,
                    "record_count": 1,
                    "first_seen": created,
                    "last_seen": created,
                }
            else:
                slot["record_count"] += 1
                # ISO-8601 strings sort lexicographically — safe without parsing.
                if created:
                    if slot["first_seen"] is None or created < slot["first_seen"]:
                        slot["first_seen"] = created
                    if slot["last_seen"] is None or created > slot["last_seen"]:
                        slot["last_seen"] = created

        users = sorted(
            aggregates.values(),
            key=lambda u: u["last_seen"] or "",
            reverse=True,
        )
        return {"users": users}
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("admin users endpoint failed")
        raise HTTPException(status_code=500, detail="Failed to load admin users") from exc


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
                "topic, question, user_transcript, coach_response, "
                "better_expression, better_expression_zh, next_question, "
                "weakness_tag, memory_snapshot, created_at"
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


@app.post("/admin/user/{user_id}/diagnosis")
@limiter.limit("10/minute")
async def admin_user_diagnosis(
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

        response = (
            supabase_admin.table("practice_records")
            .select("question, user_transcript, coach_response, created_at")
            .eq("user_id", user_id)
            .order("created_at", desc=False)
            .execute()
        )

        records = response.data or []

        if len(records) == 0:
            return {
                "user_id": user_id,
                "total_records": 0,
                "generated_at": datetime.utcnow().isoformat() + "Z",
                "diagnosis_markdown": "（這位使用者還沒有練習記錄，無法診斷）",
            }

        system_prompt, user_prompt = build_diagnosis_prompt(records)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        completion = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.3,
            max_tokens=800,
        )

        diagnosis_text = completion.choices[0].message.content

        return {
            "user_id": user_id,
            "total_records": len(records),
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "diagnosis_markdown": diagnosis_text,
        }

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("admin diagnosis endpoint failed", extra={"target_user_id": user_id})
        raise HTTPException(status_code=500, detail=f"Diagnosis failed: {str(exc)}") from exc


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
