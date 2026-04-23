from fastapi import FastAPI, UploadFile, File, Request, Form, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import datetime
from typing import Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
import logging
import os
import json
import tempfile
import re
import time
from collections import Counter

load_dotenv()

GOOGLE_TTS_API_KEY   = os.getenv("GOOGLE_TTS_API_KEY")
SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

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


def get_user_recent_records(user_id: str, limit: int = 10) -> list[dict]:
    if not supabase_admin:
        raise HTTPException(status_code=503, detail="Database service not configured")

    try:
        response = (
            supabase_admin.table("practice_records")
            .select("user_transcript, topic, question, created_at")
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
    dynamic_weak_words = extract_dynamic_weak_words_from_history(transcripts)
    tracked_weak_words = WEAK_WORDS.union(dynamic_weak_words)
    for transcript in transcripts:
        if not transcript:
            continue
        words = tokenize_words(transcript)
        for word in words:
            if word in tracked_weak_words:
                counts[word] += 1

    ranked = [
        word
        for word, count in counts.most_common()
        if count >= 1
    ]
    return ranked[:max_items]


def detect_weakness_tag(
    transcript: str,
    weak_words_from_history: Optional[list[str]] = None,
) -> tuple[str, list[str]]:
    text = (transcript or "").strip().lower()
    tokens = tokenize_words(text)
    token_count = len(tokens)
    tracked_weak_words = WEAK_WORDS.union(weak_words_from_history or [])
    weak_word_hits = [word for word in tokens if word in tracked_weak_words]
    repeated_weak_words = [
        word for word in (weak_words_from_history or [])
        if re.search(rf"\b{re.escape(word.lower())}\b", text)
    ]

    if weak_word_hits:
        return "weak_vocab", repeated_weak_words

    if has_pattern_match(text, SAFE_PATTERNS):
        return "safe_answer", repeated_weak_words

    has_expansion_signal = has_pattern_match(text, EXPANSION_SIGNAL_PATTERNS)
    sentence_count = len([part for part in re.split(r"[.!?]+", text) if part.strip()])
    if token_count <= 8 or (token_count <= 14 and not has_expansion_signal) or (sentence_count <= 1 and token_count <= 10):
        return "lack_detail", repeated_weak_words

    return "grammar_minor", repeated_weak_words


def build_memory_block(weak_patterns: list[str]) -> str:
    if not weak_patterns:
        return ""

    repeated_words = ", ".join(weak_patterns)
    return (
        "\n【使用者歷史弱點】\n"
        f"- This user tends to overuse these weak words: {repeated_words}\n"
        "- If the user uses one of them again in this answer, point it out directly.\n"
        "- Repeated mistakes should be addressed more directly than first-time mistakes.\n"
        "- Keep the Blabby rule: choose only one main problem to correct.\n"
    )


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
    last_error = None
    for attempt in range(2):
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
            logger.warning("run_groq parse attempt %s failed: %s", attempt + 1, exc)
            if attempt == 0:
                time.sleep(0.3)
    raise HTTPException(status_code=502, detail="Coach response parsing failed") from last_error


def build_system_prompt(topic="General", memory_block="", repeated_weak_words: Optional[list[str]] = None):
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
- 第一次犯 → 自然、簡短、像教練提醒
- 重複犯 → 更直接，不要客氣，但不要羞辱
- 如果 memory 顯示這是 repeated weakness，而且這次又用了，直接指出這是重複問題
- 禁止空話，例如「你做得不錯」「可以再更好」「很棒喔」「整體來說還不錯」

【輸出規則】
- coach_response 必須用繁體中文
- 長度控制在 2 到 4 句
- 不要寫成小作文
- 一定要可執行，讓學生知道下一句要怎麼講得更好
- 必要時可以給一個短示範，但只能給一個
- 不要給三種版本，不要一次給很多替代答案
- 請用 JSON 的 single_issue 和 correction 強迫自己只處理一個問題

【絕對禁止】
- 不給總分
- 不給超過三個建議
- 不說「good job」「well done」這種空話

【Few-shot 範例】
Example 1 — first-time weakness
User answer:
"I think my hometown is very good and interesting."
Output:
{
  "single_issue": "你這次用了 very good / interesting 這種太空的詞，畫面還沒出來。",
  "correction": "先不要重講全部，只把 hometown 具體化。直接補一個細節，例如夜市很擠、街道很安靜，選一個講清楚。",
  "next_question": "What do people usually do in your hometown on weekends?",
  "better_expression": "lively night market",
  "better_expression_zh": "這個說法比 interesting 更有畫面，能直接把地方特色講出來。",
  "on_topic": true
}

Example 2 — repeated weakness
User answer:
"It depends. Sometimes reading is very good for me."
Output:
{
  "single_issue": "你又回到 it depends 跟 very good 這種安全說法了，這是重複問題。",
  "correction": "不要先躲，直接選一邊講。你就說最近一次讀了什麼、為什麼那次對你有幫助，給一個具體場景就夠。",
  "next_question": "What kind of books do you usually read?",
  "better_expression": "help me slow down",
  "better_expression_zh": "這種說法比 good for me 更自然，也更容易接出後面的細節。",
  "on_topic": true
}

【JSON 回應格式，不得偏離】
{
  "single_issue": "用繁體中文，一句話點名唯一痛點",
  "correction": "用繁體中文，2 到 3 句內給唯一矯正動作；必要時附一個短示範",
  "next_question": "下一個英文問題，自然銜接",
  "better_expression": "一個值得學的英文詞或短語",
  "better_expression_zh": "為什麼這個詞好用（中文）",
  "on_topic": true
}
"""
    diagnosis_context_block = (
        "\n【本次答案診斷】\n"
        + repeated_line
        + "- Diagnose the most painful issue from the transcript itself.\n"
        + "- Use memory only as context, not as a forced label.\n"
    )
    return base + memory_block + diagnosis_context_block + f"\n【本題主題】\n{topic}\n"


@app.post("/process")
@limiter.limit("20/minute")
async def process(
    request: Request,
    audio: UploadFile = File(...),
    level: str = Form("Band 5"),
    topic: str = Form("Free Time"),
    history: str = Form("[]"),
    authorization: Optional[str] = Header(None),
):
    try:
        user_id = verify_token(authorization)
        recent_records = get_user_recent_records(user_id)
        recent_transcripts = [
            record.get("user_transcript", "")
            for record in recent_records
            if record.get("user_transcript")
        ]
        weak_patterns = extract_weak_patterns(recent_transcripts)
        memory_block = build_memory_block(weak_patterns)

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
        weakness_tag, repeated_weak_words = detect_weakness_tag(user_text, weak_patterns)

        # Step 2: Groq chat (no extra round-trip to browser in between)
        try:
            history_list = json.loads(history) if history else []
            if not isinstance(history_list, list):
                history_list = []
        except json.JSONDecodeError:
            history_list = []
        messages = [{
            "role": "system",
            "content": build_system_prompt(topic, memory_block, repeated_weak_words)
        }]
        for msg in history_list[-10:]:
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})

        # run_groq already json.loads() the response, so `parsed` is a dict.
        parsed = run_groq(messages)
        if not isinstance(parsed, dict):
            parsed = {}
        single_issue = (parsed.get("single_issue") or "").strip()
        correction = (parsed.get("correction") or "").strip()
        if not single_issue and not correction:
            raise HTTPException(status_code=502, detail="Coach response was empty, please retry")
        coach_response = "\n".join(part for part in [single_issue, correction] if part)
        return {
            "text":                 user_text,
            "coach_response":       coach_response,
            "next_question":        parsed.get("next_question", ""),
            "better_expression":    parsed.get("better_expression", ""),
            "better_expression_zh": parsed.get("better_expression_zh", ""),
            "on_topic":             parsed.get("on_topic", True),
            "weakness_tag":         weakness_tag,
        }
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
