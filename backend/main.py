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

【選擇層級的判斷流程】

1. 如果 memory_block 顯示使用者某 weak word 歷史出現 ≥ 3 次，且這次又用了 → 層級 C（直球問）
2. 如果 memory_block 顯示某 weak word 歷史 < 3 次，且這次又用了 → 層級 A（預設）+ 提一次「又出現」
3. 如果 memory_block 有 weak word 記錄，且這次的回答裡**看不到那個 weak word**，也**有具體細節出現** → 層級 B（具體肯定）；肯定那個具體細節，再處理其他問題
4. 其他情況 → 層級 A（預設）

【輸出規則】
- coach_response 必須用繁體中文
- 長度控制在 2 到 4 句
- 不要寫成小作文
- 一定要可執行，讓學生知道下一句要怎麼講得更好
- correction 的結構只能是這樣，不得偏離：
  第一句：用中文說這個動作要怎麼做（不含任何示範）
  第二句（選填）：「試試：'[一個英文示範句]'」——只能一句，必須英文，必須用單引號包住
- correction 裡絕對禁止：頓號或逗號分隔的詞彙清單、中文示範句、超過一個示範、「另外」「同時」「也可以」等連接詞
- 請用 JSON 的 single_issue 和 correction 強迫自己只處理一個問題
- 如果 on_topic 是 false（學生完全沒回答題目），better_expression 和 better_expression_zh 可以是空字串，不要硬塞詞彙。優先把學生拉回題目，詞彙下次再教。

【on_topic 判斷規則】
- 如果有提供【本題題目】，判斷學生的回答是否真的在回答這個題目
- 只要學生的回答跟題目主軸對得上，即使細節薄弱也算 on_topic: true
- 只有明顯離題（例如題目問地點，學生講的完全是別的話題）才 on_topic: false
- 如果判斷 on_topic: false，single_issue 要把「答非所問」當作首要問題，優先於 weak words
- 當學生偏題時（on_topic: false），即使他也用了 weak word（very / good / interesting 等），single_issue **只能**提偏題，**不要**同時提 weak word。weak word 下次再處理。
- 這條是鐵律：偏題時 single_issue 不可出現任何 weak word 的評論。

【絕對禁止】
- 不給總分
- 不給超過三個建議
- 不說「good job」「well done」這種空話
- 不在 correction 裡列出多個替代詞或多個動詞選項；選最好的那一個，其他捨棄

【Few-shot 範例】
Example 1 — first-time weakness
User answer:
"I think my hometown is very good and interesting."
Output:
{
  "single_issue": "你這次用了 very good / interesting 這種太空的詞，畫面還沒出來。",
  "correction": "先不要重講全部，只把 hometown 具體化。給一個細節 — 試試：'The night market gets so crowded you can barely move.'",
  "next_question": "What do people usually do in your hometown on weekends?",
  "better_expression": "lively night market",
  "better_expression_zh": "這個說法比 interesting 更有畫面，能直接把地方特色講出來。",
  "weakness_tag": "weak_vocab",
  "on_topic": true
}

Example 2 — repeated weakness
User answer:
"It depends. Sometimes reading is very good for me."
Output:
{
  "single_issue": "你又回到 it depends 跟 very good 這種安全說法了，這是重複問題。",
  "correction": "不要先躲，直接選一邊講。給一個具體場景 — 試試：'Last week I read for an hour before bed and actually slept better.'",
  "next_question": "What kind of books do you usually read?",
  "better_expression": "help me slow down",
  "better_expression_zh": "這種說法比 good for me 更自然，也更容易接出後面的細節。",
  "weakness_tag": "weak_vocab",
  "on_topic": true
}

Example 3 — off-topic answer
Question:
"Where is the most interesting place you have visited?"
User answer:
"I think reading books is very good and I like stories a lot."
Output:
{
  "single_issue": "你沒有回答題目 — 題目問地點，你整段在講閱讀。",
  "correction": "先回到題目。給一個具體地點。試試：'I went to Jiufen last summer.'",
  "next_question": "Let's try again — name one specific place you've been to.",
  "better_expression": "",
  "better_expression_zh": "",
  "weakness_tag": "off_topic",
  "on_topic": false
}

Example 4 — off-topic + weak word (priority: off-topic wins)
Question:
"What kind of trash do you see in your community?"
User answer:
"I think reading books makes me very happy."
Output:
{
  "single_issue": "你沒有回答題目 — 題目問垃圾種類，你整段在講閱讀。",
  "correction": "先回到題目。給一個具體的垃圾種類。試試：'I see a lot of plastic bottles in my neighborhood.'",
  "next_question": "Let's come back to the question — name one kind of trash you see.",
  "better_expression": "",
  "better_expression_zh": "",
  "weakness_tag": "off_topic",
  "on_topic": false
}

Example 5 — real progress (層級 B)
Memory:
- History weak words: very (出現 3 次)
- Past feedback taught: "memorable"
User's answer this time:
"One place that stuck with me is Kyoto. The quiet streets left a really memorable feeling."
Output:
{
  "single_issue": "memorable 這個詞用得好，上次我們聊過 very 太模糊，你記住了。這次接著挑戰：用一個動詞讓畫面更具體。",
  "correction": "把「stuck with me」延伸下去 — 換成一個動詞。試試：'The stillness of the streets lingered.'",
  "next_question": "What exactly were you doing when that feeling hit you?",
  "better_expression": "the stillness lingered",
  "better_expression_zh": "lingered 比 memorable 更有畫面感，描繪「停留不散」的氛圍。",
  "weakness_tag": "lack_detail",
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
  "single_issue": "very 這個詞你已經用很多次了，每次提醒好像都卡住。是詞彙不夠，還是習慣問題？",
  "correction": "下次講話前停一下，想想 very 之外還能怎麼講。試試：'The food was absolutely mouth-watering.'",
  "next_question": "Let's pause here. What's one word other than 'very' you could try next time?",
  "better_expression": "mouth-watering",
  "better_expression_zh": "比 very good 更精準，直接傳達「看了就想吃」。",
  "weakness_tag": "weak_vocab",
  "on_topic": true
}

【JSON 回應格式，不得偏離】
{
  "single_issue": "用繁體中文，一句話點名唯一痛點",
  "correction": "用繁體中文，2 到 3 句內給唯一矯正動作；必要時附一個短示範",
  "next_question": "",
  "better_expression": "一個值得學的英文詞或短語；若學生偏題則可為空字串",
  "better_expression_zh": "為什麼這個詞好用（中文）；若 better_expression 為空則一併留空",
  "weakness_tag": "本次回答最主要的問題分類，只能從這五個值選一個：weak_vocab（用 very/good/interesting 等空泛詞）、safe_answer（回答太空泛）、lack_detail（缺乏細節）、grammar_minor（文法小錯）、off_topic（完全沒回答題目）。若同時有多個問題，選最嚴重的那一個；若是 off_topic 必定選 off_topic，優先於所有其他 tag。",
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
    return (
        base
        + memory_block
        + diagnosis_context_block
        + question_block
        + f"\n【本題主題】\n{topic}\n"
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
                memory_block=memory_block + tier_b_override,
                repeated_weak_words=repeated_weak_words,
            )
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
        next_question        = parsed.get("next_question", "") or ""
        better_expression    = parsed.get("better_expression", "") or ""
        better_expression_zh = parsed.get("better_expression_zh", "") or ""
        on_topic             = parsed.get("on_topic", True)

        # Groq now produces weakness_tag itself. Validate against the allow-list
        # so a hallucinated value never pollutes the DB / admin dashboards.
        weakness_tag = (parsed.get("weakness_tag") or "").strip()
        if weakness_tag not in ALLOWED_WEAKNESS_TAGS:
            if weakness_tag:
                logger.warning(
                    "Groq returned invalid weakness_tag: %r, falling back to empty",
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

        return {
            "text":                 user_text,
            "coach_response":       coach_response,
            "next_question":        next_question,
            "better_expression":    better_expression,
            "better_expression_zh": better_expression_zh,
            "on_topic":             on_topic,
            "weakness_tag":         weakness_tag,
            "memory_snapshot":      memory_snapshot,
            "persisted":            persisted,
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
