from fastapi import FastAPI, UploadFile, File, Request, Form, Depends, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response, StreamingResponse
from openai import OpenAI
from groq import Groq
from dotenv import load_dotenv
from supabase import create_client, Client
from datetime import date, timedelta, datetime
from typing import Optional
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from pydantic import BaseModel
import asyncio
import httpx
import os
import json
import base64
import tempfile
import requests as http_requests

load_dotenv()

GOOGLE_TTS_API_KEY   = os.getenv("GOOGLE_TTS_API_KEY")
SUPABASE_URL         = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

# Shared secret between frontend and backend.
# Set APP_API_KEY env var on the server to enable endpoint protection.
# If unset, checks are bypassed (backward-compatible during rollout).
APP_API_KEY = os.getenv("APP_API_KEY", "")

groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

app = FastAPI()
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Supabase admin client (service role — bypasses RLS)
supabase_admin: Client = (
    create_client(SUPABASE_URL, SUPABASE_SERVICE_KEY)
    if SUPABASE_URL and SUPABASE_SERVICE_KEY else None
)


async def require_app_key(x_app_key: Optional[str] = Header(None)):
    """Reject requests that don't carry the shared app key.
    Bypass when APP_API_KEY env var is not set (dev / gradual rollout)."""
    if APP_API_KEY and x_app_key != APP_API_KEY:
        raise HTTPException(status_code=403, detail="Forbidden")


async def require_user_token(authorization: Optional[str] = Header(None)):
    """Require a valid Supabase Bearer token on AI endpoints."""
    if not supabase_admin:
        raise HTTPException(status_code=503, detail="Auth service not configured")
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Please sign in first")
    token = authorization[7:]
    try:
        resp = supabase_admin.auth.get_user(token)
        if not resp.user:
            raise HTTPException(status_code=401, detail="Invalid or expired token")
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Token verification failed")


def verify_token(authorization: str) -> str:
    """Verify Bearer JWT and return user_id, or raise ValueError."""
    if not supabase_admin:
        raise ValueError("Supabase not configured on server")
    if not authorization or not authorization.startswith("Bearer "):
        raise ValueError("Missing or invalid authorization header")
    token = authorization[7:]
    resp = supabase_admin.auth.get_user(token)
    if not resp.user:
        raise ValueError("Invalid or expired token")
    return resp.user.id


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
    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=messages,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


def build_system_prompt(level: str, topic: str) -> str:
    """Placeholder IELTS Speaking system prompt — replace with real logic."""
    return (
        "You are an IELTS Speaking examiner and coach. "
        f"Target band level: {level}. Topic: {topic}. "
        "Reply in JSON with fields: "
        "'reply' (your natural follow-up question or response), "
        "'feedback' (constructive feedback on fluency/lexical/grammar/pronunciation), "
        "'error_tags' (array of issue tags; use ['perfect'] if no errors)."
    )


@app.post("/process")
@limiter.limit("20/minute")
async def process(
    request: Request,
    audio: UploadFile = File(...),
    level: str = Form("Band 5"),
    topic: str = Form("Free Time"),
    history: str = Form("[]")
):
    try:
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

        # Step 2: Groq chat (no extra round-trip to browser in between)
        history_list = json.loads(history)
        messages = [{"role": "system", "content": build_system_prompt(level, topic)}]
        for msg in history_list[-10:]:
            role    = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": user_text})

        result = run_groq(messages)
        return {
            "text":       user_text,
            "reply":      result.get("reply", ""),
            "feedback":   result.get("feedback", ""),
            "error_tags": result.get("error_tags", [])
        }
    except Exception as e:
        print(f"Process error: {str(e)}")
        return {"error": str(e)}


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000)
