"""
Regression tests for the /process audio decodability pre-flight probe
(P0 fix, 2026-07-31).

Root cause: stopRecording() in index.html called MediaRecorder.requestData()
unconditionally right before .stop(), including on the webm/desktop Chrome
path where it was never needed. That extra forced flush split the recording
into two ondataavailable events and left Chromium's webm muxer finalizing
the Segment before it had a real duration to write — the resulting blob
carried full-size, structurally valid audio but decoded to ~0s duration,
which Groq's Whisper endpoint rejected as "Audio file is too short."

These tests cover the backend backstop: _media_duration_seconds must be
probed BEFORE the Groq call, and a file that fails the probe must be
rejected with a clear 422 — never silently forwarded to Groq to fail there.
"""

from __future__ import annotations

import asyncio
import base64
import io

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("anthropic")
pytest.importorskip("supabase")

from unittest.mock import MagicMock, patch

from fastapi import HTTPException, UploadFile

import main  # noqa: E402

# 1.5s of real opus audio, encoded with ffmpeg (libopus) — decodable,
# well over the pre-existing <1024-byte empty-blob guard in /process.
VALID_WEBM = base64.b64decode(
    "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"
)

# The same recording, truncated to 200 bytes (cutting off before the
# Cluster/audio-frame data) and padded with zeros to stay above the
# <1024-byte empty-blob guard. Still opens with the EBML magic, so
# _detect_audio_container() accepts it as webm — but hachoir (and the
# EBML-Cluster fallback) find no duration in it, reproducing the class of
# damage this task's probe exists to catch: a structurally-plausible but
# undecodable blob that reaches /process with a normal-looking size.
CORRUPTED_WEBM = VALID_WEBM[:200] + b"\x00" * 900


def _run(coro):
    return asyncio.run(coro)


class _FakeResult:
    def __init__(self, data=None, count=0):
        self.data = data if data is not None else []
        self.count = count


class _FakeQuery:
    def insert(self, payload, *a, **k):
        return self

    def __getattr__(self, _name):
        def _chain(*a, **k):
            return self
        return _chain

    def execute(self):
        return _FakeResult(data=[{"id": "fake-record-id"}], count=0)


class _FakeSupabase:
    def table(self, _name):
        return _FakeQuery()


VALID_CORRECTION = {
    "correction": {
        "quoted": "I very like it",
        "why_it_hurts": "非正式且文法上不常見的強調方式。",
        "better_phrasing_en": "I really like it",
        "better_phrasing_zh": "我真的很喜歡",
        "next_task": "再說一次，把 very 換成 really。",
    },
    "on_topic": True,
    "tag": "grammar_minor",
    "progress_note": "",
}


def _upload(data: bytes, filename: str = "recording.webm"):
    return UploadFile(filename=filename, file=io.BytesIO(data), headers={"content-type": "audio/webm"})


def _call_process(*, audio_bytes: bytes, transcribe_mock=None):
    kwargs = dict(
        request=MagicMock(),
        audio=_upload(audio_bytes),
        level="Band 5",
        topic="Hobbies",
        question="What do you usually do in your free time?",
        history="[]",
        text_override="",
        dev_bypass_secret="",
        mode="",
        drill_tag="",
        previous_transcript="",
        retry_of="",
        authorization="Bearer fake",
    )
    transcribe = transcribe_mock or MagicMock(
        return_value=MagicMock(text="I very like sports and I play them every weekend.")
    )
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", return_value="user-abc"), \
         patch.object(main, "get_user_pro_status", return_value=True), \
         patch.object(main, "get_user_recent_records", return_value=[]), \
         patch.object(main, "run_claude", return_value=dict(VALID_CORRECTION)), \
         patch.object(main, "supabase_admin", _FakeSupabase()), \
         patch.object(main.groq_client.audio.transcriptions, "create", transcribe):
        return _run(main.process(**kwargs)), transcribe


def test_header_damaged_webm_rejected_with_422_before_groq():
    """A webm that passes container sniffing (starts with the EBML magic)
    but has no decodable duration must be rejected by the probe — and Groq
    must never be called for it."""
    assert CORRUPTED_WEBM[:4] == b"\x1aE\xdf\xa3"  # still looks like webm to container sniffing
    assert len(CORRUPTED_WEBM) >= 1024  # past the pre-existing empty-blob guard

    with pytest.raises(HTTPException) as excinfo:
        _call_process(audio_bytes=CORRUPTED_WEBM)
    assert excinfo.value.status_code == 422

    # Re-run capturing the transcribe mock to assert it was never invoked.
    transcribe = MagicMock()
    with pytest.raises(HTTPException):
        _call_process(audio_bytes=CORRUPTED_WEBM, transcribe_mock=transcribe)
    transcribe.assert_not_called()


def test_valid_webm_passes_probe_and_reaches_groq():
    """Sanity check: the probe must not false-positive-reject real audio."""
    result, transcribe = _call_process(audio_bytes=VALID_WEBM)
    transcribe.assert_called_once()
    assert result["text"]
