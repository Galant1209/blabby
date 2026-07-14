"""P1 regression coverage for bounded and validated Part 2 uploads."""

from __future__ import annotations

import asyncio
import base64
import io
import time
from unittest.mock import patch

import pytest
from fastapi import HTTPException, UploadFile

import main


VALID_WEBM_HEADER = b"\x1aE\xdf\xa3" + b"\x00" * 28
VALID_MP4_HEADER = (16).to_bytes(4, "big") + b"ftyp" + b"M4A " + b"\x00" * 4
REAL_WEBM = base64.b64decode(
    "GkXfo59ChoEBQveBAULygQRC84EIQoKEd2VibUKHgQRChYECGFOAZwEAAAAAAAJjEU2bdLpNu4tTq4QVSalmU6yBoU27i1OrhBZUrmtTrIHYTbuMU6uEElTDZ1OsggFCTbuMU6uEHFO7a1OsggJN7AEAAAAAAABZAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAVSalmsirXsYMPQkBNgI1MYXZmNjIuMTIuMTAwV0GNTGF2ZjYyLjEyLjEwMESJiEBwIAAAAAAAFlSua+WuAQAAAAAAAFzXgQFzxYiW0S4RPjSPX5yBACK1nIN1bmSIgQCGhkFfT1BVU1aqg2MuoFa7hATEtACDgQLhkZ+BAbWIQM9AAAAAAABiZIEQY6KTT3B1c0hlYWQBATgBgD4AAAAAABJUw2f9c3OgY8CAZ8iaRaOHRU5DT0RFUkSHjUxhdmY2Mi4xMi4xMDBzc9djwItjxYiW0S4RPjSPX2fIokWjh0VOQ09ERVJEh5VMYXZjNjIuMjguMTAwIGxpYm9wdXNnyKFFo4hEVVJBVElPTkSHkzAwOjAwOjAwLjI1ODAwMDAwMAAfQ7Z1QIPngQCjh4EAAIC4//6jh4EAFYC4//6jh4EAKYC4//6jh4EAPYC4//6jh4EAUYC4//6jh4EAZYC4//6jh4EAeYC4//6jh4EAjYC4//6jh4EAoYC4//6jh4EAtYC4//6jh4EAyYC4//6jh4EA3YC4//6gkqGHgQDxALj//puBEXWigzVn4BxTu2uRu4+zgQC3iveBAfGCAcTwgQM="
)
REAL_MP4 = base64.b64decode(
    "AAAAHGZ0eXBNNEEgAAACAE00QSBpc29taXNvMgAAAAhmcmVlAAAALW1kYXTeAgBMYXZjNjIuMjguMTAwAAIwQA4BGCAHARggBwEYIAcBGCAHAAADD21vb3YAAABsbXZoZAAAAAAAAAAAAAAAAAAAA+gAAAD6AAEAAAEAAAAAAAAAAAAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAIAAAI5dHJhawAAAFx0a2hkAAAAAwAAAAAAAAAAAAAAAQAAAAAAAAD6AAAAAAAAAAAAAAABAQAAAAABAAAAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAQAAAAAAAAAAAAAAAAAAAJGVkdHMAAAAcZWxzdAAAAAAAAAABAAAA+gAABAAAAQAAAAABsW1kaWEAAAAgbWRoZAAAAAAAAAAAAAAAAAAAPoAAABOgVcQAAAAAAC1oZGxyAAAAAAAAAABzb3VuAAAAAAAAAAAAAAAAU291bmRIYW5kbGVyAAAAAVxtaW5mAAAAEHNtaGQAAAAAAAAAAAAAACRkaW5mAAAAHGRyZWYAAAAAAAAAAQAAAAx1cmwgAAAAAQAAASBzdGJsAAAAanN0c2QAAAAAAAAAAQAAAFptcDRhAAAAAAAAAAEAAAAAAAAAAAABABAAAAAAPoAAAAAAADZlc2RzAAAAAAOAgIAlAAEABICAgBdAFQAAAAABDYgAAAOuBYCAgAUUCFblAAaAgIABAgAAACBzdHRzAAAAAAAAAAIAAAAEAAAEAAAAAAEAAAOgAAAAHHN0c2MAAAAAAAAAAQAAAAEAAAAFAAAAAQAAAChzdHN6AAAAAAAAAAAAAAAFAAAAFQAAAAQAAAAEAAAABAAAAAQAAAAUc3RjbwAAAAAAAAABAAAALAAAABpzZ3BkAQAAAHJvbGwAAAACAAAAAf//AAAAHHNiZ3AAAAAAcm9sbAAAAAEAAAAFAAAAAQAAAGJ1ZHRhAAAAWm1ldGEAAAAAAAAAIWhkbHIAAAAAAAAAAG1kaXJhcHBsAAAAAAAAAAAAAAAALWlsc3QAAAAlqXRvbwAAAB1kYXRhAAAAAQAAAABMYXZmNjIuMTIuMTAw"
)


def _upload(data: bytes, content_type: str = "audio/webm", filename: str = "untrusted.bin"):
    return UploadFile(filename=filename, file=io.BytesIO(data), headers={"content-type": content_type})


def _run(coro):
    return asyncio.run(coro)


def _multipart(audio: bytes, mime: str = "audio/webm") -> bytes:
    boundary = b"blabby-security-boundary"
    return b"".join([
        b"--" + boundary + b"\r\n",
        b'Content-Disposition: form-data; name="audio"; filename="sample.bin"\r\n',
        b"Content-Type: " + mime.encode("ascii") + b"\r\n\r\n",
        audio,
        b"\r\n--" + boundary + b"--\r\n",
    ])


def _run_body_limit(body: bytes, *, limit: int, content_length=None, chunks=None):
    downstream_calls = []
    sent = []

    async def downstream(scope, receive, send):
        downstream_calls.append(scope)
        received = bytearray()
        while True:
            event = await receive()
            received.extend(event.get("body", b""))
            if not event.get("more_body", False):
                break
        await send({"type": "http.response.start", "status": 204, "headers": []})
        await send({"type": "http.response.body", "body": bytes(received)})

    headers = [(b"content-type", b"multipart/form-data; boundary=blabby-security-boundary")]
    if content_length is not None:
        headers.append((b"content-length", str(content_length).encode("ascii")))
    scope = {
        "type": "http", "method": "POST", "path": "/part2/evaluate",
        "headers": headers,
    }
    pieces = chunks if chunks is not None else [body]
    events = [
        {"type": "http.request", "body": piece, "more_body": i < len(pieces) - 1}
        for i, piece in enumerate(pieces)
    ]

    async def receive():
        return events.pop(0)

    async def send(message):
        sent.append(message)

    middleware = main.Part2RequestBodyLimitMiddleware(downstream, max_bytes=limit)
    _run(middleware(scope, receive, send))
    return sent, downstream_calls


def _response_status_and_body(sent):
    status = next(m["status"] for m in sent if m["type"] == "http.response.start")
    body = b"".join(m.get("body", b"") for m in sent if m["type"] == "http.response.body")
    return status, body


def test_preparser_rejects_oversized_content_length_without_invoking_endpoint():
    sent, calls = _run_body_limit(b"not-read", limit=8, content_length=9)
    assert _response_status_and_body(sent) == (413, b'{"detail":"Request body too large"}')
    assert calls == []


def test_preparser_rejects_oversized_body_without_content_length():
    sent, calls = _run_body_limit(b"123456789", limit=8)
    assert _response_status_and_body(sent)[0] == 413
    assert calls == []


def test_preparser_rejects_forged_smaller_content_length():
    sent, calls = _run_body_limit(b"123456789", limit=8, content_length=4)
    assert _response_status_and_body(sent)[0] == 413
    assert calls == []


def test_preparser_rejects_chunked_multipart_before_endpoint_or_provider():
    body = _multipart(VALID_WEBM_HEADER)
    chunks = [body[:10], body[10:30], body[30:]]
    sent, calls = _run_body_limit(body, limit=len(body) - 1, chunks=chunks)
    assert _response_status_and_body(sent)[0] == 413
    assert calls == []


@pytest.mark.parametrize(
    ("audio", "mime"),
    [(REAL_WEBM, "audio/webm"), (REAL_MP4, "audio/mp4")],
    ids=["valid-webm", "valid-mp4"],
)
def test_preparser_allows_valid_audio_below_limit(audio, mime):
    body = _multipart(audio, mime)
    sent, calls = _run_body_limit(body, limit=len(body), content_length=len(body))
    assert _response_status_and_body(sent) == (204, body)
    assert len(calls) == 1


def test_preparser_rejection_creates_no_temporary_file(tmp_path):
    old_tempdir = main.tempfile.tempdir
    main.tempfile.tempdir = str(tmp_path)
    try:
        sent, calls = _run_body_limit(b"x" * 9, limit=8)
        assert _response_status_and_body(sent)[0] == 413
        assert calls == []
        assert list(tmp_path.iterdir()) == []
    finally:
        main.tempfile.tempdir = old_tempdir


def test_empty_file_rejected():
    with pytest.raises(HTTPException) as exc:
        _run(main._read_upload_bounded(_upload(b""), 100))
    assert exc.value.status_code == 400


def test_oversized_file_rejected_without_unbounded_read():
    with pytest.raises(HTTPException) as exc:
        _run(main._read_upload_bounded(_upload(b"x" * 10), 8))
    assert exc.value.status_code == 413


@pytest.mark.parametrize(
    "data",
    [b"<html>not webm</html>", b"PK\x03\x04fake zip", b"\x1aE", b"\x00\x00\x00\x08ftyp"],
    ids=["fake-webm-html", "fake-webm-zip", "truncated-webm", "truncated-mp4"],
)
def test_fake_or_truncated_audio_rejected(data):
    with pytest.raises(HTTPException) as exc:
        main._detect_audio_container(data)
    assert exc.value.status_code == 415


def test_unsupported_content_type_rejected_before_read_or_provider():
    upload = _upload(VALID_WEBM_HEADER, "application/zip")
    with pytest.raises(HTTPException) as exc:
        _run(main._part2_evaluate_for_user("u", upload, "topic", "[]"))
    assert exc.value.status_code == 415
    assert upload.file.tell() == 0


@pytest.mark.parametrize(
    ("data", "mime", "container"),
    [
        (VALID_WEBM_HEADER, "audio/webm", "webm"),
        (VALID_MP4_HEADER, "audio/mp4", "mp4"),
    ],
)
def test_valid_webm_and_mp4_reach_provider_with_safe_detected_suffix(data, mime, container):
    persisted = []
    with patch.object(main, "_media_duration_seconds", return_value=30.0), \
         patch.object(main, "_transcribe_audio_file", return_value="A safe transcript."), \
         patch.object(main, "run_claude", return_value={"criteria": []}), \
         patch.object(main, "_persist_part2", side_effect=lambda *args: persisted.append(args)):
        result = _run(main._part2_evaluate_for_user(
            "user-a", _upload(data, mime, "attacker.exe"), "Topic", "[]"
        ))
    assert result["transcript"] == "A safe transcript."
    assert persisted
    assert main._detect_audio_container(data)[0] == container


@pytest.mark.parametrize(("data", "suffix"), [(REAL_WEBM, ".webm"), (REAL_MP4, ".m4a")])
def test_real_valid_webm_and_mp4_have_structural_duration(tmp_path, data, suffix):
    path = tmp_path / f"sample{suffix}"
    path.write_bytes(data)
    assert main._detect_audio_container(data)[0] in {"webm", "mp4"}
    assert 0 < main._media_duration_seconds(str(path)) < 1


def test_duration_limit_rejected_before_provider():
    with patch.object(main, "_media_duration_seconds", return_value=126.0), \
         patch.object(main, "_transcribe_audio_file") as transcribe:
        with pytest.raises(HTTPException) as exc:
            _run(main._part2_evaluate_for_user(
                "u", _upload(VALID_WEBM_HEADER), "Topic", "[]"
            ))
    assert exc.value.status_code == 413
    transcribe.assert_not_called()


def test_timeout_cleans_temporary_file(tmp_path):
    def slow_transcribe(_path):
        time.sleep(0.1)
        return "late"

    old_timeout = main.PART2_PROVIDER_TIMEOUT_SECONDS
    old_tempdir = main.tempfile.tempdir
    main.PART2_PROVIDER_TIMEOUT_SECONDS = 0.01
    main.tempfile.tempdir = str(tmp_path)
    try:
        with patch.object(main, "_media_duration_seconds", return_value=30.0), \
             patch.object(main, "_transcribe_audio_file", side_effect=slow_transcribe):
            with pytest.raises(HTTPException) as exc:
                _run(main._part2_evaluate_for_user(
                    "u", _upload(VALID_WEBM_HEADER), "Topic", "[]"
                ))
        assert exc.value.status_code == 504
        assert list(tmp_path.iterdir()) == []
    finally:
        main.PART2_PROVIDER_TIMEOUT_SECONDS = old_timeout
        main.tempfile.tempdir = old_tempdir


def test_parallel_submissions_for_same_user_are_rejected():
    async def scenario():
        entered = asyncio.Event()
        release = asyncio.Event()

        async def holder():
            async with main._part2_user_slot("same-user"):
                entered.set()
                await release.wait()

        task = asyncio.create_task(holder())
        await entered.wait()
        try:
            with pytest.raises(HTTPException) as exc:
                async with main._part2_user_slot("same-user"):
                    pass
            assert exc.value.status_code == 409
        finally:
            release.set()
            await task
        assert "same-user" not in main._part2_active_users

    _run(scenario())
