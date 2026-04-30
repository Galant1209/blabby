"""
Sprint 2A verification: hit deployed /process with mode=drill for both
drill_tags and report whether the v6 evidence shapes land in the response.

Usage:
    BLABBY_ACCESS_TOKEN=<user-JWT> python3 scripts/verify_drill_v6.py

To grab BLABBY_ACCESS_TOKEN:
  1. Open https://blabby.vercel.app/ in your browser, sign in
  2. DevTools → Application → Local Storage → blabby.vercel.app
  3. Find a key like  sb-<project>-auth-token
     Value is JSON with an  access_token  field — copy that string

Or run in the JS console on a logged-in tab:
    JSON.parse(localStorage.getItem(
      Object.keys(localStorage).find(k => k.startsWith('sb-'))
    )).access_token

DEV_BYPASS_SECRET is read from backend/.env so Whisper is skipped and
we feed a deterministic text_override per drill tag — no audio needed.
"""

import json
import os
import sys
from pathlib import Path
from urllib.parse import urlparse
import urllib.request

API_BASE = "https://blabby-backend.onrender.com"

# Sample transcripts chosen to exercise both drill axes:
SAMPLES = {
    "weak_vocab": (
        "I think the British rock is very good and very interesting. "
        "I love it very much because I like it very much."
    ),
    "lack_detail": (
        "It was nice. We had fun. The place was good and the people were friendly. "
        "I had a good time."
    ),
}


def load_dev_bypass_secret() -> str:
    env_path = Path(__file__).resolve().parent.parent / "backend" / ".env"
    if not env_path.exists():
        return ""
    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        if k.strip() == "DEV_BYPASS_SECRET":
            return v.strip().strip('"').strip("'")
    return ""


def post_drill(token: str, dev_bypass: str, drill_tag: str) -> dict:
    boundary = "----blabby-verify-drill-boundary"
    fields = {
        "level": "Band 5",
        "topic": "Music",
        "question": "What kind of music do you enjoy most?",
        "history": "[]",
        "text_override": SAMPLES[drill_tag],
        "dev_bypass_secret": dev_bypass,
        "mode": "drill",
        "drill_tag": drill_tag,
    }
    parts: list[bytes] = []
    for name, val in fields.items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(
            f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode()
        )
        parts.append(val.encode("utf-8"))
        parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    body = b"".join(parts)

    req = urllib.request.Request(
        f"{API_BASE}/process",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": f"multipart/form-data; boundary={boundary}",
            "Content-Length": str(len(body)),
        },
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return {
                "status": resp.status,
                "body": json.loads(resp.read().decode("utf-8")),
            }
    except urllib.error.HTTPError as e:
        return {
            "status": e.code,
            "body": json.loads(e.read().decode("utf-8") or "{}"),
        }


def report(drill_tag: str, result: dict) -> None:
    print(f"\n=== drill_tag={drill_tag} | HTTP {result['status']} ===")
    body = result.get("body") or {}
    if result["status"] != 200:
        print("ERROR body:", json.dumps(body, ensure_ascii=False, indent=2))
        return
    drill_score = body.get("drill_score")
    if drill_score is None:
        print("FAIL: response has no drill_score field")
        return
    print("axis:            ", drill_score.get("axis"))
    print("score:           ", drill_score.get("score"))
    print("threshold_passed:", drill_score.get("threshold_passed"))
    print("feedback:        ", drill_score.get("feedback"))
    evidence = drill_score.get("evidence")
    if evidence is None:
        print("FAIL: drill_score.evidence missing")
        return
    print("evidence keys:   ", sorted(evidence.keys()))
    expected = {
        "weak_vocab": {"safe_words_found", "b2_plus_found"},
        "lack_detail": {
            "time_dimensions_found",
            "place_dimensions_found",
            "number_dimensions_found",
            "sense_dimensions_found",
            "person_dimensions_found",
        },
    }[drill_tag]
    actual = set(evidence.keys())
    missing = expected - actual
    extra = actual - expected
    if missing:
        print("FAIL: missing evidence keys:", sorted(missing))
    if extra:
        print("WARN: unexpected extra keys:", sorted(extra))
    if not missing and not extra:
        print("PASS: evidence shape matches v6 spec for", drill_tag)
    for k in sorted(actual):
        v = evidence[k]
        if isinstance(v, list):
            print(f"  {k} ({len(v)}):", v)
        else:
            print(f"  {k}:", v, f"(type={type(v).__name__})")


def main() -> int:
    token = os.environ.get("BLABBY_ACCESS_TOKEN", "").strip()
    if not token:
        print("ERROR: BLABBY_ACCESS_TOKEN env var is empty.")
        print("Grab it from your logged-in Supabase session — see header doc.")
        return 1
    dev_bypass = load_dev_bypass_secret()
    if not dev_bypass:
        print("WARN: DEV_BYPASS_SECRET not found in backend/.env — "
              "Whisper will run on whatever audio path the backend hits.")
    for tag in ("weak_vocab", "lack_detail"):
        result = post_drill(token, dev_bypass, tag)
        report(tag, result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
