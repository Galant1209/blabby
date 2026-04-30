#!/usr/bin/env python3
"""
Sprint 2B verify — quota gating + tracking endpoints smoke test.

Usage:
  SSL_CERT_FILE=$(python3 -m certifi) \
   BLABBY_ACCESS_TOKEN='access_token' \
   DEV_BYPASS_SECRET='dev_secret' \
   python3 scripts/verify_sprint2b.py
"""
import json
import os
import sys
import urllib.request
import urllib.error

API_BASE = "https://blabby-backend.onrender.com"

DUMMY_TRANSCRIPT = (
    "Last weekend I went to a very very very nice place with a friend. "
    "It was very good and very fun. We had a very nice time."
)


def make_json_request(path, method="GET", data=None, token=None):
    url = f"{API_BASE}{path}"
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    body = None
    if data is not None:
        body = json.dumps(data).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def post_drill(token, dev_secret, drill_tag):
    boundary = "----WebKitFormBoundary7MA4YWxkTrZu0gW"
    parts = []
    fields = {
        "topic": "General",
        "question": "Describe a place you visited recently.",
        "history": "[]",
        "mode": "drill",
        "drill_tag": drill_tag,
        "text_override": DUMMY_TRANSCRIPT,
        "dev_bypass_secret": dev_secret,
    }
    for name, value in fields.items():
        parts.append(f"--{boundary}\r\n")
        parts.append(f'Content-Disposition: form-data; name="{name}"\r\n\r\n')
        parts.append(f"{value}\r\n")
    parts.append(f"--{boundary}--\r\n")
    body = "".join(parts).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": f"multipart/form-data; boundary={boundary}",
    }
    req = urllib.request.Request(
        f"{API_BASE}/process", data=body, headers=headers, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raw = e.read().decode("utf-8")
        try:
            return e.code, json.loads(raw)
        except json.JSONDecodeError:
            return e.code, raw


def main():
    token = os.environ.get("BLABBY_ACCESS_TOKEN", "").strip()
    dev_secret = os.environ.get("DEV_BYPASS_SECRET", "").strip()
    if not token:
        print("ERROR: BLABBY_ACCESS_TOKEN env var not set")
        return 1
    if not dev_secret:
        print("ERROR: DEV_BYPASS_SECRET env var not set")
        return 1

    print("=" * 60)
    print("Sprint 2B Verify - START")
    print("=" * 60)

    print("\n[Test 1] GET /api/drill/check_quota - initial state")
    status, body = make_json_request("/api/drill/check_quota", token=token)
    print(f"  HTTP: {status}")
    print(f"  Body: {json.dumps(body, indent=2, ensure_ascii=False)}")
    if status != 200:
        print("  FAIL: check_quota should return 200")
        return 1
    expected_keys = {"drill_count", "free_quota", "remaining", "is_pro",
                     "should_upgrade", "quota_resets_at", "window_days"}
    actual_keys = set(body.keys()) if isinstance(body, dict) else set()
    missing = expected_keys - actual_keys
    if missing:
        print(f"  FAIL: missing keys {missing}")
        return 1
    print(f"  PASS - drill_count={body['drill_count']}, "
          f"remaining={body['remaining']}, "
          f"window_days={body['window_days']}")

    if body["remaining"] < 3:
        print(f"\nABORT: remaining={body['remaining']} < 4, cannot run 4 drills")
        print("  Clean drill_usage table in Supabase first:")
        print("    DELETE FROM drill_usage;")
        return 1

    for i in range(1, 5):
        print(f"\n[Test {1+i}] Drill #{i} - POST /process mode=drill")
        status, body = post_drill(token, dev_secret, "weak_vocab")
        print(f"  HTTP: {status}")

        if i <= 3:
            if status != 200:
                print(f"  FAIL: drill #{i} should return 200, got {status}")
                print(f"  Body: {json.dumps(body, indent=2, ensure_ascii=False)[:500]}")
                return 1
            drill_score = body.get("drill_score") if isinstance(body, dict) else None
            if not drill_score:
                print("  FAIL: drill_score missing")
                return 1
            evidence = drill_score.get("evidence", {})
            print(f"  PASS - drill_score.score={drill_score.get('score')}, "
                  f"evidence_keys={list(evidence.keys()) if isinstance(evidence, dict) else 'N/A'}")
        else:
            if status != 403:
                print(f"  FAIL: drill #4 should return 403, got {status}")
                print(f"  Body: {json.dumps(body, indent=2, ensure_ascii=False)[:500]}")
                return 1
            detail = body.get("detail") if isinstance(body, dict) else None
            if not isinstance(detail, dict):
                print(f"  WARN: detail is not dict: {detail}")
            else:
                if detail.get("error") != "quota_exceeded":
                    print(f"  FAIL: error should be 'quota_exceeded', got {detail.get('error')}")
                    return 1
                if detail.get("redirect") != "/upgrade":
                    print(f"  FAIL: redirect should be '/upgrade', got {detail.get('redirect')}")
                    return 1
            print("  PASS - quota gate triggered, 403 + redirect=/upgrade")

    print("\n[Test 6] POST /api/track/upgrade_page_view")
    status, body = make_json_request(
        "/api/track/upgrade_page_view", method="POST", data={}, token=token
    )
    print(f"  HTTP: {status}")
    if status == 200:
        print("  PASS")
    else:
        print(f"  FAIL: should return 200, got {status}")

    print("\n[Test 7] POST /api/track/upgrade_interest")
    status, body = make_json_request(
        "/api/track/upgrade_interest", method="POST",
        data={"email": "verify_test@example.com",
              "timestamp": "2026-04-29T11:00:00Z"},
        token=token,
    )
    print(f"  HTTP: {status}")
    if status == 200:
        print("  PASS")
    else:
        print(f"  FAIL: should return 200, got {status}")

    print("\n" + "=" * 60)
    print("Sprint 2B Verify - DONE")
    print("=" * 60)
    print("\nCleanup test data:")
    print("  Supabase SQL Editor: DELETE FROM drill_usage;")
    return 0


if __name__ == "__main__":
    sys.exit(main())
