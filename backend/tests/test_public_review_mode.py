"""Focused contracts for the temporary ECPay public review surface."""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import main


APP_DIR = Path(__file__).parents[2] / "frontend" / "app"
MIGRATION = Path(__file__).parents[2] / "supabase" / "migrations" / "20260813_anonymous_process_quota.sql"

VALID_CORRECTION = {
    "correction": {
        "quoted": "I very like it",
        "why_it_hurts": "這個強調方式不自然。",
        "better_phrasing_en": "I really like it",
        "better_phrasing_zh": "我真的很喜歡",
        "next_task": "把 very 換成 really 再說一次。",
    },
    "on_topic": True,
    "tag": "grammar_minor",
    "progress_note": "",
}


class _QuotaStore:
    def __init__(self):
        self.used: dict[str, int] = {}

    def rpc(self, name, params):
        visitor = params["p_visitor_hash"]
        limit = params["p_limit"]

        def execute():
            used = self.used.get(visitor, 0)
            if name == "consume_anonymous_process_quota" and used < limit:
                used += 1
                self.used[visitor] = used
            return SimpleNamespace(data={
                "allowed": used < limit if name == "get_anonymous_process_quota" else used <= limit,
                "used": min(used, limit),
                "remaining": max(limit - used, 0),
            })

        return SimpleNamespace(execute=execute)


def _request(host: str = "203.0.113.8"):
    req = MagicMock()
    req.client = SimpleNamespace(host=host)
    return req


def _run_process(store: _QuotaStore, visitor_id: str, *, correction=VALID_CORRECTION):
    kwargs = dict(
        request=_request(),
        audio=None,
        level="Band 5",
        topic="Hobbies",
        question="What do you do in your free time?",
        history="[]",
        text_override="I very like sports.",
        dev_bypass_secret="test-secret",
        mode="",
        drill_tag="",
        previous_transcript="",
        retry_of="",
        authorization=None,
        x_blabby_visitor_id=visitor_id,
    )
    run_claude = MagicMock(return_value=dict(correction))
    with patch.dict(os.environ, {
        "PUBLIC_REVIEW_MODE": "true",
        "ANONYMOUS_ID_HASH_SECRET": "unit-test-hmac-secret",
        "DEV_BYPASS_SECRET": "test-secret",
    }), patch.object(main.limiter, "enabled", False), \
         patch.object(main, "supabase_admin", store), \
         patch.object(main, "_enforce_anonymous_rate_limit"), \
         patch.object(main, "pick_next_question", return_value="Next question?"), \
         patch.object(main, "run_claude", run_claude):
        return asyncio.run(main.process(**kwargs))


def test_anonymous_requests_one_through_ten_succeed_then_eleven_is_rejected():
    store = _QuotaStore()
    visitor = str(uuid.uuid4())
    for expected_used in range(1, 11):
        result = _run_process(store, visitor)
        assert result["anonymous_trial"] == {
            "limit": 10,
            "used": expected_used,
            "remaining": 10 - expected_used,
        }

    with pytest.raises(HTTPException) as excinfo:
        _run_process(store, visitor)
    assert excinfo.value.status_code == 403
    assert excinfo.value.detail["error"] == "anonymous_quota_exceeded"


def test_failed_process_does_not_consume_anonymous_quota():
    store = _QuotaStore()
    visitor = str(uuid.uuid4())
    with patch.dict(os.environ, {
        "PUBLIC_REVIEW_MODE": "true",
        "ANONYMOUS_ID_HASH_SECRET": "unit-test-hmac-secret",
        "DEV_BYPASS_SECRET": "test-secret",
    }), patch.object(main.limiter, "enabled", False), \
         patch.object(main, "supabase_admin", store), \
         patch.object(main, "_enforce_anonymous_rate_limit"), \
         patch.object(main, "pick_next_question", return_value="Next question?"), \
         patch.object(main, "run_claude", side_effect=RuntimeError("provider failed")):
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(main.process(
                request=_request(), audio=None, level="Band 5", topic="Hobbies",
                question="What do you do?", history="[]",
                text_override="I play sports.", dev_bypass_secret="test-secret",
                mode="", drill_tag="", previous_transcript="", retry_of="",
                authorization=None, x_blabby_visitor_id=visitor,
            ))
    assert excinfo.value.status_code == 500
    assert store.used == {}


def test_different_visitors_have_separate_lifetime_quota():
    store = _QuotaStore()
    first = _run_process(store, str(uuid.uuid4()))
    second = _run_process(store, str(uuid.uuid4()))
    assert first["anonymous_trial"]["used"] == 1
    assert second["anonymous_trial"]["used"] == 1
    assert len(store.used) == 2


def test_spoofing_visitor_id_does_not_bypass_ip_rate_limit():
    main._anonymous_rate_buckets.clear()
    ip_hash = "a" * 64
    for timestamp in (1.0, 2.0, 3.0):
        main._enforce_anonymous_rate_limit(ip_hash, now=timestamp)
    with pytest.raises(HTTPException) as excinfo:
        main._enforce_anonymous_rate_limit(ip_hash, now=4.0)
    assert excinfo.value.status_code == 429
    assert excinfo.value.detail["error"] == "anonymous_rate_limit_exceeded"


def test_identity_hashes_ip_and_does_not_trust_forwarded_header():
    req = _request("198.51.100.42")
    req.headers = {"x-forwarded-for": "1.2.3.4"}
    with patch.dict(os.environ, {"ANONYMOUS_ID_HASH_SECRET": "test-secret"}):
        visitor_hash, ip_hash = main._anonymous_identity(req, str(uuid.uuid4()))
        spoofed_hash = main._anonymous_hash("1.2.3.4", "ip")
    assert len(visitor_hash) == len(ip_hash) == 64
    assert "198.51.100.42" not in ip_hash
    assert ip_hash != spoofed_hash


def test_review_mode_off_restores_process_auth_gate():
    with patch.dict(os.environ, {"PUBLIC_REVIEW_MODE": "false"}), \
         patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", side_effect=HTTPException(status_code=401)) as verify:
        with pytest.raises(HTTPException) as excinfo:
            asyncio.run(main.process(
                request=_request(), audio=None, level="Band 5", topic="Hobbies",
                question="What do you do?", history="[]", text_override="",
                dev_bypass_secret="", mode="", drill_tag="",
                previous_transcript="", retry_of="", authorization=None,
                x_blabby_visitor_id=None,
            ))
    assert excinfo.value.status_code == 401
    verify.assert_called_once_with(None)


def test_public_pages_show_price_term_refund_and_real_contact():
    index = (APP_DIR / "index.html").read_text(encoding="utf-8")
    upgrade = (APP_DIR / "upgrade.html").read_text(encoding="utf-8")
    config = (APP_DIR / "config.js").read_text(encoding="utf-8")
    for source in (index, upgrade):
        assert "Blabby Pro" in source
        assert "NT$199 / 30 天" in source
        assert "不自動續訂" in source
        assert "退款申請" in source
        assert "14–21 個工作天" in source
        assert "galant19951209@gmail.com" in source
    assert "PUBLIC_REVIEW_MODE" in index
    assert "publicReviewMode: true" in config
    assert "X-Blabby-Visitor-ID" in index
    assert "pro_waitlist" not in index
    assert "Notify me" not in index
    assert "NT$199／月" not in index + upgrade
    assert index.count('id="pro-modal-overlay"') == 1
    assert "#pro-modal-overlay {" not in index


def test_upgrade_stays_public_and_checkout_keeps_authenticated_attribution():
    upgrade = (APP_DIR / "upgrade.html").read_text(encoding="utf-8")
    assert "checkoutBtn.disabled = true" in upgrade  # only in-flight lock
    boot = upgrade.split("(async function boot()", 1)[1]
    assert "checkoutBtn.disabled = true" not in boot
    assert "/?login=1" in upgrade
    assert "/api/payment/create-order" in upgrade
    assert "Bearer ${token}" in upgrade


def test_migration_stores_only_hashes_and_is_not_client_callable():
    sql = MIGRATION.read_text(encoding="utf-8")
    assert "visitor_hash text PRIMARY KEY" in sql
    assert "ip_hash text NOT NULL" in sql
    assert "ENABLE ROW LEVEL SECURITY" in sql
    assert "REVOKE ALL" in sql
    assert "TO service_role" in sql
    assert "raw_ip" not in sql.lower()
    assert "insert into public.profiles" not in sql.lower()
    assert "update public.profiles" not in sql.lower()
    assert "insert into public.subscriptions" not in sql.lower()
    assert "update public.subscriptions" not in sql.lower()
