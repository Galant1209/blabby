"""Contracts for the optional payment-funnel capture seam."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import telemetry


def test_capture_is_noop_without_the_optional_capture_key(monkeypatch):
    monkeypatch.delenv("POSTHOG_CAPTURE_API_KEY", raising=False)
    monkeypatch.setenv("POSTHOG_PERSONAL_API_KEY", "query-only-secret")

    with patch.object(telemetry.requests, "post") as post:
        assert telemetry.capture_event(
            "checkout_order_created",
            distinct_id="user-123",
            properties={"order_id": "20260904ABC"},
        ) is False
    post.assert_not_called()


def test_capture_posts_only_safe_scalar_properties(monkeypatch):
    monkeypatch.setenv("POSTHOG_CAPTURE_API_KEY", "capture-key")
    monkeypatch.setenv("POSTHOG_HOST", "https://posthog.example.test/")
    response = SimpleNamespace(status_code=200)

    properties = {
        "order_id": "20260904ABC",
        "amount_twd": 199,
        "currency": "TWD",
        "payment_provider": "ecpay",
        "subscription_status": "active",
        "source": "vocab_limit",
        "email": "learner@example.test",
        "access_token": "Bearer secret",
        "raw_callback": {"TradeAmt": "199"},
    }
    with patch.object(telemetry.requests, "post", return_value=response) as post:
        assert telemetry.capture_event(
            "subscription_activated",
            distinct_id="user-123",
            properties=properties,
        ) is True

    post.assert_called_once_with(
        "https://posthog.example.test/capture/",
        json={
            "api_key": "capture-key",
            "event": "subscription_activated",
            "distinct_id": "user-123",
            "properties": {
                "order_id": "20260904ABC",
                "amount_twd": 199,
                "currency": "TWD",
                "payment_provider": "ecpay",
                "subscription_status": "active",
                "source": "vocab_limit",
            },
        },
        timeout=1.5,
    )


def test_capture_failure_is_fail_open(monkeypatch):
    monkeypatch.setenv("POSTHOG_CAPTURE_API_KEY", "capture-key")
    with patch.object(
        telemetry.requests,
        "post",
        side_effect=RuntimeError("network unavailable"),
    ):
        assert telemetry.capture_event(
            "subscription_activated",
            distinct_id="user-123",
            properties={"order_id": "20260904ABC"},
        ) is False
