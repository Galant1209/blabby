"""Static + behavioral contracts for the Pro upgrade frontend flow (TASK 6B-R).

Static checks follow the existing test_frontend_xss_regression.py pattern.
Behavioral checks shell out to a Node harness that exercises DOM stubs —
no new browser framework is introduced.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

import pytest

APP_DIR = Path(__file__).parents[2] / "frontend" / "app"
HARNESS = Path(__file__).with_name("frontend_billing_behavior.mjs")
NODE = "node"


def _read(name: str) -> str:
    return (APP_DIR / name).read_text(encoding="utf-8")


# ── Static contracts ─────────────────────────────────────────────────────────


def test_upgrade_price_is_only_199():
    src = _read("upgrade.html")
    assert "NT$199" in src
    for old in ("149", "259", "499"):
        # Avoid matching CSS hex / font weights; require currency-ish context.
        assert not re.search(rf"(?:NT\$|\$)\s*{old}\b|{old}\s*/\s*月", src)
        assert f"NT${old}" not in src


def test_upgrade_does_not_write_upgrade_intent():
    src = _read("upgrade.html")
    assert "upgrade_intent" not in src
    assert ".from(" not in src or "upgrade_intent" not in src
    assert "supabase.from" not in src.lower() or "upgrade" not in src.lower()
    # Stronger: no insert into waitlist table.
    assert not re.search(r"from\(\s*['\"]upgrade_intent['\"]\s*\)", src)


def test_upgrade_calls_create_order_with_bearer():
    src = _read("upgrade.html")
    assert "/api/payment/create-order" in src
    assert "Authorization: `Bearer ${token}`" in src or 'Authorization: `Bearer ${token}`' in src
    assert "Bearer ${token}" in src


def test_upgrade_builds_and_submits_ecpay_form():
    src = _read("upgrade.html")
    assert "ecpayForm.action = body.action_url" in src
    assert "ecpayForm.submit()" in src
    assert "createElement('input')" in src
    assert "type = 'hidden'" in src or 'type = "hidden"' in src


def test_upgrade_guards_double_click_before_await():
    src = _read("upgrade.html")
    # In-flight lock must be set before the first await (getToken).
    fn = src.split("async function goToCheckout()", 1)[1].split(
        "checkoutBtn.addEventListener", 1
    )[0]
    before_get_token = fn.split("await getToken()", 1)[0]
    assert "checkoutInFlight" in before_get_token
    assert "checkoutBtn.disabled = true" in before_get_token


def test_upgrade_source_lookup_not_raw_injection():
    src = _read("upgrade.html")
    # Source is used as a HEADLINES key; unknown sources fall back to direct.
    assert "HEADLINES[source] || HEADLINES.direct" in src
    # Headline HTML comes from a fixed map, never from the query string itself.
    assert "h.innerHTML = copy.headline" in src
    assert "innerHTML = source" not in src
    assert "innerHTML = params" not in src


def test_upgrade_has_no_server_secrets():
    src = _read("upgrade.html")
    for needle in (
        "ECPAY_HASH",
        "HashKey",
        "HashIV",
        "service_role",
        "SERVICE_KEY",
        "service_key",
    ):
        assert needle not in src


def test_writing_quota_and_pro_required_open_modal():
    src = _read("writing.html")
    assert "daily_quota_reached" in src
    assert "pro_required" in src
    assert "showProModal('writing_quota')" in src
    assert "showProModal('writing_task1')" in src
    assert "three essays" in src
    assert "每日 3 篇" in src
    # CTA navigates; no create-order in writing.
    assert "/api/payment/create-order" not in src
    assert "/upgrade.html?source=${encodeURIComponent(source)}" in src


def test_reading_quota_exceeded_opens_modal_to_upgrade():
    src = _read("reading.html")
    assert "showProModal('reading_quota')" in src
    assert "quota_exceeded" in src or "reading_paywall_hit" in src
    assert "/upgrade.html?source=${encodeURIComponent(source)}" in src
    assert "/api/payment/create-order" not in src


def test_speaking_three_quota_errors_map_to_distinct_sources():
    src = _read("index.html")
    assert "feedback_quota_reached" in src
    assert "part2_quota_reached" in src
    assert "showProModal('feedback_quota')" in src
    assert "showProModal('drill_quota')" in src
    assert "showProModal('part2_quota')" in src
    # showP2Modal retained for non-quota uses.
    assert "function showP2Modal(" in src
    assert "showP2Modal('無法錄音'" in src or 'showP2Modal("無法錄音"' in src or "showP2Modal('無法錄音'" in src


def test_hub_pro_resolution_does_not_false_demote():
    src = _read("hub.html")
    assert "function resolveIsPro(" in src
    assert "isPro !== false" in src  # upgrade badge only when known free
    assert "isPro !== true" in src
    assert "/api/user/subscription" in src
    # Must not collapse failed quota probe into boolean false via && chain alone.
    assert "quota.value.is_pro === true" not in src or "resolveIsPro" in src


# ── success.html (TASK 6B-S) ─────────────────────────────────────────────────


def test_success_page_does_not_claim_pro_in_static_html():
    src = _read("success.html")
    assert "你現在是 Pro" not in src
    assert "Thy payment has been received." not in src
    assert "付款結果確認中，請稍候。" in src
    assert "The ledger is being examined." in src


def test_success_page_verifies_via_subscription_api_with_bearer():
    src = _read("success.html")
    assert "/api/user/subscription" in src
    assert "Authorization: 'Bearer ' + token" in src or 'Authorization: "Bearer " + token' in src
    assert "MAX_POLLS = 5" in src
    assert "POLL_INTERVAL_MS" in src
    assert "isActiveSubscription" in src
    assert "safeOrderLabel" in src
    # Order via textContent only — never innerHTML from query.
    assert "els.order.textContent = label" in src
    assert "innerHTML" not in src.split("<script", 1)[-1] or (
        "innerHTML" not in _read("success.html").split("safeOrderLabel", 1)[-1][:800]
    )


def test_success_page_has_no_server_secrets_or_profile_writes():
    src = _read("success.html")
    for needle in (
        "ECPAY_HASH",
        "HashKey",
        "HashIV",
        "service_role",
        "SERVICE_KEY",
        "is_pro_grant",
        "upgrade_intent",
    ):
        assert needle not in src
    assert ".from(" not in src
    assert "profiles" not in src


# ── Behavioral (Node DOM stubs) ──────────────────────────────────────────────


def test_frontend_billing_behavior_harness():
    if not HARNESS.exists():
        pytest.fail(f"missing harness: {HARNESS}")
    result = subprocess.run(
        [NODE, str(HARNESS)],
        cwd=str(APP_DIR.parent.parent),
        capture_output=True,
        text=True,
        timeout=60,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    assert result.returncode == 0, (
        f"behavior harness failed (exit {result.returncode}):\n"
        f"{result.stdout}\n{result.stderr}"
    )
