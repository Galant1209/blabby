"""
Payment callback containment: signature, idempotency, and empty-row handling.

Covers the four gates POST /api/payment/callback must clear before it can move
entitlement, plus the LemonSqueezy replay ledger. Before this, the endpoint had
no auth, no signature, no idempotency and no replay protection, swallowed every
exception and returned 200 unconditionally — any logged-in user could mint an
order id and then upgrade themselves with an unauthenticated form POST.

Hermetic: supabase, credentials and the clock are all patched. No network.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import main


HASH_KEY = "test-hash-key"
HASH_IV = "test-hash-iv"
ORDER_ID = "BLABBY-ABCDEF123456"
USER_ID = "user-abc"


# ── fake supabase ────────────────────────────────────────────────────────
class _Resp:
    def __init__(self, data):
        self.data = data


class _DuplicateKey(Exception):
    """Stand-in for PostgREST's 23505 unique-violation error."""
    code = "23505"
    message = 'duplicate key value violates unique constraint "payment_events_idem_uniq"'


class _FakeTable:
    def __init__(self, owner, name):
        self._owner = owner
        self._name = name
        self._op = None

    def insert(self, row):
        self._op = "insert"
        self._owner.record(self._name, "insert", row)
        return self

    def upsert(self, row, **kw):
        self._op = "upsert"
        self._owner.record(self._name, "upsert", row)
        return self

    def update(self, row):
        self._op = "update"
        self._owner.record(self._name, "update", row)
        return self

    def select(self, *a, **k):
        self._op = "select"
        return self

    def eq(self, *a, **k):
        return self

    def is_(self, *a, **k):
        return self

    def limit(self, *a, **k):
        return self

    def execute(self):
        if self._name == "payment_events" and self._op == "insert":
            self._owner.ledger_inserts += 1
            key = self._owner.last_event_key
            if key in self._owner.seen_keys:
                raise _DuplicateKey()
            self._owner.seen_keys.add(key)
            return _Resp([{"id": "evt-1"}])
        return _Resp(self._owner.response_for(self._name, self._op))


class _FakeSupabase:
    def __init__(self, subscriptions_data=None, profiles_data=None):
        self.subscriptions_data = (
            [{"user_id": USER_ID}] if subscriptions_data is None else subscriptions_data
        )
        self.profiles_data = (
            [{"id": USER_ID}] if profiles_data is None else profiles_data
        )
        self.writes = []
        self.seen_keys = set()
        self.ledger_inserts = 0
        self.last_event_key = None

    def table(self, name):
        return _FakeTable(self, name)

    def record(self, name, op, row):
        self.writes.append((name, op, row))
        if name == "payment_events" and op == "insert":
            self.last_event_key = (
                row.get("merchant_trade_no"),
                row.get("total_success_times"),
                row.get("source"),
            )

    def response_for(self, name, op):
        if name == "subscriptions":
            return self.subscriptions_data
        if name == "profiles":
            return self.profiles_data
        return [{"id": "x"}]

    # helpers the assertions use
    def entitlement_writes(self):
        return [w for w in self.writes
                if w[0] in ("subscriptions", "profiles") and w[1] in ("update", "upsert")]

    def ledger_rows(self):
        return [w for w in self.writes if w[0] == "payment_events" and w[1] == "insert"]


# ── request plumbing ─────────────────────────────────────────────────────
def _signed_form(**overrides):
    params = {
        "MerchantTradeNo": ORDER_ID,
        "RtnCode": "1",
        "RtnMsg": "Succeeded",
        "TradeAmt": "299",
        "PaymentDate": "2026/07/26 10:00:00",
        "EncryptType": "1",
    }
    params.update(overrides)
    params["CheckMacValue"] = main._ecpay_check_mac_value(params, HASH_KEY, HASH_IV)
    return params


def _call_callback(form_params, fake):
    request = MagicMock()
    request.form = AsyncMock(return_value=form_params)
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "ECPAY_HASH_KEY", HASH_KEY), \
         patch.object(main, "ECPAY_HASH_IV", HASH_IV), \
         patch.object(main, "supabase_admin", fake):
        return asyncio.run(main.payment_callback(request=request))


# ── gate 1: fail closed without credentials ──────────────────────────────
def test_missing_merchant_credentials_fail_closed_503():
    request = MagicMock()
    request.form = AsyncMock(return_value=_signed_form())
    fake = _FakeSupabase()
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "ECPAY_HASH_KEY", ""), \
         patch.object(main, "ECPAY_HASH_IV", ""), \
         patch.object(main, "supabase_admin", fake):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(main.payment_callback(request=request))
    assert exc.value.status_code == 503
    assert fake.writes == []


# ── gate 2: signature ────────────────────────────────────────────────────
def test_unsigned_callback_returns_401_and_writes_nothing():
    fake = _FakeSupabase()
    form = _signed_form()
    del form["CheckMacValue"]
    with pytest.raises(HTTPException) as exc:
        _call_callback(form, fake)
    assert exc.value.status_code == 401
    assert fake.writes == [], "a rejected callback must not touch any table"


def test_wrong_signature_returns_401_and_writes_nothing():
    fake = _FakeSupabase()
    form = _signed_form()
    form["CheckMacValue"] = "F" * 64
    with pytest.raises(HTTPException) as exc:
        _call_callback(form, fake)
    assert exc.value.status_code == 401
    assert fake.entitlement_writes() == []
    assert fake.ledger_rows() == []


def test_tampered_field_invalidates_the_signature():
    """Re-signing is the only way in: editing a field after signing must fail."""
    fake = _FakeSupabase()
    form = _signed_form(TradeAmt="299")
    form["TradeAmt"] = "1"          # attacker edits the amount post-signature
    with pytest.raises(HTTPException) as exc:
        _call_callback(form, fake)
    assert exc.value.status_code == 401


def test_unsupported_encrypt_type_fails_closed():
    fake = _FakeSupabase()
    form = _signed_form()
    form["EncryptType"] = "0"       # MD5 variant, not implemented
    with pytest.raises(HTTPException) as exc:
        _call_callback(form, fake)
    assert exc.value.status_code == 401


def test_check_mac_value_is_deterministic_and_order_independent():
    a = main._ecpay_check_mac_value(
        {"B": "2", "A": "1", "C": "3"}, HASH_KEY, HASH_IV)
    b = main._ecpay_check_mac_value(
        {"C": "3", "A": "1", "B": "2"}, HASH_KEY, HASH_IV)
    assert a == b
    assert len(a) == 64 and a == a.upper()


# ── gate 3: idempotency ──────────────────────────────────────────────────
def test_same_merchant_trade_no_five_times_changes_entitlement_once():
    fake = _FakeSupabase()
    form = _signed_form()

    first = _call_callback(form, fake)
    assert first["status"] == "ok"

    for _ in range(4):
        again = _call_callback(form, fake)
        assert again["status"] == "duplicate"

    subs = [w for w in fake.entitlement_writes() if w[0] == "subscriptions"]
    profs = [w for w in fake.entitlement_writes() if w[0] == "profiles"]
    assert len(subs) == 1, f"subscriptions written {len(subs)}x, expected once"
    assert len(profs) == 1, f"profiles written {len(profs)}x, expected once"
    assert len(fake.seen_keys) == 1, "only one distinct ledger key expected"


def test_recurring_period_is_not_swallowed_as_a_duplicate():
    """total_success_times is in the key so month 2 still lands."""
    fake = _FakeSupabase()
    _call_callback(_signed_form(TotalSuccessTimes="1"), fake)
    second = _call_callback(_signed_form(TotalSuccessTimes="2"), fake)
    assert second["status"] == "ok"
    assert len(fake.seen_keys) == 2


# ── gate 4: empty row → 500 ──────────────────────────────────────────────
def test_subscription_update_returning_no_rows_is_500():
    fake = _FakeSupabase(subscriptions_data=[])
    with pytest.raises(HTTPException) as exc:
        _call_callback(_signed_form(), fake)
    assert exc.value.status_code == 500
    assert "no rows" in exc.value.detail.lower()


def test_profile_update_returning_no_rows_is_500():
    fake = _FakeSupabase(profiles_data=[])
    with pytest.raises(HTTPException) as exc:
        _call_callback(_signed_form(), fake)
    assert exc.value.status_code == 500
    assert "no rows" in exc.value.detail.lower()


def test_failed_payment_is_recorded_without_granting_pro():
    fake = _FakeSupabase()
    body = _call_callback(_signed_form(RtnCode="0", RtnMsg="Declined"), fake)
    assert body["status"] == "recorded"
    assert fake.entitlement_writes() == [], "a declined payment must grant nothing"
    assert len(fake.ledger_rows()) == 1, "but it must still be recorded"


# ── unique-violation detection ───────────────────────────────────────────
def test_unique_violation_detection_by_code_and_message():
    assert main._is_unique_violation(_DuplicateKey())

    class _ByMessage(Exception):
        pass
    assert main._is_unique_violation(
        _ByMessage("duplicate key value violates unique constraint"))

    class _Unrelated(Exception):
        pass
    assert not main._is_unique_violation(_Unrelated("connection reset"))


def test_non_duplicate_ledger_failure_surfaces_as_500_not_a_silent_skip():
    fake = _FakeSupabase()
    fake.table = MagicMock(side_effect=RuntimeError("db down"))
    with patch.object(main, "supabase_admin", fake):
        with pytest.raises(HTTPException) as exc:
            main._record_payment_event({"source": "return_url",
                                        "merchant_trade_no": ORDER_ID,
                                        "checkmac_valid": True,
                                        "raw_payload": {}})
    assert exc.value.status_code == 500


# ── LemonSqueezy replay ──────────────────────────────────────────────────
def _ls_call(raw_body: bytes, fake, secret="ls-secret"):
    import hmac as _hmac
    sig = _hmac.new(secret.encode(), raw_body, hashlib.sha256).hexdigest()
    request = MagicMock()
    request.body = AsyncMock(return_value=raw_body)
    request.headers = {"X-Signature": sig}
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "LEMONSQUEEZY_WEBHOOK_SECRET", secret), \
         patch.object(main, "supabase_admin", fake):
        return asyncio.run(main.lemonsqueezy_webhook(request=request))


def _ls_body():
    return json.dumps({
        "meta": {"event_name": "subscription_created"},
        "data": {"id": "sub-1", "attributes": {
            "status": "active",
            "user_email": "buyer@example.com",
            "renews_at": "2026-08-26T00:00:00Z",
        }},
    }).encode()


def test_lemonsqueezy_replay_of_the_same_signed_body_is_dropped():
    fake = _FakeSupabase()
    rpc_result = MagicMock()
    rpc_result.data = USER_ID
    fake.rpc = MagicMock(return_value=MagicMock(execute=lambda: rpc_result))

    body = _ls_body()
    first = _ls_call(body, fake)
    assert first["status"] == "ok"

    second = _ls_call(body, fake)
    assert second["status"] == "duplicate", "an identical signed body is a replay"

    profs = [w for w in fake.writes if w[0] == "profiles" and w[1] == "update"]
    assert len(profs) == 1, "replay must not re-apply the entitlement change"


def test_lemonsqueezy_mirrors_into_subscriptions():
    fake = _FakeSupabase()
    rpc_result = MagicMock()
    rpc_result.data = USER_ID
    fake.rpc = MagicMock(return_value=MagicMock(execute=lambda: rpc_result))

    _ls_call(_ls_body(), fake)
    mirrors = [w for w in fake.writes if w[0] == "subscriptions" and w[1] == "upsert"]
    assert len(mirrors) == 1, "LS revenue must be visible in subscriptions"
    assert mirrors[0][2]["user_id"] == USER_ID
    assert mirrors[0][2]["expires_at"], "the mirror needs an expiry for is_user_pro"


# ── admin extend must not pollute the paid column ────────────────────────
def test_admin_extend_writes_grant_columns_not_is_pro():
    src = main.admin_extend_subscription.__wrapped__.__code__.co_consts
    flat = " ".join(str(c) for c in src)
    assert "is_pro_grant" in flat
    assert "'is_pro'" not in flat and '"is_pro"' not in flat
