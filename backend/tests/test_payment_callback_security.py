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
from urllib.parse import urlencode

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
            [{"user_id": USER_ID, "amount": main.ecpay.PRO_MONTHLY_TWD,
              "status": "pending", "expires_at": None}]
            if subscriptions_data is None else subscriptions_data
        )
        self.profiles_data = (
            [{"id": USER_ID}] if profiles_data is None else profiles_data
        )
        self.writes = []
        self.seen_keys = set()
        self.ledger_inserts = 0
        self.last_event_key = None
        self.rpc_calls = []
        self.activations = []
        self.accepted_events = []
        self.activated_orders = set()

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

    def rpc(self, name, params):
        self.rpc_calls.append((name, params))
        assert name == "accept_ecpay_payment"

        def execute():
            order = params["p_merchant_trade_no"]
            key = (order, params["p_total_success_times"], "ecpay_callback")
            if order in self.activated_orders or key in self.seen_keys:
                return _Resp([{"result": "duplicate"}])
            self.activated_orders.add(order)
            self.seen_keys.add(key)
            self.accepted_events.append(params)
            self.activations.append(params)
            return _Resp([{"result": "activated"}])

        return MagicMock(execute=execute)


# ── request plumbing ─────────────────────────────────────────────────────
def _signed_form(**overrides):
    params = {
        "MerchantID": "3002607",
        "MerchantTradeNo": ORDER_ID,
        "RtnCode": "1",
        "RtnMsg": "Succeeded",
        "TradeAmt": str(main.ecpay.PRO_MONTHLY_TWD),
        "CustomField1": USER_ID,
        "PaymentDate": "2026/07/26 10:00:00",
        "EncryptType": "1",
    }
    params.update(overrides)
    params["CheckMacValue"] = main._ecpay_check_mac_value(params, HASH_KEY, HASH_IV)
    return params


def _assert_ack(response):
    """Every durably-recorded outcome answers ECPay with exactly 1|OK."""
    assert response.body == b"1|OK"
    assert response.status_code == 200
    assert response.media_type == "text/plain"


# payment_callback now parses the raw body itself (ecpay.parse_ecpay_form)
# rather than request.form() — Starlette's urlencoded parser decodes bytes as
# latin-1 before unquoting, which mangles ECPay's literal (non-percent-encoded)
# UTF-8 RtnMsg into mojibake and breaks CheckMacValue for any non-ASCII field.
# Test fixtures build the same raw wire bytes real ECPay sends, keyed through
# request.body() to match.
def _form_body(params: dict) -> bytes:
    return urlencode(params).encode("utf-8")


def _call_callback(form_params, fake):
    request = MagicMock()
    request.body = AsyncMock(return_value=_form_body(form_params))
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "ECPAY_HASH_KEY", HASH_KEY), \
         patch.object(main, "ECPAY_HASH_IV", HASH_IV), \
         patch.object(main.ecpay, "_CONFIG", main.ecpay.EcpayConfig(
             merchant_id="3002607",
             env="stage",
             action_url="https://payment-stage.ecpay.com.tw/Cashier/AioCheckOut/V5",
             return_url="https://api.example.com/api/payment/callback",
             client_back_url="https://app.example.com/success.html",
             order_result_url="https://api.example.com/api/payment/return",
         )), \
         patch.object(main.ecpay, "_CONFIG_ERROR", None), \
         patch.object(main, "supabase_admin", fake):
        return asyncio.run(main.payment_callback(request=request))


# ── gate 1: fail closed without credentials ──────────────────────────────
def test_missing_merchant_credentials_fail_closed_503():
    request = MagicMock()
    request.body = AsyncMock(return_value=_form_body(_signed_form()))
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
    form = _signed_form(TradeAmt=str(main.ecpay.PRO_MONTHLY_TWD))
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
    _assert_ack(first)

    for _ in range(4):
        _assert_ack(_call_callback(form, fake))

    profs = [w for w in fake.entitlement_writes() if w[0] == "profiles"]
    assert len(fake.activations) == 1
    assert len(fake.accepted_events) == 1
    assert profs == [], "entitlement is the subscriptions window, not profiles.is_pro"
    assert len(fake.seen_keys) == 1, "only one distinct ledger key expected"


def test_changed_success_counter_cannot_extend_a_one_off_order():
    """No auto-renewal: one MerchantTradeNo can activate exactly one period."""
    fake = _FakeSupabase()
    _call_callback(_signed_form(TotalSuccessTimes="1"), fake)
    second = _call_callback(_signed_form(TotalSuccessTimes="2"), fake)
    _assert_ack(second)
    assert len(fake.activations) == 1
    assert len(fake.accepted_events) == 1


# ── gate 4: empty row → 500 ──────────────────────────────────────────────
def test_unknown_subscription_is_acknowledged_without_entitlement():
    fake = _FakeSupabase(subscriptions_data=[])
    _assert_ack(_call_callback(_signed_form(), fake))
    assert fake.rpc_calls == []


def test_subscription_without_a_user_id_is_acknowledged_without_entitlement():
    """A matched row that carries no owner cannot be turned into entitlement."""
    fake = _FakeSupabase(subscriptions_data=[{"user_id": None}])
    _assert_ack(_call_callback(_signed_form(), fake))
    assert fake.rpc_calls == []


def test_successful_callback_never_writes_profiles_is_pro():
    """Replaces the old profiles-empty-row test: that write no longer exists.

    Pro is the subscriptions time window read by is_user_pro(). Setting the
    bare profiles.is_pro boolean here would hand the buyer a flag that outlives
    the 30 days they paid for, which is the leak the window closes.
    """
    fake = _FakeSupabase()
    _assert_ack(_call_callback(_signed_form(), fake))
    assert [w for w in fake.writes if w[0] == "profiles"] == []
    assert len(fake.activations) == 1


def test_failed_payment_does_not_consume_accepted_success_identity():
    fake = _FakeSupabase()
    response = _call_callback(_signed_form(RtnCode="0", RtnMsg="Declined"), fake)
    _assert_ack(response)
    assert fake.entitlement_writes() == [], "a declined payment must grant nothing"
    assert fake.accepted_events == []
    assert fake.seen_keys == set()


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
