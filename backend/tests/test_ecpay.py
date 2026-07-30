"""ECPay signing, order naming, checkout construction and the callback's
wire-level response contract.

The load-bearing test here is test_official_known_answer_vector: it is the only
one that can tell us our CheckMacValue matches ECPay's rather than merely
matching itself. Everything else in this file is downstream of that.

Hermetic: credentials, supabase and the clock are patched or fabricated. The
HashKey/HashIV used below are ECPay's own *published sample* values from the
public documentation page, not merchant credentials.
"""

from __future__ import annotations

import asyncio
import importlib
import os
import re
import string
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

import ecpay
import main


# ── ECPay's published worked example ─────────────────────────────────────
# Source: https://developers.ecpay.com.tw/2902/  「檢查碼機制說明」
# Fetched 2026-07-30. Sample merchant 3002607 from the public docs.
DOC_HASH_KEY = "pwFHCqoQZGmho4w6"
DOC_HASH_IV = "EkRm7iFT261dpevs"
DOC_PARAMS = {
    "ChoosePayment":     "ALL",
    "EncryptType":       "1",
    "ItemName":          "Apple iphone 15",
    "MerchantID":        "3002607",
    "MerchantTradeDate": "2023/03/12 15:30:23",
    "MerchantTradeNo":   "ecpay20230312153023",
    "PaymentType":       "aio",
    "ReturnURL":         "https://www.ecpay.com.tw/receive.php",
    "TotalAmount":       "30000",
    "TradeDesc":         "促銷方案",
}
DOC_EXPECTED = "6C51C9E6888DE861FD62FB1DD17029FC742634498FD813DC43D4243B5685B840"


def test_official_known_answer_vector():
    """Our digest must equal the one ECPay publishes for these exact inputs."""
    assert ecpay.build_check_mac_value(
        DOC_PARAMS, DOC_HASH_KEY, DOC_HASH_IV) == DOC_EXPECTED


def test_known_answer_vector_verifies_end_to_end():
    payload = dict(DOC_PARAMS, CheckMacValue=DOC_EXPECTED)
    assert ecpay.verify_check_mac_value(payload, DOC_HASH_KEY, DOC_HASH_IV)


def test_known_answer_vector_is_reached_through_main_too():
    """main._ecpay_check_mac_value is the name the callback suite patches."""
    assert main._ecpay_check_mac_value(
        DOC_PARAMS, DOC_HASH_KEY, DOC_HASH_IV) == DOC_EXPECTED


# ── .NET url encoding, character by character ────────────────────────────
@pytest.mark.parametrize("literal", ["-", "_", ".", "!", "*", "(", ")"])
def test_urlencode_leaves_the_seven_dotnet_literals_alone(literal):
    """quote_plus escapes these; HttpUtility.UrlEncode does not."""
    assert ecpay._ecpay_urlencode(literal) == literal


def test_urlencode_maps_space_to_plus():
    assert ecpay._ecpay_urlencode("a b") == "a+b"


def test_urlencode_lowercases_percent_escapes():
    # '@' -> %40, '/' -> %2F which must come back lowercase as %2f
    assert ecpay._ecpay_urlencode("/") == "%2f"
    assert ecpay._ecpay_urlencode("@") == "%40"


def test_urlencode_still_escapes_the_dangerous_separators():
    assert ecpay._ecpay_urlencode("&") == "%26"
    assert ecpay._ecpay_urlencode("<") == "%3c"
    assert ecpay._ecpay_urlencode("=") == "%3d"


def test_tilde_is_left_literal_exactly_as_ecpays_own_sdk_leaves_it():
    """Documenting a real divergence from HttpUtility.UrlEncode.

    .NET escapes '~' as %7E; Python's quote_plus has it in _ALWAYS_SAFE and
    leaves it alone, and no replacement pair puts it back. ECPay's official
    Python SDK computes the digest as quote_plus(s, safe='-_.!*()'), whose safe
    set is *added* to _ALWAYS_SAFE — so the official SDK leaves '~' literal too.
    Matching ECPay's own implementation is the bar that matters, and this test
    pins the behaviour so a future "fix" toward strict .NET parity is a
    deliberate, visible decision rather than an accident.
    """
    assert ecpay._ecpay_urlencode("~") == "~"


def test_three_of_the_seven_replacements_are_already_no_ops_in_python():
    """'-', '_' and '.' never get escaped by quote_plus in the first place.

    The pairs are kept because the ECPay docs list all seven and dropping the
    inert three would make the implementation harder to diff against the spec.
    """
    for char in "-_.":
        assert ecpay._ecpay_urlencode(char) == char


def test_urlencode_handles_utf8():
    assert ecpay._ecpay_urlencode("促銷方案") == (
        "%e4%bf%83%e9%8a%b7%e6%96%b9%e6%a1%88")


# ── verification semantics ───────────────────────────────────────────────
def test_unknown_future_fields_are_included_not_whitelisted():
    """ECPay adds fields over time; every one of them is signed."""
    params = dict(DOC_PARAMS, SomeFieldEcpayAddsIn2027="x")
    signed = dict(params,
                  CheckMacValue=ecpay.build_check_mac_value(
                      params, DOC_HASH_KEY, DOC_HASH_IV))
    assert ecpay.verify_check_mac_value(signed, DOC_HASH_KEY, DOC_HASH_IV)

    # Dropping the unknown field must break the signature, proving it counted.
    stripped = {k: v for k, v in signed.items()
                if k != "SomeFieldEcpayAddsIn2027"}
    assert not ecpay.verify_check_mac_value(stripped, DOC_HASH_KEY, DOC_HASH_IV)


def test_missing_or_wrong_check_mac_value_is_rejected():
    assert not ecpay.verify_check_mac_value(
        dict(DOC_PARAMS), DOC_HASH_KEY, DOC_HASH_IV)
    assert not ecpay.verify_check_mac_value(
        dict(DOC_PARAMS, CheckMacValue="F" * 64), DOC_HASH_KEY, DOC_HASH_IV)


def test_md5_encrypt_type_fails_closed():
    params = dict(DOC_PARAMS, EncryptType="0")
    signed = dict(params, CheckMacValue=ecpay.build_check_mac_value(
        params, DOC_HASH_KEY, DOC_HASH_IV))
    assert not ecpay.verify_check_mac_value(signed, DOC_HASH_KEY, DOC_HASH_IV)


# ── MerchantTradeNo ──────────────────────────────────────────────────────
def test_merchant_trade_no_is_alphanumeric_and_within_20_chars():
    for _ in range(200):
        value = ecpay.generate_merchant_trade_no()
        assert len(value) <= 20, value
        assert value.isalnum(), value
        assert re.fullmatch(r"[A-Za-z0-9]+", value), value


def test_merchant_trade_no_has_no_collisions_over_10000_draws():
    values = {ecpay.generate_merchant_trade_no() for _ in range(10_000)}
    assert len(values) == 10_000


def test_merchant_trade_no_carries_a_readable_timestamp():
    fixed = datetime(2026, 7, 30, 14, 5, tzinfo=timezone.utc)
    assert ecpay.generate_merchant_trade_no(now=fixed).startswith("BLB2607301405")


def test_the_old_uuid_shape_would_have_been_rejected():
    """Regression pin: the hyphen was the violation, not the length."""
    legacy = "BLABBY-ABCDEF123456"
    assert len(legacy) <= 20 and not legacy.isalnum()


def test_prefix_too_long_to_leave_a_random_suffix_is_refused():
    with pytest.raises(ValueError):
        ecpay.generate_merchant_trade_no(prefix="A" * 12)


def test_merchant_trade_date_format():
    fixed = datetime(2026, 7, 30, 14, 5, 9, tzinfo=timezone.utc)
    assert ecpay.merchant_trade_date(fixed) == "2026/07/30 14:05:09"


# ── ECPAY_ENV ────────────────────────────────────────────────────────────
def _reload_ecpay_with(env_value):
    with patch.dict(os.environ, {"ECPAY_ENV": env_value}):
        return importlib.reload(ecpay)


def test_ecpay_env_accepts_only_stage_or_production():
    try:
        assert _reload_ecpay_with("stage").ECPAY_ENV == "stage"
        assert _reload_ecpay_with("production").ECPAY_ENV == "production"
        with pytest.raises(ValueError):
            _reload_ecpay_with("prod")
        with pytest.raises(ValueError):
            _reload_ecpay_with("staging")
    finally:
        importlib.reload(ecpay)          # restore the process-wide module


def test_unset_ecpay_env_defaults_to_the_staging_cashier():
    """A forgotten env var must not be able to take real money."""
    try:
        module = _reload_ecpay_with("")
        assert module.ECPAY_ENV == "stage"
        assert "payment-stage.ecpay.com.tw" in module.aio_checkout_url()
    finally:
        importlib.reload(ecpay)


def test_action_url_follows_the_environment_and_is_not_hardcoded():
    assert ecpay.aio_checkout_url("stage") == (
        "https://payment-stage.ecpay.com.tw/Cashier/AioCheckOut/V5")
    assert ecpay.aio_checkout_url("production") == (
        "https://payment.ecpay.com.tw/Cashier/AioCheckOut/V5")


# ── AioCheckOut parameter set ────────────────────────────────────────────
CHECKOUT_KWARGS = dict(
    merchant_id="3002607",
    merchant_trade_no="BLB2607301405ABCDEFG",
    total_amount=199,
    return_url="https://api.example.com/api/payment/callback",
    client_back_url="https://app.example.com/success.html",
    order_result_url="https://api.example.com/api/payment/return",
    hash_key=DOC_HASH_KEY,
    hash_iv=DOC_HASH_IV,
    trade_date="2026/07/30 14:05:09",
)


def test_checkout_params_are_complete_and_self_verifying():
    params = ecpay.build_aio_checkout_params(**CHECKOUT_KWARGS)
    required = {
        "MerchantID", "MerchantTradeNo", "MerchantTradeDate", "PaymentType",
        "TotalAmount", "TradeDesc", "ItemName", "ReturnURL", "ChoosePayment",
        "EncryptType", "CheckMacValue",
    }
    assert required <= set(params)
    assert params["PaymentType"] == "aio"
    assert params["ChoosePayment"] == "Credit"
    assert params["EncryptType"] == "1"
    assert params["TotalAmount"] == "199"
    # The signature covers the set we are about to POST.
    assert ecpay.verify_check_mac_value(params, DOC_HASH_KEY, DOC_HASH_IV)


def test_phase_1_carries_no_recurring_parameters():
    params = ecpay.build_aio_checkout_params(**CHECKOUT_KWARGS)
    assert not [k for k in params if k.startswith("Period")]


def test_trade_desc_and_item_name_are_plain_ascii_within_limits():
    allowed = set(string.ascii_letters + string.digits + " -")
    assert set(ecpay.TRADE_DESC) <= allowed
    assert set(ecpay.ITEM_NAME) <= allowed
    assert len(ecpay.TRADE_DESC) <= 200
    assert len(ecpay.ITEM_NAME) <= 400


@pytest.mark.parametrize("bad", [
    {"merchant_trade_no": "BLABBY-ABCDEF123456"},      # hyphen
    {"merchant_trade_no": "B" * 21},                   # too long
    {"total_amount": 0},
    {"total_amount": -199},
    {"total_amount": 199.5},
])
def test_illegal_checkout_inputs_are_refused(bad):
    with pytest.raises(ValueError):
        ecpay.build_aio_checkout_params(**{**CHECKOUT_KWARGS, **bad})


def test_return_url_and_order_result_url_must_differ():
    same = "https://api.example.com/api/payment/callback"
    with pytest.raises(ValueError):
        ecpay.build_aio_checkout_params(
            **{**CHECKOUT_KWARGS, "return_url": same, "order_result_url": same})


# ── fake supabase for the endpoint tests ─────────────────────────────────
HASH_KEY = "test-hash-key"
HASH_IV = "test-hash-iv"
USER_ID = "user-abc"


class _Resp:
    def __init__(self, data):
        self.data = data


class _DuplicateKey(Exception):
    code = "23505"
    message = 'duplicate key value violates unique constraint'


class _FakeTable:
    def __init__(self, owner, name):
        self._owner, self._name, self._op = owner, name, None

    def insert(self, row):
        self._op = "insert"
        self._owner.record(self._name, "insert", row)
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
            key = self._owner.last_event_key
            if key in self._owner.seen_keys:
                raise _DuplicateKey()
            self._owner.seen_keys.add(key)
            return _Resp([{"id": "evt-1"}])
        if self._name == "subscriptions":
            return _Resp(self._owner.subscriptions_data)
        return _Resp([{"id": "x"}])


class _FakeSupabase:
    def __init__(self, subscriptions_data=None):
        self.subscriptions_data = (
            [{"user_id": USER_ID}] if subscriptions_data is None
            else subscriptions_data
        )
        self.writes = []
        self.seen_keys = set()
        self.last_event_key = None

    def table(self, name):
        return _FakeTable(self, name)

    def record(self, name, op, row):
        self.writes.append((name, op, row))
        if name == "payment_events" and op == "insert":
            self.last_event_key = (row.get("merchant_trade_no"),
                                   row.get("total_success_times"),
                                   row.get("source"))

    def rows(self, name, op):
        return [w for w in self.writes if w[0] == name and w[1] == op]


def _signed_form(**overrides):
    params = {
        "MerchantTradeNo": "BLB2607301405ABCDEFG",
        "RtnCode":         "1",
        "RtnMsg":          "Succeeded",
        "TradeAmt":        "199",
        "EncryptType":     "1",
    }
    params.update(overrides)
    params["CheckMacValue"] = ecpay.build_check_mac_value(
        params, HASH_KEY, HASH_IV)
    return params


def _callback(form_params, fake, hash_key=HASH_KEY, hash_iv=HASH_IV):
    request = MagicMock()
    request.form = AsyncMock(return_value=form_params)
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "ECPAY_HASH_KEY", hash_key), \
         patch.object(main, "ECPAY_HASH_IV", hash_iv), \
         patch.object(main, "supabase_admin", fake):
        return asyncio.run(main.payment_callback(request=request))


# ── the six-cell response matrix ─────────────────────────────────────────
def test_matrix_1_missing_credentials_is_503_and_writes_nothing():
    fake = _FakeSupabase()
    with pytest.raises(HTTPException) as exc:
        _callback(_signed_form(), fake, hash_key="", hash_iv="")
    assert exc.value.status_code == 503
    assert fake.writes == []


def test_matrix_2_bad_signature_is_401_and_writes_nothing():
    fake = _FakeSupabase()
    form = _signed_form()
    form["CheckMacValue"] = "F" * 64
    with pytest.raises(HTTPException) as exc:
        _callback(form, fake)
    assert exc.value.status_code == 401
    assert fake.writes == []


def test_matrix_3_duplicate_event_is_200_1ok_without_regranting():
    fake = _FakeSupabase()
    form = _signed_form()
    _callback(form, fake)
    before = len(fake.rows("subscriptions", "update"))

    response = _callback(form, fake)
    assert response.status_code == 200
    assert response.body == b"1|OK"
    assert len(fake.rows("subscriptions", "update")) == before


def test_matrix_4_declined_payment_is_200_1ok_without_entitlement():
    fake = _FakeSupabase()
    response = _callback(_signed_form(RtnCode="10100058", RtnMsg="Declined"), fake)
    assert response.status_code == 200
    assert response.body == b"1|OK"
    assert fake.rows("subscriptions", "update") == []
    assert len(fake.rows("payment_events", "insert")) == 1


def test_matrix_5_successful_payment_is_200_1ok_and_grants():
    fake = _FakeSupabase()
    response = _callback(_signed_form(), fake)
    assert response.status_code == 200
    assert response.body == b"1|OK"
    updates = fake.rows("subscriptions", "update")
    assert len(updates) == 1
    assert updates[0][2]["status"] == "active"
    assert updates[0][2]["expires_at"]


def test_matrix_6_empty_row_is_500():
    fake = _FakeSupabase(subscriptions_data=[])
    with pytest.raises(HTTPException) as exc:
        _callback(_signed_form(), fake)
    assert exc.value.status_code == 500


def test_ack_body_is_byte_exact_and_text_plain():
    """No JSON, no trailing newline, no BOM — ECPay matches the string."""
    fake = _FakeSupabase()
    response = _callback(_signed_form(), fake)
    assert response.body == b"1|OK"
    assert len(response.body) == 4
    assert response.media_type == "text/plain"
    assert response.headers["content-type"].startswith("text/plain")


def test_ack_is_byte_exact_on_the_wire():
    """The one assertion that matches what ECPay's parser actually receives.

    Calling the handler directly cannot see middleware, serialisation or
    charset handling. This goes through the full ASGI stack.

    TestClient is used WITHOUT its context manager on purpose: entering it runs
    the lifespan, which starts APScheduler and queues the writing-question
    pregeneration job. Plain requests need no lifespan.
    """
    from fastapi.testclient import TestClient

    fake = _FakeSupabase()
    client = TestClient(main.app)
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "ECPAY_HASH_KEY", HASH_KEY), \
         patch.object(main, "ECPAY_HASH_IV", HASH_IV), \
         patch.object(main, "supabase_admin", fake):
        response = client.post("/api/payment/callback", data=_signed_form())

    assert response.status_code == 200
    assert response.content == b"1|OK"
    assert response.headers["content-length"] == "4"
    assert response.headers["content-type"].split(";")[0].strip() == "text/plain"


def test_ledger_source_is_ecpay_callback():
    fake = _FakeSupabase()
    _callback(_signed_form(), fake)
    assert fake.rows("payment_events", "insert")[0][2]["source"] == "ecpay_callback"


def test_five_resends_grant_once_and_leave_one_ledger_row():
    fake = _FakeSupabase()
    form = _signed_form()
    for _ in range(5):
        assert _callback(form, fake).body == b"1|OK"
    assert len(fake.rows("subscriptions", "update")) == 1
    assert len(fake.seen_keys) == 1


# ── create-order ─────────────────────────────────────────────────────────
def _create_order(fake, body=None, merchant_id="3002607"):
    request = MagicMock()
    request.json = AsyncMock(return_value=body or {})
    request.body = AsyncMock(return_value=b"")
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", return_value=USER_ID), \
         patch.object(main, "ECPAY_HASH_KEY", HASH_KEY), \
         patch.object(main, "ECPAY_HASH_IV", HASH_IV), \
         patch.object(main, "PUBLIC_BACKEND_URL", "https://api.example.com"), \
         patch.object(main, "PUBLIC_FRONTEND_URL", "https://app.example.com"), \
         patch.object(ecpay, "ECPAY_MERCHANT_ID", merchant_id), \
         patch.object(main, "supabase_admin", fake):
        result = asyncio.run(
            main.payment_create_order(request=request, authorization="Bearer t"))
    return result, request


def test_create_order_returns_a_signed_form_not_a_fake_payment_url():
    fake = _FakeSupabase()
    result, _ = _create_order(fake)
    assert "payment_url" not in result
    assert result["action_url"].endswith("/Cashier/AioCheckOut/V5")
    assert main._ecpay_check_mac_value(
        result["params"], HASH_KEY, HASH_IV) == result["params"]["CheckMacValue"]


def test_create_order_ignores_any_client_supplied_amount():
    fake = _FakeSupabase()
    result, request = _create_order(
        fake, body={"amount": 1, "TotalAmount": 1, "price": 1})
    assert result["params"]["TotalAmount"] == str(ecpay.PRO_MONTHLY_TWD)
    assert result["amount"] == ecpay.PRO_MONTHLY_TWD
    # The strongest form of the guarantee: the body is never even read.
    request.json.assert_not_called()
    request.body.assert_not_called()


def test_create_order_persists_the_merchant_trade_no_as_pending():
    fake = _FakeSupabase()
    result, _ = _create_order(fake)
    inserted = fake.rows("subscriptions", "insert")[0][2]
    assert inserted["merchant_trade_no"] == result["merchant_trade_no"]
    assert inserted["status"] == "pending"
    assert inserted["amount"] == ecpay.PRO_MONTHLY_TWD
    assert inserted["user_id"] == USER_ID


def test_create_order_urls_are_absolute_https_from_env():
    fake = _FakeSupabase()
    result, _ = _create_order(fake)
    params = result["params"]
    assert params["ReturnURL"] == "https://api.example.com/api/payment/callback"
    assert params["OrderResultURL"] == "https://api.example.com/api/payment/return"
    assert params["ClientBackURL"] == "https://app.example.com/success.html"


def test_create_order_without_merchant_id_is_503():
    fake = _FakeSupabase()
    with pytest.raises(HTTPException) as exc:
        _create_order(fake, merchant_id="")
    assert exc.value.status_code == 503
    assert fake.writes == []


def test_create_order_without_public_urls_is_503():
    fake = _FakeSupabase()
    request = MagicMock()
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", return_value=USER_ID), \
         patch.object(main, "ECPAY_HASH_KEY", HASH_KEY), \
         patch.object(main, "ECPAY_HASH_IV", HASH_IV), \
         patch.object(main, "PUBLIC_BACKEND_URL", ""), \
         patch.object(main, "PUBLIC_FRONTEND_URL", ""), \
         patch.object(ecpay, "ECPAY_MERCHANT_ID", "3002607"), \
         patch.object(main, "supabase_admin", fake):
        with pytest.raises(HTTPException) as exc:
            asyncio.run(main.payment_create_order(
                request=request, authorization="Bearer t"))
    assert exc.value.status_code == 503
    assert fake.writes == []


def test_create_order_empty_insert_row_is_500():
    fake = _FakeSupabase()
    fake.subscriptions_data = []
    with pytest.raises(HTTPException) as exc:
        _create_order(fake)
    assert exc.value.status_code == 500


def test_repeat_clicks_mint_distinct_trade_numbers():
    fake = _FakeSupabase()
    first, _ = _create_order(fake)
    second, _ = _create_order(fake)
    assert first["merchant_trade_no"] != second["merchant_trade_no"]
    assert len(fake.rows("subscriptions", "insert")) == 2


# ── /api/payment/return ──────────────────────────────────────────────────
def _payment_return(method, form=None, query=None):
    request = MagicMock()
    request.method = method
    request.form = AsyncMock(return_value=form or {})
    request.query_params = query or {}
    with patch.object(main, "PUBLIC_FRONTEND_URL", "https://app.example.com"), \
         patch.object(main, "supabase_admin", _FakeSupabase()):
        return asyncio.run(main.payment_return(request=request))


def test_return_accepts_post_and_redirects_to_an_absolute_frontend_url():
    response = _payment_return("POST", form={"MerchantTradeNo": "BLB123ABC"})
    assert response.status_code == 303
    assert response.headers["location"] == (
        "https://app.example.com/success.html?order=BLB123ABC")


def test_return_still_works_over_get():
    response = _payment_return("GET", query={"MerchantTradeNo": "BLB123ABC"})
    assert response.headers["location"].startswith(
        "https://app.example.com/success.html")


def test_return_grants_nothing():
    """User-driven and unsigned: navigation only, never entitlement."""
    fake = _FakeSupabase()
    request = MagicMock()
    request.method = "POST"
    request.form = AsyncMock(return_value={"MerchantTradeNo": "BLB123ABC",
                                           "RtnCode": "1"})
    request.query_params = {}
    with patch.object(main, "PUBLIC_FRONTEND_URL", "https://app.example.com"), \
         patch.object(main, "supabase_admin", fake):
        asyncio.run(main.payment_return(request=request))
    assert fake.writes == []


# ── credentials must never reach the logs ────────────────────────────────
def test_no_credential_material_in_the_ecpay_module_source():
    source = (ecpay.__file__ and open(ecpay.__file__, encoding="utf-8").read())
    for needle in ("ECPAY_HASH_KEY", "ECPAY_HASH_IV"):
        assert needle not in source, (
            "ecpay.py takes credentials as arguments; it must not read or "
            "name the environment variables that hold them")
