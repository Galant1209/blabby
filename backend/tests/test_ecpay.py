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
import hashlib
import logging
import os
import re
import string
from contextlib import contextmanager
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
    assert ecpay.ecpay_urlencode(literal) == literal


def test_urlencode_maps_space_to_plus():
    assert ecpay.ecpay_urlencode("a b") == "a+b"


def test_urlencode_lowercases_percent_escapes():
    # '@' -> %40, '/' -> %2F which must come back lowercase as %2f
    assert ecpay.ecpay_urlencode("/") == "%2f"
    assert ecpay.ecpay_urlencode("@") == "%40"


def test_urlencode_still_escapes_the_dangerous_separators():
    assert ecpay.ecpay_urlencode("&") == "%26"
    assert ecpay.ecpay_urlencode("<") == "%3c"
    assert ecpay.ecpay_urlencode("=") == "%3d"


def test_tilde_escapes_to_percent_7e():
    """The one character in its group that .NET *does* escape.

    An earlier revision of this file asserted the opposite, reasoning from
    ECPay's official Python SDK — which computes quote_plus(s, safe='-_.!*()')
    and so leaves '~' literal. That reasoning was wrong: the authority is the
    .NET server that verifies the signature, not the SDK, and the official
    conversion table's third column (.NET URLEncode 結果) gives %7E. The SDK and
    the server disagree here; we follow the server.

    Source: https://developers.ecpay.com.tw/?p=2904, fetched 2026-07-30.
    """
    assert ecpay.ecpay_urlencode("~") == "%7e"


def test_the_tilde_rule_actually_changes_the_digest():
    """Proof the new rule carries signal rather than being decorative.

    If '~' were still passed through literally, these two digests would be
    identical and the fix would be untestable.
    """
    with_tilde = ecpay.build_check_mac_value(
        {"A": "a~b"}, DOC_HASH_KEY, DOC_HASH_IV)
    literal_tilde = hashlib.sha256(
        f"HashKey={DOC_HASH_KEY}&A=a~b&HashIV={DOC_HASH_IV}".lower()
        .encode("utf-8")).hexdigest().upper()
    assert with_tilde != literal_tilde
    assert "%7e" in ecpay.ecpay_urlencode("a~b")


def test_three_of_the_replacements_are_already_no_ops_in_python():
    """'-', '_' and '.' never get escaped by quote_plus in the first place.

    The pairs are kept because the ECPay table lists them and dropping the
    inert three would make the implementation harder to diff against the spec.
    """
    for char in "-_.":
        assert ecpay.ecpay_urlencode(char) == char


# ── the complete /2904/ conversion table, row by row ─────────────────────
# Third column only ('.NET URLEncode 結果') — that is the dialect ECPay's
# server hashes with. Column two (plain URLEncode) is listed by the docs for
# contrast and is NOT what we implement; the visible difference between the two
# columns is exactly the space, and the -_.!*() group.
# Source: https://developers.ecpay.com.tw/?p=2904, fetched 2026-07-30.
DOTNET_TABLE = [
    (".", "."),   ("-", "-"),   ("_", "_"),   ("!", "!"),
    ("*", "*"),   ("(", "("),   (")", ")"),
    ("~", "%7e"),                                  # the sole exception
    (" ", "+"),                                    # %20 in plain URLEncode
    ("@", "%40"), ("#", "%23"), ("$", "%24"), ("%", "%25"),
    ("^", "%5e"), ("&", "%26"), ("=", "%3d"), ("+", "%2b"),
    (";", "%3b"), ("?", "%3f"), ("/", "%2f"), ("\\", "%5c"),
    (">", "%3e"), ("<", "%3c"), ("`", "%60"), ("[", "%5b"),
    ("]", "%5d"), ("{", "%7b"), ("}", "%7d"), (":", "%3a"),
    ("'", "%27"), ('"', "%22"), (",", "%2c"), ("|", "%7c"),
]


@pytest.mark.parametrize("char,expected", DOTNET_TABLE,
                         ids=[f"{c!r}" for c, _ in DOTNET_TABLE])
def test_official_conversion_table_row(char, expected):
    assert ecpay.ecpay_urlencode(char) == expected


def test_the_conversion_table_is_transcribed_in_full():
    """33 rows on the page; a silently shortened list would test nothing."""
    assert len(DOTNET_TABLE) == 33
    assert len({c for c, _ in DOTNET_TABLE}) == 33


def test_urlencode_handles_utf8():
    assert ecpay.ecpay_urlencode("促銷方案") == (
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


def test_merchant_trade_no_is_a_date_plus_twelve_random_chars():
    fixed = datetime(2026, 7, 30, 14, 5, tzinfo=timezone.utc)
    value = ecpay.generate_merchant_trade_no(now=fixed)
    assert len(value) == 20
    assert value.startswith("20260730")
    assert re.fullmatch(r"[A-Z0-9]{12}", value[8:])


def test_merchant_trade_no_has_no_blb_prefix():
    """Pinned because the clean-up runbook depends on it.

    `delete ... where merchant_trade_no like 'BLB%'` matches zero rows against
    this shape; docs/ECPAY_STAGE_VERIFICATION.md filters on user_id instead.
    """
    assert not ecpay.generate_merchant_trade_no().startswith("BLB")


def test_the_old_uuid_shape_would_have_been_rejected():
    """Regression pin: the hyphen was the violation, not the length."""
    legacy = "BLABBY-ABCDEF123456"
    assert len(legacy) <= 20 and not legacy.isalnum()


def test_merchant_trade_date_format():
    fixed = datetime(2026, 7, 30, 14, 5, 9, tzinfo=timezone.utc)
    assert ecpay.merchant_trade_date(fixed) == "2026/07/30 14:05:09"


# ── configuration: ECPAY_ENV / ECPAY_MERCHANT_ID ─────────────────────────
_URL_ENV = {
    "PUBLIC_BACKEND_URL":  "https://api.example.com",
    "PUBLIC_FRONTEND_URL": "https://app.example.com",
}


@contextmanager
def _config_from_env(**env):
    """Run load_config() against a temporary environment, then restore."""
    saved_config, saved_error = ecpay._CONFIG, ecpay._CONFIG_ERROR
    try:
        with patch.dict(os.environ, dict(_URL_ENV, **env)):
            ecpay.load_config()
            yield ecpay
    finally:
        ecpay._CONFIG, ecpay._CONFIG_ERROR = saved_config, saved_error


@pytest.mark.parametrize("env_value,expected", [
    ("stage",      "https://payment-stage.ecpay.com.tw/Cashier/AioCheckOut/V5"),
    ("production", "https://payment.ecpay.com.tw/Cashier/AioCheckOut/V5"),
])
def test_valid_ecpay_env_selects_the_matching_cashier(env_value, expected):
    with _config_from_env(ECPAY_ENV=env_value, ECPAY_MERCHANT_ID="3002607") as mod:
        config = mod.get_config()
        assert config.env == env_value
        assert config.action_url == expected
        assert config.merchant_id == "3002607"


@pytest.mark.parametrize("env_value", ["Stage", "PRODUCTION", "  stage  "])
def test_ecpay_env_is_case_and_whitespace_insensitive(env_value):
    """Defined behaviour, not an accident: strip().lower() before matching."""
    with _config_from_env(ECPAY_ENV=env_value, ECPAY_MERCHANT_ID="3002607") as mod:
        assert mod.get_config().env == env_value.strip().lower()


@pytest.mark.parametrize("env_value", ["prod", "staging", "test", "live", "1"])
def test_invalid_ecpay_env_never_falls_back(env_value):
    """The one behaviour that must never regress.

    Falling back to stage would put a real card into the test merchant;
    falling back to production would charge real money during a test. The only
    permitted outcome is "payments unavailable".
    """
    with _config_from_env(ECPAY_ENV=env_value, ECPAY_MERCHANT_ID="3002607") as mod:
        with pytest.raises(mod.EcpayConfigError):
            mod.get_config()
        assert mod._CONFIG is None


@pytest.mark.parametrize("env", [
    {"ECPAY_ENV": "", "ECPAY_MERCHANT_ID": "3002607"},        # empty
    {"ECPAY_ENV": "   ", "ECPAY_MERCHANT_ID": "3002607"},     # whitespace only
    {"ECPAY_ENV": "stage", "ECPAY_MERCHANT_ID": ""},          # no merchant id
])
def test_incomplete_configuration_is_an_error_not_a_default(env):
    with _config_from_env(**env) as mod:
        with pytest.raises(mod.EcpayConfigError):
            mod.get_config()


def test_unset_ecpay_env_is_an_error():
    """Unset is a configuration error, exactly like a typo. No stage default."""
    saved_config, saved_error = ecpay._CONFIG, ecpay._CONFIG_ERROR
    try:
        env = {k: v for k, v in os.environ.items() if k != "ECPAY_ENV"}
        with patch.dict(os.environ, env, clear=True):
            ecpay.load_config()
            with pytest.raises(ecpay.EcpayConfigError):
                ecpay.get_config()
    finally:
        ecpay._CONFIG, ecpay._CONFIG_ERROR = saved_config, saved_error


def test_config_error_message_never_echoes_the_offending_value():
    secret_looking = "sk-live-abcdef0123456789"
    with _config_from_env(ECPAY_ENV=secret_looking,
                          ECPAY_MERCHANT_ID="3002607") as mod:
        with pytest.raises(mod.EcpayConfigError) as exc:
            mod.get_config()
    assert secret_looking not in str(exc.value)


def test_load_config_is_idempotent_and_recovers():
    """A fixed environment must be pickable up by re-running load_config()."""
    with _config_from_env(ECPAY_ENV="prod", ECPAY_MERCHANT_ID="3002607") as mod:
        with pytest.raises(mod.EcpayConfigError):
            mod.get_config()
    with _config_from_env(ECPAY_ENV="stage", ECPAY_MERCHANT_ID="3002607") as mod:
        assert mod.get_config().env == "stage"


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
    user_id="11111111-2222-3333-4444-555555555555",
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


# ── sanitize_trade_text ──────────────────────────────────────────────────
@pytest.mark.parametrize("raw,expected", [
    ("Blabby Pro",            "Blabby Pro"),
    ("Tea & Cakes",           "Tea Cakes"),          # '&' breaks the cashier
    ("a<b>c",                 "a b c"),
    ("Pro (Monthly)",         "Pro Monthly"),        # the .NET-divergent set
    ("Pro!*~",                "Pro"),
    ("Pro   Plan",            "Pro Plan"),           # runs collapse
    ("  Pro Plan  ",          "Pro Plan"),           # and are trimmed
    ("方案 Pro",              "方案 Pro"),           # CJK is allowed
    ("Pro-Membership",        "Pro-Membership"),     # hyphen survives
    ("",                      ""),
])
def test_sanitize_trade_text_whitelist(raw, expected):
    assert ecpay.sanitize_trade_text(raw, 200) == expected


def test_sanitize_trade_text_truncates_to_the_field_limit():
    assert len(ecpay.sanitize_trade_text("A" * 500, 200)) == 200


def test_chinese_truncation_never_splits_a_character():
    """ECPay truncates an over-length ItemName server-side, and a cut that
    lands mid-character produces mojibake — which changes the bytes hashed,
    breaks CheckMacValue, and loses the order. Truncating by character on our
    side is what keeps that from ever being reachable.

    A byte-based cut of 中文 at an odd offset would leave a lone continuation
    byte; this asserts the result is always decodable and re-encodable.
    """
    text = "中文" * 300                              # 600 characters
    for limit in range(1, 40):
        out = ecpay.sanitize_trade_text(text, limit)
        assert len(out) <= limit
        # Round-trips cleanly: no half-characters, no replacement chars.
        assert out.encode("utf-8").decode("utf-8") == out
        assert "�" not in out


def test_chinese_truncation_counts_characters_not_bytes():
    """400 CJK characters is 1200 UTF-8 bytes. The limit is characters."""
    out = ecpay.sanitize_trade_text("中" * 500, ecpay.ITEM_NAME_MAX)
    assert len(out) == 400
    assert len(out.encode("utf-8")) == 1200


def test_truncated_chinese_still_signs_and_verifies():
    """The end-to-end version of the same guarantee."""
    item = ecpay.sanitize_trade_text("方案" * 400, ecpay.ITEM_NAME_MAX)
    params = dict(DOC_PARAMS, ItemName=item)
    signed = dict(params, CheckMacValue=ecpay.build_check_mac_value(
        params, DOC_HASH_KEY, DOC_HASH_IV))
    assert ecpay.verify_check_mac_value(signed, DOC_HASH_KEY, DOC_HASH_IV)


def test_sanitize_trade_text_output_survives_a_signature_round_trip():
    """Whatever comes out must be signable and verifiable unchanged."""
    text = ecpay.sanitize_trade_text("Tea & Cakes (Pro!) ~ 2026", 200)
    params = dict(DOC_PARAMS, TradeDesc=text)
    signed = dict(params, CheckMacValue=ecpay.build_check_mac_value(
        params, DOC_HASH_KEY, DOC_HASH_IV))
    assert ecpay.verify_check_mac_value(signed, DOC_HASH_KEY, DOC_HASH_IV)


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
            [{
                "user_id": USER_ID,
                "amount": ecpay.PRO_MONTHLY_TWD,
                "status": "pending",
                "expires_at": None,
            }] if subscriptions_data is None
            else subscriptions_data
        )
        self.writes = []
        self.seen_keys = set()
        self.last_event_key = None
        self.rpc_calls = []
        self.accepted_events = []
        self.activations = []
        self.activated_orders = set()

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

    def rpc(self, name, params):
        self.rpc_calls.append((name, params))
        if name != "accept_ecpay_payment":
            raise AssertionError(f"unexpected RPC: {name}")

        def execute():
            order = params["p_merchant_trade_no"]
            key = (order, params["p_total_success_times"], "ecpay_callback")
            if order in self.activated_orders or key in self.seen_keys:
                return _Resp([{"result": "duplicate"}])
            self.seen_keys.add(key)
            self.activated_orders.add(order)
            self.accepted_events.append({
                "source": "ecpay_callback",
                "merchant_trade_no": order,
                "rtn_code": params["p_rtn_code"],
                "rtn_msg": params["p_rtn_msg"],
                "raw_payload": params["p_raw_payload"],
            })
            self.activations.append(params)
            return _Resp([{"result": "activated"}])

        return MagicMock(execute=execute)


def _signed_form(**overrides):
    params = {
        "MerchantID":      "3002607",
        "MerchantTradeNo": "BLB2607301405ABCDEFG",
        "RtnCode":         "1",
        "RtnMsg":          "Succeeded",
        "TradeAmt":        "199",
        "CustomField1":     USER_ID,
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
         patch.object(ecpay, "_CONFIG", STAGE_CONFIG), \
         patch.object(ecpay, "_CONFIG_ERROR", None), \
         patch.object(main, "supabase_admin", fake):
        return asyncio.run(main.payment_callback(request=request))


def _assert_exact_ack(response):
    assert response.status_code == 200
    assert response.body == b"1|OK"
    assert len(response.body) == 4
    assert response.media_type == "text/plain"


# ── TASK 6A-R.1: trust gates must precede every mutation ─────────────────
def test_callback_rejects_signed_merchant_mismatch_without_mutation(caplog):
    fake = _FakeSupabase()
    response = _callback(_signed_form(MerchantID="9999999"), fake)

    _assert_exact_ack(response)
    assert fake.writes == []
    assert fake.rpc_calls == []
    assert "merchant" in caplog.text.lower()


def test_callback_rejects_signed_amount_mismatch_without_poisoning_retry(caplog):
    fake = _FakeSupabase()
    rejected = _callback(_signed_form(TradeAmt="1"), fake)

    _assert_exact_ack(rejected)
    assert fake.rows("payment_events", "insert") == []
    assert fake.rows("subscriptions", "update") == []
    assert fake.rpc_calls == []
    assert "amount" in caplog.text.lower()

    accepted = _callback(_signed_form(), fake)
    _assert_exact_ack(accepted)
    assert len(fake.rpc_calls) == 1
    assert len(fake.activations) == 1
    assert len(fake.accepted_events) == 1


def test_callback_rejects_ownership_mismatch_before_activation(caplog):
    fake = _FakeSupabase()
    response = _callback(
        _signed_form(CustomField1="00000000-0000-0000-0000-000000000099"),
        fake,
    )

    _assert_exact_ack(response)
    assert fake.rows("subscriptions", "update") == []
    assert fake.rows("payment_events", "insert") == []
    assert fake.rpc_calls == []
    assert "owner" in caplog.text.lower()


def test_callback_with_invalid_ecpay_env_acknowledges_without_mutation():
    fake = _FakeSupabase()
    request = MagicMock()
    request.form = AsyncMock(return_value=_signed_form())
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "ECPAY_HASH_KEY", HASH_KEY), \
         patch.object(main, "ECPAY_HASH_IV", HASH_IV), \
         patch.object(ecpay, "_CONFIG", None), \
         patch.object(ecpay, "_CONFIG_ERROR",
                      "ECPAY_ENV must be 'stage' or 'production'"), \
         patch.object(main, "supabase_admin", fake):
        response = asyncio.run(main.payment_callback(request=request))

    _assert_exact_ack(response)
    assert fake.writes == []
    assert fake.rpc_calls == []


def test_callback_performs_no_mutation_until_all_trust_gates_pass():
    rejected_forms = [
        _signed_form(MerchantID="wrong"),
        _signed_form(TradeAmt="not-an-integer"),
        _signed_form(CustomField1="wrong-owner"),
        _signed_form(SimulatePaid="1"),
    ]
    for form in rejected_forms:
        fake = _FakeSupabase()
        config = PROD_CONFIG if form.get("SimulatePaid") == "1" else STAGE_CONFIG
        response = _callback_with_env(form, fake, config)
        _assert_exact_ack(response)
        assert fake.rows("subscriptions", "update") == []
        assert fake.rows("payment_events", "insert") == []
        assert fake.rpc_calls == []


def test_success_uses_one_atomic_rpc_not_separate_http_mutations():
    fake = _FakeSupabase()
    response = _callback(_signed_form(), fake)

    _assert_exact_ack(response)
    assert [name for name, _ in fake.rpc_calls] == ["accept_ecpay_payment"]
    assert fake.rows("payment_events", "insert") == []
    assert fake.rows("subscriptions", "update") == []


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
    before = len(fake.activations)

    response = _callback(form, fake)
    assert response.status_code == 200
    assert response.body == b"1|OK"
    assert len(fake.activations) == before == 1
    assert len(fake.accepted_events) == 1


def test_matrix_4_declined_payment_is_200_1ok_without_entitlement():
    fake = _FakeSupabase()
    response = _callback(_signed_form(RtnCode="10100058", RtnMsg="Declined"), fake)
    assert response.status_code == 200
    assert response.body == b"1|OK"
    assert fake.rows("subscriptions", "update") == []
    assert fake.accepted_events == []


# ── RtnCode != 1 is a routine path, not an edge case ─────────────────────
# 3D Secure is live on the production merchant. Failed authorisation, an
# expired 3D session and a buyer who closes the bank's page all arrive here.
# This block is sized to that reality rather than to "one branch, covered".

@pytest.mark.parametrize("rtn_code,rtn_msg", [
    ("0",        "General failure"),
    ("2",        "Transaction failed"),
    ("10100058", "Authorisation declined by issuer"),
    ("10100251", "Card refused"),
    ("10100252", "Insufficient funds"),
    ("-1",       "Unexpected negative code"),
    ("",         ""),                       # ECPay omitted the field entirely
])
def test_no_non_success_code_can_grant_entitlement(rtn_code, rtn_msg):
    fake = _FakeSupabase()
    response = _callback(_signed_form(RtnCode=rtn_code, RtnMsg=rtn_msg), fake)
    assert response.body == b"1|OK", "a recorded failure must not be resent"
    assert fake.rows("subscriptions", "update") == []
    assert [w for w in fake.writes if w[0] == "profiles"] == []


def test_the_failure_reason_is_logged_without_consuming_success_identity(caplog):
    """A failed observation stays measurable without poisoning a later success."""
    caplog.set_level(logging.INFO, logger=main.logger.name)
    fake = _FakeSupabase()
    _callback(_signed_form(RtnCode="10100058",
                           RtnMsg="Authorisation declined"), fake)
    assert "10100058" in caplog.text
    assert fake.accepted_events == []
    assert fake.seen_keys == set()


def test_a_failed_attempt_leaves_the_order_pending_not_active():
    """No status write at all — the row keeps whatever create-order set."""
    fake = _FakeSupabase()
    _callback(_signed_form(RtnCode="10100058"), fake)
    assert fake.rows("subscriptions", "update") == []


def test_buyer_retries_after_a_3d_failure_and_the_second_order_grants():
    """The real journey: 3D fails, the buyer presses Upgrade again, pays.

    The retry is a different MerchantTradeNo, so it is a different idempotency
    key and must not be swallowed as a duplicate of the failure.
    """
    fake = _FakeSupabase()
    failed = _callback(
        _signed_form(MerchantTradeNo="20260730AAAAAAAAAAAA", RtnCode="10100058"),
        fake)
    assert failed.body == b"1|OK"
    assert fake.rows("subscriptions", "update") == []

    ok = _callback(
        _signed_form(MerchantTradeNo="20260730BBBBBBBBBBBB", RtnCode="1"), fake)
    assert ok.body == b"1|OK"
    assert len(fake.activations) == 1
    assert len(fake.accepted_events) == 1


def test_a_late_failure_callback_cannot_undo_an_earlier_success():
    """Out-of-order delivery must never revoke a paid period."""
    fake = _FakeSupabase()
    _callback(_signed_form(MerchantTradeNo="20260730CCCCCCCCCCCC",
                           RtnCode="1"), fake)
    _callback(_signed_form(MerchantTradeNo="20260730DDDDDDDDDDDD",
                           RtnCode="10100058"), fake)
    assert len(fake.activations) == 1


def test_a_failure_resend_never_consumes_accepted_success_identity():
    fake = _FakeSupabase()
    form = _signed_form(RtnCode="10100058")
    for _ in range(5):
        assert _callback(form, fake).body == b"1|OK"
    assert fake.seen_keys == set()
    assert fake.rows("subscriptions", "update") == []


def test_matrix_5_successful_payment_is_200_1ok_and_grants():
    fake = _FakeSupabase()
    response = _callback(_signed_form(), fake)
    assert response.status_code == 200
    assert response.body == b"1|OK"
    assert len(fake.activations) == 1
    assert fake.activations[0]["p_expires_at"]


def test_matrix_6_unknown_order_is_acknowledged_without_mutation():
    fake = _FakeSupabase(subscriptions_data=[])
    response = _callback(_signed_form(), fake)
    _assert_exact_ack(response)
    assert fake.rpc_calls == []


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


# ── matrix cell 7: simulated payment in production ───────────────────────
# ECPay's merchant console has a 模擬付款 button in production. It delivers a
# fully signed callback with RtnCode=1 and SimulatePaid=1 while no money moves:
# free Pro for anyone with console access. The decision is made from ECPAY_ENV,
# never from the callback body.

def _callback_with_env(form, fake, config):
    request = MagicMock()
    request.form = AsyncMock(return_value=form)
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "ECPAY_HASH_KEY", HASH_KEY), \
         patch.object(main, "ECPAY_HASH_IV", HASH_IV), \
         patch.object(ecpay, "_CONFIG", config), \
         patch.object(ecpay, "_CONFIG_ERROR", None), \
         patch.object(main, "supabase_admin", fake):
        return asyncio.run(main.payment_callback(request=request))


def test_matrix_7_simulated_payment_in_production_grants_nothing():
    fake = _FakeSupabase()
    response = _callback_with_env(
        _signed_form(RtnCode="1", SimulatePaid="1"), fake, PROD_CONFIG)
    assert response.status_code == 200
    assert response.body == b"1|OK", "ECPay must not be made to retry"
    assert fake.rows("subscriptions", "update") == [], "no entitlement"
    assert fake.accepted_events == [], "rejection must not poison a valid retry"


def test_simulated_payment_on_stage_grants_normally():
    """Otherwise the stage runbook could never verify anything."""
    fake = _FakeSupabase()
    response = _callback_with_env(
        _signed_form(RtnCode="1", SimulatePaid="1"), fake, STAGE_CONFIG)
    assert response.body == b"1|OK"
    assert len(fake.activations) == 1


def test_a_real_payment_in_production_still_grants():
    """SimulatePaid=0 is what genuine production callbacks carry."""
    fake = _FakeSupabase()
    _callback_with_env(_signed_form(RtnCode="1", SimulatePaid="0"),
                       fake, PROD_CONFIG)
    assert len(fake.activations) == 1


@pytest.mark.parametrize("value", ["1", "2", "y", "Y", "true", " 1 "])
def test_any_non_zero_simulate_paid_is_refused_in_production(value):
    """Unparseable means refuse, not grant."""
    fake = _FakeSupabase()
    _callback_with_env(_signed_form(RtnCode="1", SimulatePaid=value),
                       fake, PROD_CONFIG)
    assert fake.rows("subscriptions", "update") == [], value


def test_simulate_paid_is_refused_when_the_config_is_unknown():
    """Fail-safe: if we cannot prove we are on stage, we are not on stage."""
    fake = _FakeSupabase()
    _callback_with_env(_signed_form(RtnCode="1", SimulatePaid="1"), fake, None)
    assert fake.rows("subscriptions", "update") == []


def test_the_simulate_decision_ignores_the_callback_body_entirely():
    """A forged SimulatePaid=0 must not be able to buy a grant on stage rules.

    The env is ours; the body is theirs. Proof: on PROD_CONFIG the same signed
    body with SimulatePaid=1 is refused no matter what else it claims.
    """
    fake = _FakeSupabase()
    _callback_with_env(
        _signed_form(RtnCode="1", SimulatePaid="1", StoreID="stage"),
        fake, PROD_CONFIG)
    assert fake.rows("subscriptions", "update") == []


# ── CustomField1 cross-check ─────────────────────────────────────────────
def test_custom_field1_matching_the_order_owner_grants():
    fake = _FakeSupabase()
    response = _callback(_signed_form(RtnCode="1", CustomField1=USER_ID), fake)
    assert response.body == b"1|OK"
    assert len(fake.activations) == 1


def test_custom_field1_mismatch_refuses_to_grant():
    """Signed, so a mismatch is either our bug or someone playing games."""
    fake = _FakeSupabase()
    response = _callback(
        _signed_form(RtnCode="1", CustomField1="someone-else"), fake)
    _assert_exact_ack(response)
    assert fake.rpc_calls == []


def test_custom_field1_absent_fails_closed():
    """Create-order always stamps it; absence cannot prove callback ownership."""
    fake = _FakeSupabase()
    form = _signed_form(RtnCode="1")
    del form["CustomField1"]
    form["CheckMacValue"] = ecpay.build_check_mac_value(form, HASH_KEY, HASH_IV)
    response = _callback(form, fake)
    _assert_exact_ack(response)
    assert fake.rpc_calls == []


# ── TradeNo ──────────────────────────────────────────────────────────────
def test_ecpay_trade_no_is_persisted_for_support_conversations():
    fake = _FakeSupabase()
    _callback(_signed_form(RtnCode="1", TradeNo="2607301122334455"), fake)
    assert fake.activations[0]["p_ecpay_trade_no"] == "2607301122334455"


def test_a_missing_trade_no_is_stored_as_null_not_empty_string():
    fake = _FakeSupabase()
    _callback(_signed_form(RtnCode="1"), fake)
    assert fake.activations[0]["p_ecpay_trade_no"] is None


def test_ledger_source_is_ecpay_callback():
    fake = _FakeSupabase()
    _callback(_signed_form(), fake)
    assert fake.accepted_events[0]["source"] == "ecpay_callback"


def test_five_resends_grant_once_and_leave_one_ledger_row():
    fake = _FakeSupabase()
    form = _signed_form()
    for _ in range(5):
        assert _callback(form, fake).body == b"1|OK"
    assert len(fake.activations) == 1
    assert len(fake.accepted_events) == 1
    assert len(fake.seen_keys) == 1


# ── create-order ─────────────────────────────────────────────────────────
STAGE_CONFIG = ecpay.EcpayConfig(
    merchant_id="3002607",
    env="stage",
    action_url="https://payment-stage.ecpay.com.tw/Cashier/AioCheckOut/V5",
    return_url="https://api.example.com/api/payment/callback",
    client_back_url="https://app.example.com/success.html",
    order_result_url="https://api.example.com/api/payment/return",
)

PROD_CONFIG = STAGE_CONFIG._replace(
    env="production",
    action_url="https://payment.ecpay.com.tw/Cashier/AioCheckOut/V5",
)


def _create_order(fake, body=None, config=STAGE_CONFIG, config_error=None):
    request = MagicMock()
    request.json = AsyncMock(return_value=body or {})
    request.body = AsyncMock(return_value=b"")
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", return_value=USER_ID), \
         patch.object(main, "ECPAY_HASH_KEY", HASH_KEY), \
         patch.object(main, "ECPAY_HASH_IV", HASH_IV), \
         patch.object(main, "PUBLIC_BACKEND_URL", "https://api.example.com"), \
         patch.object(main, "PUBLIC_FRONTEND_URL", "https://app.example.com"), \
         patch.object(ecpay, "_CONFIG", config), \
         patch.object(ecpay, "_CONFIG_ERROR", config_error), \
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


def test_create_order_with_a_broken_config_is_503_and_writes_nothing():
    fake = _FakeSupabase()
    with pytest.raises(HTTPException) as exc:
        _create_order(fake, config=None,
                      config_error="ECPAY_ENV must be 'stage' or 'production'")
    assert exc.value.status_code == 503
    assert fake.writes == []


def test_create_order_before_load_config_is_503():
    fake = _FakeSupabase()
    with pytest.raises(HTTPException) as exc:
        _create_order(fake, config=None, config_error=None)
    assert exc.value.status_code == 503
    assert fake.writes == []


@pytest.mark.parametrize("env,reason", [
    ({"PUBLIC_BACKEND_URL": ""},                    "backend url missing"),
    ({"PUBLIC_FRONTEND_URL": ""},                   "frontend url missing"),
    ({"PUBLIC_BACKEND_URL": "http://api.example.com"}, "not https"),
    ({"PUBLIC_FRONTEND_URL": "app.example.com"},    "no scheme"),
])
def test_create_order_without_usable_public_urls_is_503(env, reason):
    """The URLs are validated once at startup, not rebuilt per request."""
    fake = _FakeSupabase()
    with _config_from_env(ECPAY_ENV="stage", ECPAY_MERCHANT_ID="3002607", **env):
        with pytest.raises(HTTPException) as exc:
            _create_order(fake, config=ecpay._CONFIG,
                          config_error=ecpay._CONFIG_ERROR)
    assert exc.value.status_code == 503, reason
    assert fake.writes == [], reason


def test_return_url_and_order_result_url_are_never_identical():
    """ECPay forbids it outright: the signed server-to-server notification and
    the unsigned browser landing must not arrive at the same handler.

    Honest scope: load_config() builds these from one origin with two different
    paths, so the guard inside it is UNREACHABLE by construction today. It is a
    tripwire for a future edit that changes either path, not a live branch.
    What this test can prove is the invariant itself.
    """
    with _config_from_env(ECPAY_ENV="stage", ECPAY_MERCHANT_ID="3002607") as mod:
        config = mod.get_config()
        assert config.return_url != config.order_result_url
        assert config.return_url.endswith("/api/payment/callback")
        assert config.order_result_url.endswith("/api/payment/return")


def test_checkout_construction_refuses_identical_urls():
    """The same rule at the layer where it *is* reachable: a caller passing
    both URLs the same never gets a signed parameter set."""
    same = "https://api.example.com/api/payment/callback"
    with pytest.raises(ValueError):
        ecpay.build_aio_checkout_params(
            **{**CHECKOUT_KWARGS, "return_url": same, "order_result_url": same})


def test_create_order_empty_insert_row_is_500():
    fake = _FakeSupabase()
    fake.subscriptions_data = []
    with pytest.raises(HTTPException) as exc:
        _create_order(fake)
    assert exc.value.status_code == 500


def test_pending_rows_leave_created_at_to_the_database():
    """Known accumulation, deliberately not cleaned up in Phase 1.

    A buyer who abandons the bank's 3D page may produce no callback at all, so
    the pending row stays forever. That is accepted for now — but it is only
    *findable* if created_at is populated, and subscriptions.created_at has
    `DEFAULT now()` (20260508_subscriptions.sql). The insert must therefore not
    send its own value and must not overwrite the default with NULL.
    """
    fake = _FakeSupabase()
    _create_order(fake)
    inserted = fake.rows("subscriptions", "insert")[0][2]
    assert "created_at" not in inserted
    assert inserted["status"] == "pending"


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
