"""ECPay (綠界) AIO helpers — pure functions, no DB and no FastAPI.

Everything here is deliberately side-effect free so it can be unit tested
without a database, a request, or a network. main.py owns the endpoints, the
Supabase writes and the payment_events ledger; this module only knows how to
sign, verify, name and describe an order.

Import-time contract: this module must never construct anything that can raise
on a clean machine. The only import-time failure permitted is an explicitly
*invalid* ECPAY_ENV value — a missing one falls back to the staging cashier,
which cannot take real money.

Sources (fetched 2026-07-30):
  https://developers.ecpay.com.tw/2902/   檢查碼機制說明 (CheckMacValue)
  https://developers.ecpay.com.tw/?p=2862 AioCheckOut V5 訂單產生 parameter spec
  https://github.com/ECPay/ECPayAIO_Python  official SDK (generate_check_value)
"""

from __future__ import annotations

import hashlib
import hmac
import os
import secrets
import string
from datetime import datetime, timedelta, timezone
from urllib.parse import quote_plus

# ─── product price ───────────────────────────────────────────────────────────
# Single source of truth. The checkout amount is computed from this constant on
# the server and never read from the request body — a client-supplied price is
# a client-supplied discount.
PRO_MONTHLY_TWD = 199
PRO_PERIOD_DAYS = 30

# ─── environment ─────────────────────────────────────────────────────────────
ECPAY_MERCHANT_ID = os.getenv("ECPAY_MERCHANT_ID", "")

_ACTION_URLS = {
    "stage":      "https://payment-stage.ecpay.com.tw/Cashier/AioCheckOut/V5",
    "production": "https://payment.ecpay.com.tw/Cashier/AioCheckOut/V5",
}


def _resolve_env(raw: str) -> str:
    """Validate ECPAY_ENV. Unset → 'stage'; anything unrecognised → refuse.

    Defaulting to stage rather than raising keeps a clean checkout (and the
    test suite) importable, and the failure mode is the safe one: a
    misconfigured deploy points at the test cashier and takes no money. A value
    that is set but not recognised is a typo in a deploy config, which must not
    be silently coerced into either environment.
    """
    value = (raw or "").strip().lower()
    if not value:
        return "stage"
    if value not in _ACTION_URLS:
        raise ValueError(
            f"ECPAY_ENV must be 'stage' or 'production', got {value!r}"
        )
    return value


ECPAY_ENV = _resolve_env(os.getenv("ECPAY_ENV", ""))


def aio_checkout_url(env: str | None = None) -> str:
    """Cashier endpoint for the configured environment. Never hardcoded."""
    return _ACTION_URLS[_resolve_env(env) if env is not None else ECPAY_ENV]


# ─── CheckMacValue ───────────────────────────────────────────────────────────
# ECPay generates the digest with .NET's HttpUtility.UrlEncode, which leaves
# these seven characters unescaped where Python's quote_plus escapes them.
# Without the replacements every signature mismatches. The official Python SDK
# expresses the same rule as quote_plus(s, safe='-_.!*()'); the replacement form
# is kept because it is what 8e11b93 shipped and what the docs spell out.
_NET_URLENCODE_REPLACEMENTS = (
    ("%2d", "-"), ("%5f", "_"), ("%2e", "."), ("%21", "!"),
    ("%2a", "*"), ("%28", "("), ("%29", ")"),
)


def _ecpay_urlencode(value: str) -> str:
    """.NET-compatible URL encoding, lowercased — ECPay's exact dialect.

    Space becomes '+', percent escapes are lowercase, and the seven characters
    above stay literal. Extracted verbatim from the body of
    _ecpay_check_mac_value so it can be tested character by character; the
    composed behaviour is unchanged.
    """
    encoded = quote_plus(value).lower()
    for escape, literal in _NET_URLENCODE_REPLACEMENTS:
        encoded = encoded.replace(escape, literal)
    return encoded


def build_check_mac_value(params: dict, hash_key: str, hash_iv: str) -> str:
    """Compute ECPay's CheckMacValue over a set of form parameters.

    ECPay AIO spec: drop CheckMacValue, sort the remaining keys
    case-insensitively, wrap with HashKey/HashIV, URL-encode the whole string
    .NET-style, SHA256, uppercase.

    Verified against the worked example published at
    https://developers.ecpay.com.tw/2902/ (fetched 2026-07-30) — see
    tests/test_ecpay.py::test_official_known_answer_vector.
    """
    pairs = sorted(
        ((k, v) for k, v in params.items() if k != "CheckMacValue"),
        key=lambda kv: kv[0].lower(),
    )
    raw = "&".join(f"{k}={v}" for k, v in pairs)
    raw = f"HashKey={hash_key}&{raw}&HashIV={hash_iv}"

    return hashlib.sha256(_ecpay_urlencode(raw).encode("utf-8")).hexdigest().upper()


def verify_check_mac_value(params: dict, hash_key: str, hash_iv: str) -> bool:
    """Constant-time comparison of the supplied CheckMacValue against ours.

    Every field except CheckMacValue itself is included — no whitelist. ECPay
    adds fields over time and an unknown field is still signed, so filtering
    would break verification the day they ship one.

    Only EncryptType=1 (SHA256) is accepted. Anything else fails closed — a
    rejected callback is recoverable, a forged one is not.
    """
    supplied = (params.get("CheckMacValue") or "").strip().upper()
    if not supplied:
        return False
    encrypt_type = (params.get("EncryptType") or "1").strip()
    if encrypt_type != "1":
        return False
    expected = build_check_mac_value(params, hash_key, hash_iv)
    return hmac.compare_digest(expected, supplied)


# ─── order identity ──────────────────────────────────────────────────────────
# ECPay bills in Taipei time and shows MerchantTradeDate verbatim in the
# merchant console, so both the trade no and the trade date are stamped in
# UTC+8 rather than UTC.
_TAIPEI = timezone(timedelta(hours=8))

_TRADE_NO_ALPHABET = string.ascii_uppercase + string.digits
MERCHANT_TRADE_NO_MAX = 20


def generate_merchant_trade_no(prefix: str = "BLB", now: datetime | None = None) -> str:
    """A unique, ECPay-legal MerchantTradeNo: alphanumeric only, ≤20 chars.

    Shape: BLB + yymmddHHMM + 7 random chars = 20 characters exactly.

    The old f"BLABBY-{uuid4().hex[:12].upper()}" was 19 characters — legal on
    length, illegal on charset: ECPay allows 英數字大小寫混合 only, and the
    hyphen is the single violation.

    On the random suffix width: the spec's suggested 4 characters cannot hold.
    A batch of 10,000 ids generated inside one process shares a timestamp
    bucket, so collisions follow the birthday bound over the suffix space
    alone: 36^4 gives ~30 expected collisions per 10,000. Trading second
    precision (recoverable from created_at) for 7 characters puts the space at
    36^7 ≈ 7.8e10 and the expected count at ~6e-4.
    """
    stamp = (now or datetime.now(_TAIPEI)).strftime("%y%m%d%H%M")
    width = MERCHANT_TRADE_NO_MAX - len(prefix) - len(stamp)
    if width < 1:
        raise ValueError(f"prefix {prefix!r} leaves no room for a random suffix")
    suffix = "".join(secrets.choice(_TRADE_NO_ALPHABET) for _ in range(width))
    return f"{prefix}{stamp}{suffix}"


def merchant_trade_date(now: datetime | None = None) -> str:
    """MerchantTradeDate in ECPay's required yyyy/MM/dd HH:mm:ss, Taipei time."""
    return (now or datetime.now(_TAIPEI)).strftime("%Y/%m/%d %H:%M:%S")


# ─── AioCheckOut ─────────────────────────────────────────────────────────────
# Plain ASCII, no punctuation beyond spaces and hyphens: ECPay rejects special
# characters in TradeDesc, and & / < in any field must be escaped upstream.
TRADE_DESC = "Subscription to the Blabby Pro Membership for a Term of Thirty Days"
ITEM_NAME = "Blabby Pro Membership - Thirty Days"

# Field limits from the AioCheckOut V5 spec (developers.ecpay.com.tw/?p=2862).
_MAX_LENGTHS = {
    "MerchantID": 10, "MerchantTradeNo": 20, "MerchantTradeDate": 20,
    "PaymentType": 20, "TradeDesc": 200, "ItemName": 400,
    "ReturnURL": 200, "ChoosePayment": 20,
    "ClientBackURL": 200, "OrderResultURL": 200,
}


def build_aio_checkout_params(
    *,
    merchant_id: str,
    merchant_trade_no: str,
    total_amount: int,
    return_url: str,
    client_back_url: str,
    order_result_url: str,
    hash_key: str,
    hash_iv: str,
    trade_date: str | None = None,
) -> dict:
    """The complete, signed parameter set for a Phase 1 one-off Credit charge.

    Phase 1 is a single NT$199 / 30-day purchase: ChoosePayment=Credit and no
    Period* parameters. Recurring billing is Phase 2 and is deliberately absent.

    The caller POSTs these to aio_checkout_url() as a form. CheckMacValue is
    computed last, over every other field.
    """
    if not merchant_trade_no.isalnum():
        raise ValueError("MerchantTradeNo must be alphanumeric")
    if len(merchant_trade_no) > MERCHANT_TRADE_NO_MAX:
        raise ValueError("MerchantTradeNo must be at most 20 characters")
    if not isinstance(total_amount, int) or total_amount <= 0:
        raise ValueError("TotalAmount must be a positive whole number of TWD")
    # ECPay rejects a checkout whose server callback and browser return point at
    # the same place, and silently mixing them would let a user-driven GET stand
    # in for the signed server-to-server notification.
    if return_url == order_result_url:
        raise ValueError("ReturnURL and OrderResultURL must differ")

    params = {
        "MerchantID":        merchant_id,
        "MerchantTradeNo":   merchant_trade_no,
        "MerchantTradeDate": trade_date or merchant_trade_date(),
        "PaymentType":       "aio",
        "TotalAmount":       str(total_amount),
        "TradeDesc":         TRADE_DESC,
        "ItemName":          ITEM_NAME,
        "ReturnURL":         return_url,
        "ChoosePayment":     "Credit",
        "ClientBackURL":     client_back_url,
        "OrderResultURL":    order_result_url,
        "EncryptType":       "1",
    }

    for field, limit in _MAX_LENGTHS.items():
        value = params.get(field, "")
        if len(value) > limit:
            raise ValueError(f"{field} exceeds ECPay's {limit}-character limit")

    params["CheckMacValue"] = build_check_mac_value(params, hash_key, hash_iv)
    return params
