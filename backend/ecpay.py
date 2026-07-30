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
from typing import NamedTuple
from urllib.parse import quote_plus

# ─── product price ───────────────────────────────────────────────────────────
# Single source of truth. The checkout amount is computed from this constant on
# the server and never read from the request body — a client-supplied price is
# a client-supplied discount.
PRO_MONTHLY_TWD = 199
PRO_PERIOD_DAYS = 30

# ─── configuration ───────────────────────────────────────────────────────────
# Loaded from the environment by load_config() at FastAPI startup, never at
# import. An earlier revision validated ECPAY_ENV at module scope; a typo there
# ('prod') made `import main` raise, which takes the whole backend down —
# Speaking, Reading, Writing and all — over a payment setting. Blast radius now
# stops at the payment endpoints: they answer 503, everything else serves.
#
# There is deliberately NO fallback. Defaulting an unset ECPAY_ENV to 'stage'
# would mean a production deploy that forgot the variable silently sends buyers
# to the test cashier and takes no money, with nothing in the logs saying so.
# Unset is a configuration error, same as a typo.

_ACTION_URLS = {
    "stage":      "https://payment-stage.ecpay.com.tw/Cashier/AioCheckOut/V5",
    "production": "https://payment.ecpay.com.tw/Cashier/AioCheckOut/V5",
}


class EcpayConfigError(RuntimeError):
    """設定錯誤。此例外永不在 import 時被 raise。"""


class EcpayConfig(NamedTuple):
    merchant_id: str
    env: str
    action_url: str


_CONFIG: EcpayConfig | None = None
_CONFIG_ERROR: str | None = None   # 有值即代表金流不可用


def load_config() -> None:
    """Read and validate the ECPay environment. Called from FastAPI startup.

    Never raises: an invalid setting is recorded in _CONFIG_ERROR and surfaces
    as a 503 on the payment endpoints when get_config() is called. Never falls
    back to a default environment.

    Idempotent — both fields are reset first, so calling this again after
    fixing the environment fully re-evaluates it.
    """
    global _CONFIG, _CONFIG_ERROR
    _CONFIG, _CONFIG_ERROR = None, None

    env = (os.getenv("ECPAY_ENV", "") or "").strip().lower()
    merchant_id = (os.getenv("ECPAY_MERCHANT_ID", "") or "").strip()

    # The offending value is deliberately not echoed. It is not supposed to be
    # secret, but "never log credential material" is cheaper to keep absolute
    # than to keep conditional, and the valid set has two members.
    if not env:
        _CONFIG_ERROR = "ECPAY_ENV is not set"
    elif env not in _ACTION_URLS:
        _CONFIG_ERROR = "ECPAY_ENV must be 'stage' or 'production'"
    elif not merchant_id:
        _CONFIG_ERROR = "ECPAY_MERCHANT_ID is not set"
    else:
        _CONFIG = EcpayConfig(merchant_id, env, _ACTION_URLS[env])


def get_config() -> EcpayConfig:
    """The validated config, or EcpayConfigError describing why there isn't one."""
    if _CONFIG_ERROR:
        raise EcpayConfigError(_CONFIG_ERROR)
    if _CONFIG is None:
        raise EcpayConfigError("ECPay configuration has not been loaded")
    return _CONFIG


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


def ecpay_urlencode(value) -> str:
    """.NET-compatible URL encoding, lowercased — ECPay's exact dialect.

    Space becomes '+', percent escapes are lowercase, and the seven characters
    above stay literal. Extracted verbatim from the body of
    _ecpay_check_mac_value so it can be tested character by character; the
    composed behaviour is unchanged.

    str() coercion is deliberate: a caller passing TotalAmount as an int would
    otherwise raise inside quote_plus rather than sign the obvious value.
    """
    encoded = quote_plus(str(value), safe="").lower()
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

    return hashlib.sha256(ecpay_urlencode(raw).encode("utf-8")).hexdigest().upper()


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
_TRADE_NO_RANDOM_WIDTH = 12


def generate_merchant_trade_no(now: datetime | None = None) -> str:
    """A unique, ECPay-legal MerchantTradeNo: alphanumeric only, 20 chars.

    Shape: yyyyMMdd (8) + 12 random uppercase alphanumerics = 20 exactly.

    The old f"BLABBY-{uuid4().hex[:12].upper()}" was 19 characters — legal on
    length, illegal on charset: ECPay allows 英數字大小寫混合 only, and the
    hyphen was the single violation.

    No 'BLB' prefix. Anything matching `like 'BLB%'` therefore matches nothing;
    clean-up queries must filter on user_id (see
    docs/ECPAY_STAGE_VERIFICATION.md).

    36^12 ≈ 4.7e18 keeps the birthday bound irrelevant: 10,000 draws inside one
    date bucket collide with probability ~1e-11.
    """
    stamp = (now or datetime.now(_TAIPEI)).strftime("%Y%m%d")
    suffix = "".join(
        secrets.choice(_TRADE_NO_ALPHABET) for _ in range(_TRADE_NO_RANDOM_WIDTH)
    )
    return f"{stamp}{suffix}"


def merchant_trade_date(now: datetime | None = None) -> str:
    """MerchantTradeDate in ECPay's required yyyy/MM/dd HH:mm:ss, Taipei time."""
    return (now or datetime.now(_TAIPEI)).strftime("%Y/%m/%d %H:%M:%S")


# ─── AioCheckOut ─────────────────────────────────────────────────────────────
# ECPay rejects "special characters" in TradeDesc/ItemName without enumerating
# them, and '&' or '<' anywhere in the payload breaks the cashier's own parsing.
# Rather than guess at the forbidden set, allow a known-good one.
_TRADE_TEXT_ALLOWED = frozenset(string.ascii_letters + string.digits + " -")


def sanitize_trade_text(text: str, limit: int) -> str:
    """Reduce free text to the characters ECPay is certain to accept.

    Whitelist, not blacklist: ASCII letters, digits, spaces and hyphens survive;
    everything else is dropped. Runs of spaces collapse so a stripped character
    does not leave a gap, and the result is truncated to the field's limit.

    This also removes the '!' '*' '(' ')' '~' cases entirely from the outbound
    payload, which is why the known-answer vectors do not need to cover the
    .NET-vs-Python divergence on those characters for text we generate.
    """
    kept = "".join(ch if ch in _TRADE_TEXT_ALLOWED else " " for ch in text)
    return " ".join(kept.split())[:limit].strip()


# Plain ASCII, ancient-English register, already inside the whitelist.
TRADE_DESC = sanitize_trade_text(
    "Subscription to the Blabby Pro Membership for a Term of Thirty Days", 200)
ITEM_NAME = sanitize_trade_text("Blabby Pro Membership - Thirty Days", 400)

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

    Credit, not ALL. ALL exposes ATM and 超商代碼, which are non-realtime: the
    buyer gets a payment code, leaves, and pays hours or days later. That splits
    the callback into an initial "code issued" notification and a later "paid"
    one, with a different RtnCode sequence and a pending window this handler has
    no state machine for. Phase 1 grants on a single realtime authorisation.

    The caller POSTs these to EcpayConfig.action_url as a form. CheckMacValue is
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
