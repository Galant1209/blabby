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
    return_url: str            # server-to-server, signed — the only grant path
    client_back_url: str       # the "back to shop" button
    order_result_url: str      # where the buyer's browser lands


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
    backend = (os.getenv("PUBLIC_BACKEND_URL", "") or "").strip().rstrip("/")
    frontend = (os.getenv("PUBLIC_FRONTEND_URL", "") or "").strip().rstrip("/")

    return_url = f"{backend}/api/payment/callback"
    client_back_url = f"{frontend}/success.html"
    order_result_url = f"{backend}/api/payment/return"

    # The offending value is deliberately not echoed. It is not supposed to be
    # secret, but "never log credential material" is cheaper to keep absolute
    # than to keep conditional, and the valid set has two members.
    if not env:
        _CONFIG_ERROR = "ECPAY_ENV is not set"
    elif env not in _ACTION_URLS:
        _CONFIG_ERROR = "ECPAY_ENV must be 'stage' or 'production'"
    elif not merchant_id:
        _CONFIG_ERROR = "ECPAY_MERCHANT_ID is not set"
    elif not backend or not frontend:
        _CONFIG_ERROR = "PUBLIC_BACKEND_URL / PUBLIC_FRONTEND_URL are not set"
    elif not (backend.startswith("https://") and frontend.startswith("https://")):
        _CONFIG_ERROR = "PUBLIC_BACKEND_URL / PUBLIC_FRONTEND_URL must be absolute HTTPS"
    elif return_url == order_result_url:
        # ECPay's spec states outright that ReturnURL cannot equal
        # OrderResultURL. Asserted here rather than trusted to memory: if they
        # ever collided, the unsigned browser GET and the signed
        # server-to-server POST would arrive at the same handler, and the
        # difference between them is the entire security model.
        _CONFIG_ERROR = "ReturnURL and OrderResultURL must not be identical"
    else:
        _CONFIG = EcpayConfig(
            merchant_id, env, _ACTION_URLS[env],
            return_url, client_back_url, order_result_url,
        )


def get_config() -> EcpayConfig:
    """The validated config, or EcpayConfigError describing why there isn't one."""
    if _CONFIG_ERROR:
        raise EcpayConfigError(_CONFIG_ERROR)
    if _CONFIG is None:
        raise EcpayConfigError("ECPay configuration has not been loaded")
    return _CONFIG


def simulated_payment_allowed() -> bool:
    """True only when we positively know we are pointed at the test cashier.

    Fail-safe: an unloaded or broken config answers False, so a simulated
    payment is refused whenever we cannot prove we are on stage.
    """
    try:
        return get_config().env == "stage"
    except EcpayConfigError:
        return False


def is_simulated_paid(params: dict) -> bool:
    """Whether a callback describes a *simulated* payment.

    ECPay's merchant console has a "模擬付款" button in production too. It
    delivers a fully signed callback with RtnCode=1 and SimulatePaid=1 while no
    money moves at all — a free Pro subscription for anyone with console access.

    Absent means a real payment: ECPay sends 0 on genuine ones, and older
    flows may omit the field. Anything present and not "0" counts as simulated,
    so an unparseable value refuses the grant rather than allowing it.
    """
    raw = (params.get("SimulatePaid") or "").strip()
    return bool(raw) and raw != "0"


# ─── CheckMacValue ───────────────────────────────────────────────────────────
# ECPay generates the digest with .NET's HttpUtility.UrlEncode, which leaves
# these seven characters unescaped where Python's quote_plus escapes them.
# Without the replacements every signature mismatches. The official Python SDK
# expresses the same rule as quote_plus(s, safe='-_.!*()'); the replacement form
# is kept because it is what 8e11b93 shipped and what the docs spell out.
_NET_URLENCODE_REPLACEMENTS = (
    ("%2d", "-"), ("%5f", "_"), ("%2e", "."), ("%21", "!"),
    ("%2a", "*"), ("%28", "("), ("%29", ")"),
    # ~ goes the OTHER WAY. Per the official conversion table's third column
    # (.NET URLEncode 結果, https://developers.ecpay.com.tw/?p=2904, fetched
    # 2026-07-30), '~' is the single character in that group that .NET *does*
    # escape, to %7E. Python's quote_plus treats it as unreserved and leaves it
    # literal, so without this pair any value containing '~' produces a digest
    # ECPay's server disagrees with.
    #
    # Note this also diverges from ECPay's own Python SDK, which computes
    # quote_plus(s, safe='-_.!*()') and therefore leaves '~' literal too. The
    # authority is the server that verifies the signature, not the SDK.
    ("~", "%7e"),
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
# This is a dropped-order prevention mechanism, not cosmetics. ECPay truncates
# an over-length ItemName server-side, and a truncation that lands mid-character
# produces mojibake — which changes the bytes the cashier hashes, breaks
# CheckMacValue, and loses the order. Two rules follow:
#
#   * truncate by CHARACTER, never by byte. Python slicing is character-based,
#     which is the whole reason this is written as text[:limit] and not as an
#     encode/slice/decode.
#   * whitelist rather than blacklist. ECPay warns against "特殊字元" without
#     enumerating them, and '&' or '<' breaks the cashier's own parsing.
#
# Chinese is allowed: ECPay accepts UTF-8 CJK in these fields and the vectors
# prove the digest handles it. Excluding ~ ! * ( ) also keeps every value we
# send clear of the .NET-vs-Python encoding divergence.
TRADE_DESC_MAX = 200
ITEM_NAME_MAX = 400

_TRADE_TEXT_ASCII = frozenset(string.ascii_letters + string.digits + " -")
_CJK_RANGES = (
    (0x3000, 0x303F),    # CJK punctuation
    (0x4E00, 0x9FFF),    # CJK unified ideographs
    (0xFF00, 0xFFEF),    # full-width forms
)


def _is_allowed_trade_char(ch: str) -> bool:
    if ch in _TRADE_TEXT_ASCII:
        return True
    code = ord(ch)
    return any(low <= code <= high for low, high in _CJK_RANGES)


def sanitize_trade_text(text: str, limit: int) -> str:
    """Reduce free text to characters ECPay is certain to accept, then truncate.

    Whitelist: ASCII letters, digits, spaces, hyphens, and CJK. Everything else
    becomes a space; runs of spaces collapse so a stripped character leaves no
    gap. Truncation is by character, so a multi-byte character is never cut in
    half — a half-character is mojibake, and mojibake is a broken CheckMacValue.
    """
    kept = "".join(ch if _is_allowed_trade_char(ch) else " " for ch in text)
    return " ".join(kept.split())[:limit].strip()


# Deliberately far below the 400-character ceiling: the limit is a backstop,
# not a target.
TRADE_DESC = sanitize_trade_text(
    "Subscription to the Blabby Pro Membership for a Term of Thirty Days",
    TRADE_DESC_MAX)
ITEM_NAME = sanitize_trade_text(
    "Blabby Pro Membership - Thirty Days", ITEM_NAME_MAX)

# Field limits from the AioCheckOut V5 spec (developers.ecpay.com.tw/?p=2862).
_MAX_LENGTHS = {
    "MerchantID": 10, "MerchantTradeNo": 20, "MerchantTradeDate": 20,
    "PaymentType": 20, "TradeDesc": TRADE_DESC_MAX, "ItemName": ITEM_NAME_MAX,
    "ReturnURL": 200, "ChoosePayment": 20,
    "ClientBackURL": 200, "OrderResultURL": 200,
    "CustomField1": 50,
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
    user_id: str,
    trade_date: str | None = None,
) -> dict:
    """The complete, signed parameter set for a one-off Credit charge.

    Blabby Pro is a 30-day period the buyer chooses to purchase, every time.
    There are no Period* parameters here because there is no auto-renewal —
    a deliberate product decision (2026-07-30), not an unbuilt feature. A
    product that refuses to retain through gamification does not get to retain
    through "forgot to cancel" either.

    Credit, not ALL, and never IgnorePayment. Two reasons, both from ECPay's
    own guidance:
      * ATM and 超商代碼 are non-realtime — the buyer gets a code, leaves, and
        pays hours later. That splits the notification into "code issued" then
        "paid", with a pending window this handler has no state machine for.
      * ECPay keeps adding payment methods. ALL would silently route each new
        one into a callback path nobody has written code for.

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

    if not user_id:
        raise ValueError("user_id is required for the CustomField1 cross-check")

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
        # Echoed back in the callback and covered by CheckMacValue, so the
        # callback can cross-check the buyer it thinks it is granting to
        # against the buyer who actually started the checkout. A UUID is 36
        # characters and the field holds 50.
        "CustomField1":      user_id,
        # Ask for the full paid-detail set. We do not read the extra fields
        # today, but they all land in payment_events.raw_payload (jsonb, no
        # schema change), and reconciliation data you did not capture can only
        # be recovered by charging someone again.
        "NeedExtraPaidInfo": "Y",
    }

    for field, limit in _MAX_LENGTHS.items():
        value = params.get(field, "")
        if len(value) > limit:
            raise ValueError(f"{field} exceeds ECPay's {limit}-character limit")

    params["CheckMacValue"] = build_check_mac_value(params, hash_key, hash_iv)
    return params
