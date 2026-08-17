"""GET /api/payment/_diag —— 唯讀環境診斷端點。

存在的理由：production 上綠界回 10200074（找不到商店），而從外面看不出
成因是商店號屬於另一個環境、值尾端夾帶換行、還是 ECPAY_ENV 沒解析成以為
的那一邊 —— 三者症狀一模一樣。端點把執行期事實攤開，不猜。

這裡的測試不需要任何真實憑證，也不打綠界。
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

import ecpay
import main


# TestClient 不進 context manager：進去會跑 lifespan，啟動 APScheduler 並排入
# writing 題目預產生。純請求不需要 lifespan。（同 test_ecpay.py 的理由）
client = TestClient(main.app)

_VALID_ENV = {
    "ECPAY_MERCHANT_ID":   "3002607",
    "PUBLIC_BACKEND_URL":  "https://api.example.com",
    "PUBLIC_FRONTEND_URL": "https://app.example.com",
}


@pytest.fixture
def ecpay_env(monkeypatch):
    """設一組環境變數並重新 load_config()，離開時把模組全域還原。

    monkeypatch 只還原 os.environ；ecpay._CONFIG / _CONFIG_ERROR 是模組全域，
    不還原會汙染同一個 session 裡的其他測試。
    """
    def _apply(**overrides):
        for key, value in dict(_VALID_ENV, **overrides).items():
            if value is None:
                monkeypatch.delenv(key, raising=False)
            else:
                monkeypatch.setenv(key, value)
        ecpay.load_config()

    yield _apply

    monkeypatch.undo()
    ecpay.load_config()


def _get(**patches):
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_admin", MagicMock(return_value="admin-uid")), \
         patch.object(main, "ECPAY_HASH_KEY", patches.get("hash_key", "")), \
         patch.object(main, "ECPAY_HASH_IV", patches.get("hash_iv", "")):
        return client.get("/api/payment/_diag",
                          headers={"Authorization": "Bearer t"})


# ── 1. 非 admin 一律 403 ─────────────────────────────────────────────────
# 走真實的 verify_admin，不 patch 它 —— 這條測試唯一的價值就是證明那條路徑
# 真的擋得住，patch 掉就什麼也沒測到。

def _supabase_returning(email: str):
    fake = MagicMock()
    fake.auth.admin.get_user_by_id.return_value = SimpleNamespace(
        user=SimpleNamespace(email=email)
    )
    return fake


def test_non_admin_gets_403():
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", MagicMock(return_value="user-1")), \
         patch.object(main, "supabase_admin", _supabase_returning("nobody@example.com")), \
         patch.object(main, "ADMIN_EMAILS", {"admin@example.com"}):
        response = client.get("/api/payment/_diag",
                              headers={"Authorization": "Bearer t"})

    assert response.status_code == 403
    assert response.json()["detail"] == "Admin access required"


def test_missing_authorization_header_is_rejected():
    """沒有 fallback：無 header 不會退成匿名可讀。"""
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "supabase_admin", _supabase_returning("admin@example.com")):
        response = client.get("/api/payment/_diag")

    assert response.status_code in (401, 403)


def test_admin_email_passes():
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_token", MagicMock(return_value="user-1")), \
         patch.object(main, "supabase_admin", _supabase_returning("Admin@Example.com")), \
         patch.object(main, "ADMIN_EMAILS", {"admin@example.com"}):
        response = client.get("/api/payment/_diag",
                              headers={"Authorization": "Bearer t"})

    assert response.status_code == 200


# ── 2. production 解析 ───────────────────────────────────────────────────

def test_production_resolves_and_never_points_at_stage(ecpay_env):
    ecpay_env(ECPAY_ENV="production")
    body = _get().json()

    assert body["resolved_env"] == "production"
    assert body["aio_checkout_url"] == \
        "https://payment.ecpay.com.tw/Cashier/AioCheckOut/V5"
    assert "payment-stage" not in body["aio_checkout_url"]
    assert body["ecpay_env_raw"] == "production"


def test_stage_resolves_to_the_stage_cashier(ecpay_env):
    ecpay_env(ECPAY_ENV="stage")
    body = _get().json()

    assert body["resolved_env"] == "stage"
    assert "payment-stage" in body["aio_checkout_url"]


# ── 3. 髒值：把未定義行為釘成已知行為 ────────────────────────────────────

def test_padded_production_value_is_stripped_and_still_resolves(ecpay_env):
    """現行行為：ecpay.py:114 對 ECPAY_ENV 做了 .strip().lower()，所以
    " production\\n" 解析成功，不落入 invalid。

    這條測試不主張這是對的設計，只主張這是現在的行為 —— 診斷輸出裡
    ecpay_env_repr 會照實顯示 "' production\\n'"，讓髒值仍然看得見。
    """
    ecpay_env(ECPAY_ENV=" production\n")
    body = _get().json()

    assert body["resolved_env"] == "production"
    assert "payment-stage" not in body["aio_checkout_url"]
    assert body["ecpay_env_raw"] == " production\n"
    assert body["ecpay_env_repr"] == repr(" production\n")


def test_uppercase_env_also_resolves(ecpay_env):
    """.lower() 的既有行為，同樣釘住。"""
    ecpay_env(ECPAY_ENV="PRODUCTION")
    assert _get().json()["resolved_env"] == "production"


def test_typo_env_is_invalid_with_no_urls_and_a_warning(ecpay_env):
    """'prod' 不 fallback 到任何一邊 —— 這是 ecpay.py 的核心不變式。"""
    ecpay_env(ECPAY_ENV="prod")
    body = _get().json()

    assert body["resolved_env"] == "invalid"
    assert body["aio_checkout_url"] is None
    assert body["return_url"] is None
    assert body["client_back_url"] is None
    assert body["order_result_url"] is None
    assert any("ecpay config unavailable" in w for w in body["warnings"])


def test_unset_env_is_invalid(ecpay_env):
    ecpay_env(ECPAY_ENV=None)
    body = _get().json()

    assert body["resolved_env"] == "invalid"
    assert body["ecpay_env_raw"] is None
    assert body["ecpay_env_repr"] == "None"


def test_padded_merchant_id_is_surfaced(ecpay_env):
    """MerchantID 讀原始環境變數，不是 config.merchant_id —— 後者已經
    strip 過，看它永遠看不到尾端空白，而尾端空白正是 10200074 的候選成因。
    """
    ecpay_env(ECPAY_MERCHANT_ID="3002607\n", ECPAY_ENV="production")
    body = _get().json()

    assert body["merchant_id_len"] == 8
    assert body["merchant_id_stripped_differs"] is True
    assert body["merchant_id_masked"] == "****607"
    assert any("ECPAY_MERCHANT_ID has leading/trailing whitespace" in w
               for w in body["warnings"])
    # 環境仍然有效：strip 之後商店號是合法的
    assert body["resolved_env"] == "production"


# ── 4. 憑證指紋 ─────────────────────────────────────────────────────────
# 已知向量，不需要真實憑證：sha256("known-key") / sha256("known-iv")。

def test_fingerprint_and_len_for_known_strings(ecpay_env):
    import hashlib

    ecpay_env(ECPAY_ENV="production")
    body = _get(hash_key="known-key", hash_iv="known-iv").json()

    assert body["hash_key_fingerprint"] == \
        hashlib.sha256(b"known-key").hexdigest()[:8]
    assert body["hash_key_len"] == 9
    assert body["hash_key_stripped_differs"] is False

    assert body["hash_iv_fingerprint"] == \
        hashlib.sha256(b"known-iv").hexdigest()[:8]
    assert body["hash_iv_len"] == 8
    assert body["hash_iv_stripped_differs"] is False


def test_padded_hash_key_is_surfaced(ecpay_env):
    ecpay_env(ECPAY_ENV="production")
    body = _get(hash_key="known-key\n", hash_iv="known-iv").json()

    assert body["hash_key_len"] == 10
    assert body["hash_key_stripped_differs"] is True
    assert any("ECPAY_HASH_KEY has leading/trailing whitespace" in w
               for w in body["warnings"])


def test_empty_credentials_warn_instead_of_500(ecpay_env):
    ecpay_env(ECPAY_ENV="production")
    response = _get(hash_key="", hash_iv="")

    assert response.status_code == 200
    body = response.json()
    assert "ECPAY_HASH_KEY is empty" in body["warnings"]
    assert "ECPAY_HASH_IV is empty" in body["warnings"]
    # 空字串仍然有指紋，不是 None —— sha256 從不吃 None
    assert len(body["hash_key_fingerprint"]) == 8


# ── 5. 不洩漏、不打綠界 ─────────────────────────────────────────────────

def test_no_plaintext_credentials_in_the_response(ecpay_env):
    ecpay_env(ECPAY_MERCHANT_ID="3002607", ECPAY_ENV="production")
    raw = _get(hash_key="known-key", hash_iv="known-iv").text

    assert "known-key" not in raw
    assert "known-iv" not in raw
    assert "3002607" not in raw
    assert "****607" in raw


def test_env_var_names_are_listed_without_values(ecpay_env):
    ecpay_env(ECPAY_ENV="production")
    body = _get(hash_key="known-key").json()

    names = body["ecpay_env_var_names_present"]
    assert names == sorted(names)
    assert "ECPAY_ENV" in names
    assert "ECPAY_MERCHANT_ID" in names
    assert all(name.startswith("ECPAY") for name in names)


def test_dotenv_override_mirrors_the_call_site(ecpay_env):
    """main.py:118 是 load_dotenv()，沒有 override=True。若改了那行，這條
    測試要一起改 —— 這正是它的用途。"""
    import inspect
    import re

    ecpay_env(ECPAY_ENV="production")
    assert _get().json()["dotenv_override"] is False

    source = inspect.getsource(main)
    calls = re.findall(r"^load_dotenv\((.*)\)$", source, re.MULTILINE)
    assert calls == [""], f"load_dotenv 的呼叫參數變了：{calls}"


def test_trade_date_sample_is_taipei_time(ecpay_env):
    """MerchantTradeDate 用 UTC+8 —— 與 server_utc_now 的差必須是 8 小時。"""
    from datetime import datetime

    ecpay_env(ECPAY_ENV="production")
    body = _get().json()

    utc_now = datetime.fromisoformat(body["server_utc_now"])
    sample = datetime.strptime(body["merchant_trade_date_sample"],
                               "%Y/%m/%d %H:%M:%S")
    offset = sample - utc_now.replace(tzinfo=None)
    assert 7.9 * 3600 < offset.total_seconds() < 8.1 * 3600, offset


def test_endpoint_makes_no_outbound_request(ecpay_env):
    """診斷本身絕不觸發任何對綠界的請求。"""
    import requests

    ecpay_env(ECPAY_ENV="production")
    with patch.object(requests, "post", MagicMock(side_effect=AssertionError)), \
         patch.object(requests, "get", MagicMock(side_effect=AssertionError)):
        assert _get().status_code == 200
