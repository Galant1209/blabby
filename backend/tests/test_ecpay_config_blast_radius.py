"""設定失效的爆炸半徑：金流壞掉時，其他功能必須完全不受影響。

把一個真實教訓編碼成測試。前一版在 module scope 驗證 ECPAY_ENV：

    ECPAY_ENV = _resolve_env(os.getenv("ECPAY_ENV", ""))

`ECPAY_ENV=prod` 這種打錯字會讓 `import main` 直接 raise ValueError ——
不是金流壞掉，是整台 backend 起不來，Part 1 / Part 2 / Reading / Writing
全部一起死。與 main.py:128/220 讓 7 個測試模組在乾淨環境全爆是同一類 bug。

四個不可退的性質（規格步驟 A）：
  1. import 時零 raise
  2. 無效值絕不 fallback —— 不落回 stage、不落回 production
  3. 爆炸半徑 = 金流端點 503，其餘照常
  4. 必須大聲 —— startup CRITICAL log + /health 的 billing_config_ok

每個案例都跑在**全新的子行程**裡。性質 1 在「main 已經 import 過」的
行程中根本無法觀察，in-process 的測試會是假的。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE = os.path.join(_HERE, "_blast_radius_probe.py")

_BASE_ENV = {
    "PATH":                 os.environ.get("PATH", ""),
    "HOME":                 os.environ.get("HOME", ""),
    "APP_ENV":              "development",
    "SUPABASE_URL":         "",
    "SUPABASE_SERVICE_KEY": "",
    "GROQ_API_KEY":         "test-key",
    "OPENAI_API_KEY":       "test-key",
    "ANTHROPIC_API_KEY":    "test-key",
}


def _probe(**overrides) -> dict:
    env = dict(_BASE_ENV, **overrides)
    proc = subprocess.run(
        [sys.executable, _PROBE],
        env=env, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"probe crashed:\n{proc.stderr[-4000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


# ── the broken-config case ───────────────────────────────────────────────
@pytest.fixture(scope="module")
def broken():
    """ECPAY_ENV='prod' — the exact typo that used to take the service down."""
    return _probe(ECPAY_ENV="prod", ECPAY_MERCHANT_ID="3002607")


def test_1_import_main_succeeds_despite_the_typo(broken):
    assert broken["import_ok"] is True, broken.get("import_error")


def test_2_payment_endpoints_are_the_only_casualties(broken):
    assert broken["create_order_status"] == 503
    assert broken["callback_status"] == 503


@pytest.mark.parametrize("area", ["part1", "part2", "reading", "writing"])
def test_3_every_other_product_area_still_serves_200(broken, area):
    assert broken[f"{area}_status"] == 200, (
        f"{area} regressed to {broken[f'{area}_status']} because of a payment "
        "configuration error — the blast radius leaked"
    )


def test_4a_startup_emits_a_critical_log(broken):
    criticals = broken["critical_logs"]
    assert criticals, "a silent degradation is worse than an outage"
    assert any("[BILLING]" in line for line in criticals)
    assert any("503" in line for line in criticals)


def test_4b_health_reports_billing_config_ok_false(broken):
    assert broken["health_status"] == 200, "the service itself stays healthy"
    assert broken["health_body"]["billing_config_ok"] is False


def test_4c_the_critical_log_never_echoes_the_offending_value(broken):
    """'prod' is harmless, but the rule is absolute so it cannot rot."""
    assert not any("'prod'" in line for line in broken["critical_logs"])


# ── the healthy case, so the assertions above can actually fail ──────────
@pytest.fixture(scope="module")
def healthy():
    return _probe(
        ECPAY_ENV="stage",
        ECPAY_MERCHANT_ID="3002607",
        ECPAY_HASH_KEY="test-hash-key",
        ECPAY_HASH_IV="test-hash-iv",
        PUBLIC_BACKEND_URL="https://api.example.com",
        PUBLIC_FRONTEND_URL="https://app.example.com",
    )


def test_healthy_config_reports_ok_and_emits_no_critical(healthy):
    assert healthy["health_body"]["billing_config_ok"] is True
    assert healthy["critical_logs"] == []


def test_healthy_config_lets_create_order_through(healthy):
    """Not 503. (500 here: the probe's fake supabase returns an empty insert
    row, which is gate 4 doing its job — the point is the config gate opened.)"""
    assert healthy["create_order_status"] != 503


# ── the other invalid shapes, defined rather than accidental ─────────────
@pytest.mark.parametrize("env,label", [
    ({"ECPAY_ENV": "prod"},                              "typo"),
    ({"ECPAY_ENV": ""},                                  "empty"),
    ({"ECPAY_ENV": "   "},                               "whitespace"),
    ({},                                                 "unset"),
    ({"ECPAY_ENV": "stage", "ECPAY_MERCHANT_ID": ""},    "no merchant id"),
])
def test_every_invalid_configuration_degrades_the_same_way(env, label):
    payload = _probe(**dict({"ECPAY_MERCHANT_ID": "3002607"}, **env))
    assert payload["import_ok"] is True, label
    assert payload["health_body"]["billing_config_ok"] is False, label
    assert payload["create_order_status"] == 503, label
    assert payload["part2_status"] == 200, label


@pytest.mark.parametrize("value", ["Stage", "PRODUCTION", " stage "])
def test_case_and_whitespace_variants_are_accepted(value):
    """Defined behaviour: normalised, not rejected, and never guessed at."""
    payload = _probe(ECPAY_ENV=value, ECPAY_MERCHANT_ID="3002607")
    assert payload["health_body"]["billing_config_ok"] is True, value
