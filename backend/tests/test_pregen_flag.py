"""PREGEN_ENABLED — 排程預生成的總開關。

背景：Speaking 最後一次真實使用 2026-07-31，Reading 2026-08-11。零流量期間
worker 每 6 小時仍呼叫 Anthropic 生成無人閱讀的內容，那是純支出。

兩個性質必須同時成立：
  1. 預設啟用 —— 沒設變數、設空字串、設垃圾值，worker 都照跑。預設關閉會讓
     日後清空環境變數的人靜默停掉 worker，而那個失效沒有症狀。
  2. 停用時 job 根本不在 scheduler 上，不是註冊後在 job 內 early return ——
     後者會讓 scheduler 每 6 小時照常喚醒並記錄執行成功，log 開始說謊。

第 2 點只能靠實際檢查 scheduler 的 job 列表來證明，所以每個案例跑在自己的
子行程裡（見 _pregen_flag_probe.py）。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from unittest.mock import patch

import pytest

import main


_HERE = os.path.dirname(os.path.abspath(__file__))
_PROBE = os.path.join(_HERE, "_pregen_flag_probe.py")

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

PREGEN_JOB_IDS = {
    "nightly_writing_pregen",
    "nightly_reading_pregen",
    "startup_writing_pregen",
    "startup_reading_pregen",
}


def _startup(**overrides) -> tuple[dict, str]:
    env = dict(_BASE_ENV, **overrides)
    proc = subprocess.run(
        [sys.executable, _PROBE],
        env=env, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"probe crashed:\n{proc.stderr[-4000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1]), proc.stderr


# ── 預設啟用 ─────────────────────────────────────────────────────────────

def test_unset_flag_registers_every_pregen_job():
    result, stderr = _startup()

    assert result["pregen_enabled"] is True
    assert PREGEN_JOB_IDS <= set(result["job_ids"]), result["job_ids"]
    assert "pregeneration workers: enabled" in stderr


@pytest.mark.parametrize("value", ["", "   ", "true", "TRUE", "1", "on", "yes",
                                   "banana"])
def test_anything_that_is_not_an_explicit_off_keeps_workers_running(value):
    """垃圾值也照跑。停用必須是有人明確寫下 false 的結果，不是打錯字的結果。"""
    result, _ = _startup(PREGEN_ENABLED=value)

    assert result["pregen_enabled"] is True, value
    assert PREGEN_JOB_IDS <= set(result["job_ids"]), value


# ── 明確停用 ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize("value", ["false", "0", "off", "FALSE", "  false  ",
                                   "Off"])
def test_explicit_off_removes_every_pregen_job_from_the_scheduler(value):
    result, stderr = _startup(PREGEN_ENABLED=value)

    assert result["pregen_enabled"] is False, value
    assert PREGEN_JOB_IDS.isdisjoint(result["job_ids"]), result["job_ids"]
    assert "pregeneration workers: DISABLED by PREGEN_ENABLED" in stderr


def test_disabling_pregen_leaves_the_billing_sweeps_alone():
    """旗標只關預生成。同一個 scheduler 上的其他 job 不受影響。

    2026-09-03：清單從一支變兩支。lapse_expired_active_subscriptions 刻意
    註冊在 PREGEN_ENABLED 判斷之外 —— 狀態機的正確性不該跟著內容生成的
    成本開關一起被關掉。斷言維持完全相等而不是放寬成 issubset：這條要守
    的是「旗標關掉時 scheduler 上只剩帳務 job」，多出任何一支都該讓它紅。
    """
    result, _ = _startup(PREGEN_ENABLED="false")

    assert result["job_ids"] == [
        "expire_stale_pending_subscriptions",
        "lapse_expired_active_subscriptions",
    ]


def test_startup_prime_is_disabled_too():
    """startup prime 同樣呼叫 Anthropic —— 留著等於每次 Render 重啟就生成一批。

    單獨一條，因為這是最容易在「只關掉 6-hourly」時漏掉的那兩個 job。
    """
    result, _ = _startup(PREGEN_ENABLED="false")

    assert "startup_writing_pregen" not in result["job_ids"]
    assert "startup_reading_pregen" not in result["job_ids"]


# ── on-demand 補池不受旗標影響 ───────────────────────────────────────────

def test_on_demand_replenish_still_runs_when_pregen_is_disabled():
    """_replenish_reading_async 是使用者實際取用時才觸發的補池，有流量才會跑，
    不是排程支出。旗標不該碰它 —— 碰了就會在 pool 見底時無聲地停止補充。
    """
    called: list[str] = []

    with patch.object(main, "PREGEN_ENABLED", False), \
         patch.object(main, "_pregenerate_reading_topic",
                      lambda topic, *a, **k: called.append(topic)):
        main._replenish_reading_async("general")
        # fire-and-forget：等那條 thread 收工再斷言
        for thread in list(main.threading.enumerate()):
            if thread is not main.threading.current_thread():
                thread.join(timeout=5)

    assert called == ["general"]


def test_replenish_does_not_read_the_flag_at_all():
    """比行為測試更直接：那個函式的原始碼裡不該出現這個旗標。"""
    import inspect

    source = inspect.getsource(main._replenish_reading_async)
    assert "PREGEN_ENABLED" not in source
