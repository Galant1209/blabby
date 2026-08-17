"""HashKey / HashIV 的空白防護。

這兩個值直接餵進 CheckMacValue 的雜湊輸入。從 Render 後台貼值時夾帶一個
尾端換行，簽章就對不起來，而綠界只會回「CheckMacValue 錯誤」—— 從那裡回推
到「環境變數尾端有 \\n」極其昂貴。ecpay.py 的 load_config() 對 ECPAY_ENV /
ECPAY_MERCHANT_ID / PUBLIC_*_URL 早就 strip，這兩個是最後補齊的。

測試跑在**全新的子行程**裡：main.py 在 import 時求值一次這兩個變數，
in-process 改 os.environ 之後再讀 main.ECPAY_HASH_KEY 讀到的是舊值，
那樣的測試會是假的。（同 test_ecpay_config_blast_radius.py 的理由）
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from ecpay import build_check_mac_value

_HERE = os.path.dirname(os.path.abspath(__file__))
_BACKEND_DIR = os.path.abspath(os.path.join(_HERE, os.pardir))

# 綠界公開的測試特店金鑰，非正式憑證 —— 與 test_ecpay_known_answer.py 同源。
CLEAN_KEY = "pwFHCqoQZGmho4w6"
CLEAN_IV = "EkRm7iFT261dpevs"

# 固定參數，讓兩次執行唯一的變因是金鑰前後的空白。
VECTOR_PARAMS = {
    "MerchantID":      "3002607",
    "MerchantTradeNo": "20260817PROBE0000001",
    "TotalAmount":     "199",
}

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

# 印出 main 在 import 時求到的兩個值，以及用它們簽出來的 CheckMacValue。
# 兩者都要：值本身證明 strip 生效，簽章證明 strip 生效在**簽章實際會用到的
# 那個變數上**，而不是某個副本。
_PROBE = (
    "import json, sys; sys.path.insert(0, %r); "
    "import main, ecpay; "
    "print(json.dumps({"
    "'key': main.ECPAY_HASH_KEY, 'iv': main.ECPAY_HASH_IV, "
    "'mac': ecpay.build_check_mac_value(%r, main.ECPAY_HASH_KEY, main.ECPAY_HASH_IV)"
    "}))"
) % (_BACKEND_DIR, VECTOR_PARAMS)


def _import_main_with(hash_key: str, hash_iv: str) -> dict:
    env = dict(_BASE_ENV, ECPAY_HASH_KEY=hash_key, ECPAY_HASH_IV=hash_iv)
    proc = subprocess.run(
        [sys.executable, "-c", _PROBE],
        env=env, capture_output=True, text=True, timeout=180,
    )
    assert proc.returncode == 0, f"probe crashed:\n{proc.stderr[-4000:]}"
    return json.loads(proc.stdout.strip().splitlines()[-1])


@pytest.fixture(scope="module")
def clean() -> dict:
    return _import_main_with(CLEAN_KEY, CLEAN_IV)


DIRTY = [
    ("尾端換行",   f"{CLEAN_KEY}\n",     f"{CLEAN_IV}\n"),
    ("前後空白",   f"  {CLEAN_KEY}  ",   f"  {CLEAN_IV}  "),
    ("尾端 CRLF",  f"{CLEAN_KEY}\r\n",   f"{CLEAN_IV}\r\n"),
    ("尾端 tab",   f"{CLEAN_KEY}\t",     f"{CLEAN_IV}\t"),
]


@pytest.mark.parametrize("label,dirty_key,dirty_iv", DIRTY,
                         ids=[case[0] for case in DIRTY])
def test_module_level_values_are_stripped(label, dirty_key, dirty_iv):
    result = _import_main_with(dirty_key, dirty_iv)

    assert result["key"] == CLEAN_KEY, label
    assert result["iv"] == CLEAN_IV, label


@pytest.mark.parametrize("label,dirty_key,dirty_iv", DIRTY,
                         ids=[case[0] for case in DIRTY])
def test_dirty_keys_sign_identically_to_clean_ones(label, dirty_key, dirty_iv,
                                                   clean):
    """帶空白版與不帶空白版必須算出同一個 CheckMacValue。

    這條才是這個 fix 存在的理由 —— 值被 strip 是手段，簽章一致是目的。
    """
    assert _import_main_with(dirty_key, dirty_iv)["mac"] == clean["mac"], label


def test_clean_signature_matches_an_in_process_known_answer(clean):
    """子行程算出來的簽章要等於 in-process 用乾淨金鑰算的 —— 釘住 probe 本身。

    少了這條，上面兩組測試只證明「四種髒值彼此一致」，不證明它們一致到
    「正確的那個值」上。
    """
    assert clean["mac"] == build_check_mac_value(
        VECTOR_PARAMS, CLEAN_KEY, CLEAN_IV)


def test_unset_credentials_are_empty_not_none():
    """沒設變數時仍是空字串 —— main.py:3510 的 `if not ECPAY_HASH_KEY` 靠這個。"""
    result = _import_main_with("", "")

    assert result["key"] == ""
    assert result["iv"] == ""


def test_whitespace_only_credentials_collapse_to_empty():
    """只有空白的值 strip 後是空字串，會被當成「沒設定」擋在 503，
    而不是拿一個空白金鑰去簽出一個必然失敗的簽章。"""
    result = _import_main_with("   \n", "\t ")

    assert result["key"] == ""
    assert result["iv"] == ""
