"""writing.html 的 ?submission= 還原路徑（Node DOM harness 包裝）。

形狀照抄 test_frontend_billing_contracts.py 的
test_frontend_billing_behavior_harness：shell out 到 node、把 stdout/stderr
轉出來、returncode 非 0 就失敗。沒有發明新做法。

Node 不存在時的行為也與既有的一致 —— subprocess.run 會拋 FileNotFoundError,
測試變紅而不是 skip。那不是這裡新增的風險，是既有包裝已有的行為，兩支一起
改才有意義。

分數一致性那條的 payload 由這裡用真的 build_writing_feedback_view() 產生後
交給 harness。在 .mjs 裡自己造資料等於驗那支檔案的副本，不是驗後端真正回什麼。
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

import main


APP_DIR = Path(__file__).parents[2] / "frontend" / "app"
HARNESS = Path(__file__).with_name("frontend_writing_restore_behavior.mjs")
NODE = "node"


def _grader_row() -> dict:
    """交卷當下的一列：band 來自 LLM 解析，是 Python float。"""
    return {
        "id":           "abc-123",
        "word_count":   155,
        "essay_text":   "The chart shows a steady rise.",
        "priority_fix": "Compare the two lines, do not just list them.",
        "band_overall": 4.0,
        "band_ta": 4.5, "band_cc": 4.0, "band_lr": 4.0, "band_gra": 4.0,
        "feedback_ta": "fb ta", "feedback_cc": "fb cc",
        "feedback_lr": "fb lr", "feedback_gra": "fb gra",
        "fix_ta": "fix ta", "fix_cc": "fix cc",
        "fix_lr": "fix lr", "fix_gra": "fix gra",
    }


def _db_row() -> dict:
    """同一列經 PostgREST 回來：numeric 序列化成字串以保留精度。"""
    row = _grader_row()
    for column in ("band_overall", "band_ta", "band_cc", "band_lr", "band_gra"):
        row[column] = str(row[column])
    return row


def test_the_two_payloads_are_identical_before_node_sees_them():
    """先在 Python 這一側證明，harness 才有東西可比。"""
    post = main.build_writing_feedback_view(_grader_row())
    get = main.build_writing_feedback_view(_db_row())

    assert post == get
    assert post["band_overall"] == 4.0
    assert post["criteria"]["task_achievement"]["band"] == 4.5


def test_frontend_writing_restore_behavior_harness(tmp_path):
    if not HARNESS.exists():
        pytest.fail(f"missing harness: {HARNESS}")

    payloads = tmp_path / "payloads.json"
    payloads.write_text(json.dumps({
        "post": main.build_writing_feedback_view(_grader_row()),
        "get":  main.build_writing_feedback_view(_db_row()),
    }), encoding="utf-8")

    result = subprocess.run(
        [NODE, str(HARNESS), str(payloads)],
        cwd=str(APP_DIR.parent.parent),
        capture_output=True,
        text=True,
        timeout=60,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    assert result.returncode == 0, (
        f"writing restore harness failed (exit {result.returncode}):\n"
        f"{result.stdout}\n{result.stderr}"
    )
    # harness 自己會斷言，但 skip 掉分數一致性時 returncode 仍是 0 ——
    # 這裡確認那一條真的跑了。
    assert "分數一致性" in result.stdout
    assert "略過" not in result.stdout
