"""pending 訂單過期掃描 job。

建單寫入 status='pending'，callback 成功才轉 active。放棄付款或銀行拒絕的
訂單永遠停在 pending —— 對權限沒有影響（is_user_pro() 只認 active 且未過
期），但對帳時是雜訊。

這支測試最重要的一條是「active 且超過 24 小時 → 不動」。production 現在
就有 4 列 active 的建立時間超過 24 小時，那是真的付了 NT$199 的人；WHERE
少一個條件，砸掉的是他們。

fake 不是「記下呼叫了哪些方法」的 spy，而是真的套用 eq / lt 語義再改資料 ——
要驗的是這個 query 選出了哪些列，不是這段程式呼叫了哪些函式。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest

import main


NOW = datetime(2026, 8, 17, 12, 0, 0, tzinfo=timezone.utc)


def _row(row_id: str, status: str, age_hours: float) -> dict:
    return {
        "id":         row_id,
        "status":     status,
        "created_at": (NOW - timedelta(hours=age_hours)).isoformat(),
        "updated_at": (NOW - timedelta(hours=age_hours)).isoformat(),
    }


class _FakeTable:
    """實作 PostgREST 的 eq / lt / update 語義，不是記錄呼叫的 spy。"""

    def __init__(self, rows: list[dict], raises: Exception | None = None):
        self.rows = rows
        self.raises = raises
        self._payload: dict = {}
        self._filters: list[tuple[str, str, str]] = []

    def update(self, payload: dict):
        self._payload = payload
        return self

    def eq(self, column: str, value):
        self._filters.append(("eq", column, value))
        return self

    def lt(self, column: str, value):
        self._filters.append(("lt", column, value))
        return self

    def _matches(self, row: dict) -> bool:
        for op, column, value in self._filters:
            actual = row.get(column)
            if op == "eq" and actual != value:
                return False
            if op == "lt" and not (actual < value):
                return False
        return True

    def execute(self):
        if self.raises is not None:
            raise self.raises
        hit = [row for row in self.rows if self._matches(row)]
        for row in hit:
            row.update(self._payload)
        return type("Response", (), {"data": [dict(row) for row in hit]})()


class _FakeSupabase:
    def __init__(self, rows: list[dict], raises: Exception | None = None):
        self.table_obj = _FakeTable(rows, raises)
        self.requested: list[str] = []

    def table(self, name: str):
        self.requested.append(name)
        return self.table_obj


@pytest.fixture
def freeze_now(monkeypatch):
    """釘住 datetime.now(timezone.utc)，讓 24 小時的門檻可斷言。"""
    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW if tz else NOW.replace(tzinfo=None)

    monkeypatch.setattr(main, "datetime", _FrozenDatetime)


def _sweep(rows, caplog, raises=None):
    fake = _FakeSupabase(rows, raises)
    with caplog.at_level(logging.INFO):
        import unittest.mock as mock
        with mock.patch.object(main, "supabase_admin", fake):
            main.expire_stale_pending_subscriptions()
    return fake


# ── 選列語義 ─────────────────────────────────────────────────────────────

def test_stale_pending_is_expired(freeze_now, caplog):
    rows = [_row("stale", "pending", 25)]
    _sweep(rows, caplog)

    assert rows[0]["status"] == "expired"
    assert "[BILLING] expired 1 stale pending subscriptions" in caplog.text


def test_fresh_pending_is_left_alone(freeze_now, caplog):
    rows = [_row("fresh", "pending", 23)]
    _sweep(rows, caplog)

    assert rows[0]["status"] == "pending"
    assert "expired 0 stale pending subscriptions" in caplog.text


def test_active_older_than_the_ttl_is_never_touched(freeze_now, caplog):
    """這條最重要。production 現有 4 列 active 的建立時間超過 24 小時。"""
    rows = [_row("paid", "active", 720)]      # 30 天前付款、仍在權限內
    _sweep(rows, caplog)

    assert rows[0]["status"] == "active"
    assert rows[0]["updated_at"] == _row("paid", "active", 720)["updated_at"]
    assert "expired 0 stale pending subscriptions" in caplog.text


def test_only_the_stale_pending_rows_move_in_a_mixed_table(freeze_now, caplog):
    rows = [
        _row("stale_1",  "pending",  25),
        _row("stale_2",  "pending",  99),
        _row("fresh",    "pending",   1),
        _row("paid",     "active",  720),
        _row("expired",  "expired", 999),
    ]
    _sweep(rows, caplog)

    moved = {row["id"]: row["status"] for row in rows}
    assert moved == {
        "stale_1": "expired",
        "stale_2": "expired",
        "fresh":   "pending",
        "paid":    "active",
        "expired": "expired",
    }
    assert "expired 2 stale pending subscriptions" in caplog.text


def test_exactly_at_the_boundary_is_not_swept(freeze_now, caplog):
    """門檻是嚴格小於：剛好滿 24 小時的那一筆留著，下一輪再說。"""
    rows = [_row("boundary", "pending", main.PENDING_ORDER_TTL_HOURS)]
    _sweep(rows, caplog)

    assert rows[0]["status"] == "pending"


# ── WHERE 的形狀 ─────────────────────────────────────────────────────────

def test_status_pending_is_pinned_in_the_where_clause(freeze_now, caplog):
    """即使 created_at 條件已足夠，status='pending' 也必須明寫。

    這是刻意的冗餘 —— 讓「碰到 active」需要兩個條件同時失效，不是一個。
    """
    fake = _sweep([_row("stale", "pending", 25)], caplog)
    filters = fake.table_obj._filters

    assert ("eq", "status", "pending") in filters
    assert any(op == "lt" and column == "created_at" for op, column, _ in filters)
    assert fake.requested == ["subscriptions"]


def test_cutoff_is_exactly_the_module_constant(freeze_now, caplog):
    """24 小時來自模組層常數，不是散在 SQL 字串裡的字面值。"""
    fake = _sweep([_row("stale", "pending", 25)], caplog)
    cutoff = next(value for op, column, value in fake.table_obj._filters
                  if op == "lt" and column == "created_at")

    expected = NOW - timedelta(hours=main.PENDING_ORDER_TTL_HOURS)
    assert cutoff == expected.isoformat()


def test_payload_only_sets_status_and_updated_at(freeze_now, caplog):
    """不順手改別的欄位 —— expires_at 尤其不能碰。"""
    fake = _sweep([_row("stale", "pending", 25)], caplog)

    assert set(fake.table_obj._payload) == {"status", "updated_at"}
    assert fake.table_obj._payload["status"] == "expired"
    assert fake.table_obj._payload["updated_at"] == NOW.isoformat()


# ── 失敗模式 ─────────────────────────────────────────────────────────────

def test_zero_matches_still_logs(freeze_now, caplog):
    _sweep([], caplog)

    assert "[BILLING] expired 0 stale pending subscriptions" in caplog.text


def test_database_error_does_not_crash_the_scheduler(freeze_now, caplog):
    with caplog.at_level(logging.INFO):
        _sweep([_row("stale", "pending", 25)], caplog,
               raises=RuntimeError("connection reset"))

    assert "[BILLING] expire_stale_pending failed" in caplog.text
    assert "RuntimeError" in caplog.text, "必須留下 traceback"
    assert "Traceback" in caplog.text
    # 失敗時不得謊報筆數
    assert "stale pending subscriptions" not in caplog.text


def test_missing_database_is_a_warning_not_a_crash(caplog):
    import unittest.mock as mock

    with caplog.at_level(logging.WARNING), \
         mock.patch.object(main, "supabase_admin", None):
        main.expire_stale_pending_subscriptions()

    assert "database not configured" in caplog.text


# ── scheduler 註冊 ───────────────────────────────────────────────────────

def test_job_is_registered_on_the_shared_scheduler():
    """與 pregeneration jobs 並列在同一個 scheduler 上，每 6 小時。"""
    import inspect

    source = inspect.getsource(main.startup_event)

    assert "expire_stale_pending_subscriptions" in source
    assert source.count('CronTrigger(hour="*/6", minute=0, timezone="UTC")') == 3
