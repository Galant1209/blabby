"""active 訂閱到期掃描 job。

08-17 補了 pending 的過期掃描，但沒有任何機制把到期的 active 轉走 ——
狀態機少一條邊。權限面沒破（is_user_pro() 是 status='active' AND
expires_at > now() 雙條件），壞的是對帳可讀性，以及日後想撈「付過錢、
已經用完」的名單時撈不出來。

這支測試最重要的兩條是「active 且未到期 → 不動」與「expires_at IS NULL
→ 不動」。前者砸掉的是正在付費的人，後者砸掉的是那 10 列從未啟用的
expired 訂單 —— 它們的 expires_at 是 NULL，而 Postgres 的 NULL <= now()
是 NULL 不是 true。那個語義用推理是對的，但這裡要用測試釘住。

fake 不是「記下呼叫了哪些方法」的 spy，而是真的套用 eq / lte 語義再改
資料 —— 要驗的是這個 query 選出了哪些列，不是這段程式呼叫了哪些函式。
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

import pytest

import main


NOW = datetime(2026, 9, 3, 12, 0, 0, tzinfo=timezone.utc)


def _row(row_id: str, status: str, expires_in_days: float | None) -> dict:
    expires_at = (
        None if expires_in_days is None
        else (NOW + timedelta(days=expires_in_days)).isoformat()
    )
    return {
        "id":         row_id,
        "user_id":    f"user-{row_id}",
        "status":     status,
        "expires_at": expires_at,
        "updated_at": (NOW - timedelta(days=30)).isoformat(),
    }


class _FakeQuery:
    """實作 PostgREST 的 select / eq / lte / update 語義。

    lte 對 NULL 的處理是這個 fake 的重點：Postgres 比較 NULL 得到 NULL，
    列因此不入選。Python 的 `None <= str` 會直接 TypeError，所以不能
    照抄比較運算子，必須明寫這條規則。
    """

    def __init__(self, table: "_FakeTable", mode: str):
        self.table = table
        self.mode = mode
        self.payload: dict = {}
        self.filters: list[tuple[str, str, object]] = []

    def select(self, *_columns):
        return self

    def update(self, payload: dict):
        self.payload = payload
        return self

    def eq(self, column: str, value):
        self.filters.append(("eq", column, value))
        return self

    def lte(self, column: str, value):
        self.filters.append(("lte", column, value))
        return self

    def _matches(self, row: dict) -> bool:
        for op, column, value in self.filters:
            actual = row.get(column)
            if op == "eq":
                if actual != value:
                    return False
            elif op == "lte":
                # NULL <= anything → NULL → 不入選
                if actual is None:
                    return False
                if not (actual <= value):
                    return False
        return True

    def execute(self):
        if self.table.raises is not None:
            raise self.table.raises
        hit = [row for row in self.table.rows if self._matches(row)]
        if self.mode == "update":
            for row in hit:
                row.update(self.payload)
        return type("Response", (), {"data": [dict(row) for row in hit]})()


class _FakeTable:
    def __init__(self, rows: list[dict], raises: Exception | None = None):
        self.rows = rows
        self.raises = raises
        self.queries: list[_FakeQuery] = []

    def select(self, *columns):
        q = _FakeQuery(self, "select")
        self.queries.append(q)
        return q.select(*columns)

    def update(self, payload: dict):
        q = _FakeQuery(self, "update")
        self.queries.append(q)
        return q.update(payload)


class _FakeSupabase:
    def __init__(self, rows: list[dict], raises: Exception | None = None):
        self.table_obj = _FakeTable(rows, raises)
        self.requested: list[str] = []

    def table(self, name: str):
        self.requested.append(name)
        return self.table_obj


@pytest.fixture
def freeze_now(monkeypatch):
    """釘住 datetime.now(timezone.utc)，讓到期門檻可斷言。"""
    class _FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW if tz else NOW.replace(tzinfo=None)

    monkeypatch.setattr(main, "datetime", _FrozenDatetime)


def _sweep(rows, caplog, raises=None):
    import unittest.mock as mock

    fake = _FakeSupabase(rows, raises)
    with caplog.at_level(logging.INFO):
        with mock.patch.object(main, "supabase_admin", fake):
            main.lapse_expired_active_subscriptions()
    return fake


def _update_query(fake: _FakeSupabase) -> _FakeQuery:
    return next(q for q in fake.table_obj.queries if q.mode == "update")


# ── 選列語義 ─────────────────────────────────────────────────────────────

def test_expired_active_becomes_lapsed(freeze_now, caplog):
    rows = [_row("done", "active", -5)]
    _sweep(rows, caplog)

    assert rows[0]["status"] == "lapsed"
    assert rows[0]["updated_at"] == NOW.isoformat()
    assert "lapse sweep: scanned 1 active, lapsed 1" in caplog.text


def test_active_not_yet_expired_is_left_alone(freeze_now, caplog):
    """砸掉這條就是砸掉正在付費的人。"""
    rows = [_row("paying", "active", 13)]
    before = dict(rows[0])
    _sweep(rows, caplog)

    assert rows[0] == before
    assert "lapse sweep: scanned 1 active, lapsed 0" in caplog.text


def test_pending_is_never_touched(freeze_now, caplog):
    """這支只碰 active。pending 是另一支 job 的地盤，不得越界。"""
    rows = [
        _row("fresh_order", "pending", None),
        _row("old_order",   "pending", -99),
    ]
    before = [dict(row) for row in rows]
    _sweep(rows, caplog)

    assert rows == before
    assert "lapse sweep: scanned 0 active, lapsed 0" in caplog.text


def test_null_expiry_rows_are_not_swept(freeze_now, caplog):
    """production 現有 10 列 expired，expires_at 全是 NULL。

    Postgres 的 NULL <= now() 回傳 NULL 而不是 true，所以它們不入選。
    這條用推理是對的，但推理不是證據 —— 這裡把語義釘住。
    """
    rows = [
        _row("never_paid",   "expired", None),
        _row("null_active",  "active",  None),
    ]
    before = [dict(row) for row in rows]
    _sweep(rows, caplog)

    assert rows == before
    assert "lapse sweep: scanned 1 active, lapsed 0" in caplog.text


def test_rerunning_does_not_hit_the_same_row_twice(freeze_now, caplog):
    """冪等：轉過的列 status 已非 active，第二輪選不到。"""
    rows = [_row("done", "active", -5)]

    _sweep(rows, caplog)
    assert rows[0]["status"] == "lapsed"
    first_updated_at = rows[0]["updated_at"]

    caplog.clear()
    _sweep(rows, caplog)

    assert rows[0]["status"] == "lapsed"
    assert rows[0]["updated_at"] == first_updated_at, "第二輪不得再寫一次"
    assert "lapse sweep: scanned 0 active, lapsed 0" in caplog.text


def test_only_the_expired_active_rows_move_in_a_mixed_table(freeze_now, caplog):
    rows = [
        _row("done_1",      "active",    -1),
        _row("done_2",      "active",   -40),
        _row("paying",      "active",    13),
        _row("stale_order", "pending", None),
        _row("never_paid",  "expired", None),
        _row("already",     "lapsed",   -60),
        _row("cancelled",   "cancelled", -3),
    ]
    _sweep(rows, caplog)

    assert {row["id"]: row["status"] for row in rows} == {
        "done_1":      "lapsed",
        "done_2":      "lapsed",
        "paying":      "active",
        "stale_order": "pending",
        "never_paid":  "expired",
        "already":     "lapsed",
        "cancelled":   "cancelled",
    }
    assert "lapse sweep: scanned 3 active, lapsed 2" in caplog.text


def test_exactly_at_the_boundary_is_swept(freeze_now, caplog):
    """門檻是 <=：expires_at 剛好等於 now 代表權限已經沒了。

    is_user_pro() 用的是 expires_at > now()，所以 expires_at == now 的那
    一瞬間就已經不是 Pro 了。這裡用 <= 才對得上，用 < 會留下一列永遠
    差一秒的孤兒。
    """
    rows = [_row("boundary", "active", 0)]
    _sweep(rows, caplog)

    assert rows[0]["status"] == "lapsed"


# ── WHERE 與 payload 的形狀 ──────────────────────────────────────────────

def test_both_conditions_are_pinned_in_the_where_clause(freeze_now, caplog):
    """status='active' 必須明寫，即使 expires_at 條件看起來已經夠了。

    與另一支 job 對稱的刻意冗餘：讓「碰到 pending」需要兩個條件同時失效。
    """
    fake = _sweep([_row("done", "active", -5)], caplog)
    filters = _update_query(fake).filters

    assert ("eq", "status", "active") in filters
    assert any(op == "lte" and column == "expires_at" for op, column, _ in filters)
    assert fake.requested == ["subscriptions", "subscriptions"]


def test_cutoff_is_now_not_a_hardcoded_date(freeze_now, caplog):
    fake = _sweep([_row("done", "active", -5)], caplog)
    cutoff = next(value for op, column, value in _update_query(fake).filters
                  if op == "lte" and column == "expires_at")

    assert cutoff == NOW.isoformat()


def test_payload_only_sets_status_and_updated_at(freeze_now, caplog):
    """不順手改別的欄位 —— expires_at 尤其不能碰，那是召回名單要用的。"""
    fake = _sweep([_row("done", "active", -5)], caplog)
    payload = _update_query(fake).payload

    assert set(payload) == {"status", "updated_at"}
    assert payload["status"] == "lapsed"
    assert payload["updated_at"] == NOW.isoformat()


def test_lapsed_is_not_expired(freeze_now, caplog):
    """兩個狀態的意思不同，不得合併。"""
    fake = _sweep([_row("done", "active", -5)], caplog)

    assert _update_query(fake).payload["status"] != "expired"


# ── log ─────────────────────────────────────────────────────────────────

def test_each_moved_row_is_logged_with_user_and_expiry(freeze_now, caplog):
    rows = [_row("done_1", "active", -1), _row("done_2", "active", -40)]
    _sweep(rows, caplog)

    for row_id in ("done_1", "done_2"):
        assert f"id={row_id}" in caplog.text
        assert f"user_id=user-{row_id}" in caplog.text
    assert (NOW - timedelta(days=40)).isoformat() in caplog.text


def test_zero_matches_still_logs(freeze_now, caplog):
    _sweep([], caplog)

    assert "[BILLING] lapse sweep: scanned 0 active, lapsed 0" in caplog.text


# ── 失敗模式 ─────────────────────────────────────────────────────────────

def test_database_error_does_not_crash_the_scheduler(freeze_now, caplog):
    _sweep([_row("done", "active", -5)], caplog,
           raises=RuntimeError("connection reset"))

    assert "RuntimeError" in caplog.text, "必須留下 traceback"
    assert "Traceback" in caplog.text
    # 失敗時不得謊報筆數
    assert "lapse sweep:" not in caplog.text


def test_missing_database_is_a_warning_not_a_crash(caplog):
    import unittest.mock as mock

    with caplog.at_level(logging.WARNING), \
         mock.patch.object(main, "supabase_admin", None):
        main.lapse_expired_active_subscriptions()

    assert "database not configured" in caplog.text


# ── scheduler 註冊 ───────────────────────────────────────────────────────

def test_job_is_registered_offset_from_the_pending_sweep():
    """同樣每 6 小時，但錯開 30 分鐘 —— 兩支都改 subscriptions。"""
    import inspect

    source = inspect.getsource(main.startup_event)

    assert "lapse_expired_active_subscriptions" in source
    assert 'CronTrigger(hour="*/6", minute=30, timezone="UTC")' in source
    # 另一支維持 minute=0，沒有被順手改掉
    assert 'CronTrigger(hour="*/6", minute=0, timezone="UTC")' in source


def test_job_is_not_gated_by_the_pregeneration_flag():
    """狀態機的正確性不該跟著內容生成的成本開關一起被關掉。"""
    import inspect

    source = inspect.getsource(main.startup_event)
    before_flag = source.split("if PREGEN_ENABLED:", 1)[0]

    assert "lapse_expired_active_subscriptions" in before_flag


def test_the_pending_sweep_is_untouched():
    """新 job 是獨立的一支，另一支的 WHERE 一行未改。"""
    import inspect

    source = inspect.getsource(main.expire_stale_pending_subscriptions)

    assert '.eq("status", "pending")' in source
    assert "lapsed" not in source
    assert '"status": "expired"' in source
