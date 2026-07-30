"""Subprocess probe for test_ecpay_config_blast_radius.py.

Runs in a FRESH interpreter with a deliberately broken ECPAY_ENV, because the
property under test — "importing main does not raise" — cannot be observed in a
process where main is already imported.

Boots the real FastAPI startup handler (scheduler stubbed), captures CRITICAL
log records, then probes one endpoint per product area plus both payment
endpoints. Prints a single JSON object on stdout.

Not a test module itself: the leading underscore keeps pytest from collecting it.
"""

import json
import logging
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir)))

result = {}

# ── the property this whole file exists for ──────────────────────────────
try:
    import main
    result["import_ok"] = True
except BaseException as exc:                      # noqa: BLE001 - report anything
    print(json.dumps({"import_ok": False, "import_error": repr(exc)}))
    sys.exit(0)

from unittest.mock import AsyncMock, MagicMock, patch  # noqa: E402
from fastapi.testclient import TestClient             # noqa: E402

USER_ID = "user-blast-radius"


class _Resp:
    def __init__(self, data):
        self.data = data


class _FakeTable:
    """Accepts any PostgREST chain and answers with an empty result set."""

    def __getattr__(self, _name):
        return lambda *a, **k: self

    def execute(self):
        return _Resp([])


class _FakeSupabase:
    def table(self, _name):
        return _FakeTable()

    def rpc(self, _name, _params=None):
        return MagicMock(execute=lambda: _Resp(False))


# ── capture CRITICAL records emitted during startup ──────────────────────
class _Collector(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.CRITICAL)
        self.records = []

    def emit(self, record):
        self.records.append(record.getMessage())


collector = _Collector()
main.logger.addHandler(collector)   # not the root logger too: main's propagates

fake = _FakeSupabase()

with patch.object(main.limiter, "enabled", False), \
     patch.object(main, "verify_token", return_value=USER_ID), \
     patch.object(main, "supabase_admin", fake), \
     patch.object(main._scheduler, "start", MagicMock()), \
     patch.object(main._scheduler, "add_job", MagicMock()), \
     patch.object(main, "_reading_daily_count", MagicMock(return_value=0)), \
     patch.object(main, "get_user_pro_status", MagicMock(return_value=False)), \
     patch.object(main, "is_user_pro", AsyncMock(return_value=False)):

    # Context manager => the real startup_event runs, including load_config().
    with TestClient(main.app) as client:
        result["critical_logs"] = list(collector.records)

        health = client.get("/health")
        result["health_status"] = health.status_code
        result["health_body"] = health.json()

        # Payment endpoints must be the only casualties.
        result["create_order_status"] = client.post(
            "/api/payment/create-order", headers={"Authorization": "Bearer t"}
        ).status_code
        result["callback_status"] = client.post(
            "/api/payment/callback", data={"MerchantTradeNo": "X"}
        ).status_code

        # One endpoint per product area. All must be untouched.
        result["part1_status"] = client.get("/api/questions/bank").status_code
        result["part2_status"] = client.get("/part2/topics").status_code
        result["reading_status"] = client.get(
            "/reading/quota", headers={"Authorization": "Bearer t"}
        ).status_code
        result["writing_status"] = client.get(
            "/api/writing/history", headers={"Authorization": "Bearer t"}
        ).status_code

print(json.dumps(result))
