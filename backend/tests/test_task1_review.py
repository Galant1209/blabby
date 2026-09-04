"""Hermetic contracts for the Task 1 chart human-review loop."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from fastapi import HTTPException
from fastapi.testclient import TestClient

import main


QUESTION_ID = "491a28c2-4bfa-47f3-8e3b-484695df73d2"
MULTI_ID = "ab70b2fd-b96b-435b-81ca-ee7864be2c10"


class _Query:
    def __init__(self, store, table):
        self.store = store
        self.table = table
        self.filters = []
        self.payload = None
        self.limit_value = None

    def select(self, *_args, **_kwargs):
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def order(self, key, desc=False):
        self.store[self.table].sort(key=lambda row: str(row.get(key) or ""), reverse=desc)
        return self

    def limit(self, value):
        self.limit_value = value
        return self

    def update(self, payload):
        self.payload = dict(payload)
        return self

    def execute(self):
        rows = self.store[self.table]
        matches = [
            row for row in rows
            if all(row.get(key) == value for key, value in self.filters)
        ]
        if self.payload is not None:
            for row in matches:
                row.update(self.payload)
            return SimpleNamespace(data=[dict(row) for row in matches])
        if self.limit_value is not None:
            matches = matches[: self.limit_value]
        return SimpleNamespace(data=[dict(row) for row in matches])


class _Supabase:
    def __init__(self, rows):
        self.store = {"writing_questions": rows}

    def table(self, table):
        return _Query(self.store, table)


def _rows():
    return [
        {
            "id": QUESTION_ID,
            "task_type": "task1",
            "task1_subtype": "pie_chart",
            "prompt": "The pie chart below shows household expenditure.",
            "chart_description": "Category | Value (%)\nHousing | 50\nFood | 30\nTransport | 20",
            "chart_svg": None,
            "is_pregenerated": False,
            "used_count": 0,
            "created_at": "2026-07-14T00:00:00Z",
            "review_status": "pending",
            "review_issue": None,
            "review_note": None,
            "reviewed_at": None,
            "reviewed_by": None,
        },
        {
            "id": MULTI_ID,
            "task_type": "task1",
            "task1_subtype": "pie_chart",
            "prompt": "The pie charts below show energy sources.",
            "chart_description": "Category | 2010 | 2023\nCoal | 28 | 12\nRenewable | 15 | 42",
            "chart_svg": None,
            "is_pregenerated": False,
            "used_count": 2,
            "created_at": "2026-07-15T00:00:00Z",
            "review_status": "pending",
            "review_issue": None,
            "review_note": None,
            "reviewed_at": None,
            "reviewed_by": None,
        },
        {
            "id": "00000000-0000-0000-0000-000000000001",
            "task_type": "task2",
            "task1_subtype": None,
            "prompt": "Task 2 prompt",
            "chart_description": None,
            "chart_svg": None,
            "is_pregenerated": True,
            "used_count": 0,
            "created_at": "2026-07-16T00:00:00Z",
            "review_status": "pending",
            "review_issue": None,
            "review_note": None,
            "reviewed_at": None,
            "reviewed_by": None,
        },
    ]


def _client():
    return TestClient(main.app)


def _admin_request(method, path, **kwargs):
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_admin", MagicMock(return_value="admin-user-id")):
        return getattr(_client(), method)(path, headers={"Authorization": "Bearer test"}, **kwargs)


def test_review_get_requires_admin_authentication():
    with patch.object(main.limiter, "enabled", False), \
         patch.object(main, "verify_admin", MagicMock(side_effect=HTTPException(status_code=403, detail="Admin access required"))):
        response = _client().get("/admin/writing/task1-review")
    assert response.status_code == 403


def test_review_get_returns_task1_only_and_defaults_missing_review_status():
    fake = _Supabase(_rows())
    with patch.object(main, "supabase_admin", fake):
        response = _admin_request("get", "/admin/writing/task1-review")
    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 2
    assert {row["task1_subtype"] for row in body["questions"]} == {"pie_chart"}


def test_review_get_reuses_pie_parser_and_marks_multi_period_unsupported():
    fake = _Supabase(_rows())
    with patch.object(main, "supabase_admin", fake):
        body = _admin_request("get", "/admin/writing/task1-review").json()
    by_id = {row["question_id"]: row for row in body["questions"]}
    assert by_id[QUESTION_ID]["preview_kind"] == "pie"
    assert by_id[QUESTION_ID]["chart_data"]["values"] == [50.0, 30.0, 20.0]
    assert by_id[MULTI_ID]["preview_kind"] == "renderer_unsupported"
    assert by_id[MULTI_ID]["chart_data"] is None


def test_review_filter_rejects_unknown_status_before_db_query():
    fake = _Supabase(_rows())
    with patch.object(main, "supabase_admin", fake):
        response = _admin_request("get", "/admin/writing/task1-review?review_status=unknown")
    assert response.status_code == 422


def test_review_patch_rejects_non_review_fields():
    fake = _Supabase(_rows())
    with patch.object(main, "supabase_admin", fake):
        response = _admin_request(
            "patch", f"/admin/writing/task1-review/{QUESTION_ID}",
            json={"review_status": "approved", "is_pregenerated": True},
        )
    assert response.status_code == 422
    assert fake.store["writing_questions"][0]["review_status"] == "pending"


def test_review_patch_rejects_unknown_issue():
    fake = _Supabase(_rows())
    with patch.object(main, "supabase_admin", fake):
        response = _admin_request(
            "patch", f"/admin/writing/task1-review/{QUESTION_ID}",
            json={"review_status": "needs_fix", "review_issue": "magic"},
        )
    assert response.status_code == 422


def test_review_patch_sets_server_audit_fields_and_preserves_serving_flag():
    fake = _Supabase(_rows())
    with patch.object(main, "supabase_admin", fake), \
         patch.object(main, "datetime") as clock:
        clock.now.return_value.isoformat.return_value = "2026-09-04T08:00:00+00:00"
        response = _admin_request(
            "patch", f"/admin/writing/task1-review/{QUESTION_ID}",
            json={
                "review_status": "needs_fix",
                "review_issue": "renderer_unsupported",
                "review_note": "Stored shape needs an engineering decision.",
            },
        )
    assert response.status_code == 200
    row = fake.store["writing_questions"][0]
    assert row["review_status"] == "needs_fix"
    assert row["review_issue"] == "renderer_unsupported"
    assert row["reviewed_by"] == "admin-user-id"
    assert row["reviewed_at"] == "2026-09-04T08:00:00+00:00"
    assert row["is_pregenerated"] is False


def test_review_patch_cannot_spoof_reviewed_by_or_reviewed_at():
    fake = _Supabase(_rows())
    with patch.object(main, "supabase_admin", fake):
        response = _admin_request(
            "patch", f"/admin/writing/task1-review/{QUESTION_ID}",
            json={
                "review_status": "approved",
                "reviewed_by": "spoofed",
                "reviewed_at": "2000-01-01T00:00:00Z",
            },
        )
    assert response.status_code == 422


def test_approved_review_does_not_reactivate_retired_question():
    fake = _Supabase(_rows())
    with patch.object(main, "supabase_admin", fake):
        response = _admin_request(
            "patch", f"/admin/writing/task1-review/{MULTI_ID}",
            json={"review_status": "approved", "review_note": "Engineering reactivation candidate."},
        )
    assert response.status_code == 200
    row = next(item for item in fake.store["writing_questions"] if item["id"] == MULTI_ID)
    assert row["review_status"] == "approved"
    assert row["is_pregenerated"] is False


def test_review_patch_rejects_invalid_question_id():
    fake = _Supabase(_rows())
    with patch.object(main, "supabase_admin", fake):
        response = _admin_request("patch", "/admin/writing/task1-review/not-a-uuid", json={"review_status": "approved"})
    assert response.status_code == 400
