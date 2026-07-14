"""Staging-only anon/A/B/service-role permission matrix for the P1 migration.

Required opt-in environment variables:
  SUPABASE_SECURITY_TEST_URL
  SUPABASE_SECURITY_TEST_ANON_KEY
  SUPABASE_SECURITY_TEST_SERVICE_KEY
  SUPABASE_SECURITY_TEST_USER_A_TOKEN
  SUPABASE_SECURITY_TEST_USER_B_TOKEN

Never point this test at production. It creates uniquely named diagnosis cache
rows and subscription rows inside the configured staging project, then removes
them in a finally block through service-role access.
"""

from __future__ import annotations

import os
import uuid

import pytest

supabase = pytest.importorskip("supabase")

NAMES = (
    "SUPABASE_SECURITY_TEST_URL",
    "SUPABASE_SECURITY_TEST_ANON_KEY",
    "SUPABASE_SECURITY_TEST_SERVICE_KEY",
    "SUPABASE_SECURITY_TEST_USER_A_TOKEN",
    "SUPABASE_SECURITY_TEST_USER_B_TOKEN",
)
CFG = {name: os.getenv(name, "").strip() for name in NAMES}
pytestmark = pytest.mark.skipif(
    not all(CFG.values()),
    reason="staging Supabase security-test credentials are not configured",
)


def _client(key: str, token: str | None = None):
    client = supabase.create_client(CFG["SUPABASE_SECURITY_TEST_URL"], key)
    if token:
        client.postgrest.auth(token)
    return client


def _user_id(client, token: str) -> str:
    response = client.auth.get_user(token)
    return str(response.user.id)


def _select_denied(client, table: str, columns: str) -> bool:
    try:
        client.table(table).select(columns).limit(1).execute()
    except Exception:
        return True
    return False


def test_anon_user_a_user_b_and_service_role_matrix():
    anon = _client(CFG["SUPABASE_SECURITY_TEST_ANON_KEY"])
    service = _client(CFG["SUPABASE_SECURITY_TEST_SERVICE_KEY"])
    user_a = _client(CFG["SUPABASE_SECURITY_TEST_ANON_KEY"], CFG["SUPABASE_SECURITY_TEST_USER_A_TOKEN"])
    user_b = _client(CFG["SUPABASE_SECURITY_TEST_ANON_KEY"], CFG["SUPABASE_SECURITY_TEST_USER_B_TOKEN"])
    user_a_id = _user_id(user_a, CFG["SUPABASE_SECURITY_TEST_USER_A_TOKEN"])
    user_b_id = _user_id(user_b, CFG["SUPABASE_SECURITY_TEST_USER_B_TOKEN"])
    marker = uuid.uuid4().hex
    subscription_ids: list[str] = []

    try:
        service.table("diagnosis_cache").upsert({
            "user_id": user_a_id, "content": f"security-a-{marker}", "practice_count": 0,
        }).execute()
        service.table("diagnosis_cache").upsert({
            "user_id": user_b_id, "content": f"security-b-{marker}", "practice_count": 0,
        }).execute()
        for user_id in (user_a_id, user_b_id):
            row = service.table("subscriptions").insert({
                "user_id": user_id, "order_id": f"security-{marker}-{user_id}",
                "plan": "monthly", "status": "pending",
            }).execute().data[0]
            subscription_ids.append(row["id"])

        assert _select_denied(anon, "diagnosis_cache", "*")
        assert _select_denied(anon, "subscriptions", "*")
        assert _select_denied(anon, "reading_questions", "correct_answer")

        assert {r["user_id"] for r in user_a.table("diagnosis_cache").select("user_id").execute().data} == {user_a_id}
        assert {r["user_id"] for r in user_b.table("diagnosis_cache").select("user_id").execute().data} == {user_b_id}
        assert _select_denied(user_a, "subscriptions", "user_id,amount,order_id")
        assert _select_denied(user_b, "reading_questions", "correct_answer,explanation,evidence_quote")

        safe_questions = user_a.table("reading_questions").select(
            "id,passage_id,question_type,question_text,options,order_idx,created_at"
        ).limit(1).execute()
        assert safe_questions.data is not None

        assert service.table("diagnosis_cache").select("user_id,content").in_("user_id", [user_a_id, user_b_id]).execute().data
        assert service.table("reading_questions").select("correct_answer,explanation,evidence_quote").limit(1).execute().data is not None
    finally:
        service.table("diagnosis_cache").delete().in_("user_id", [user_a_id, user_b_id]).execute()
        if subscription_ids:
            service.table("subscriptions").delete().in_("id", subscription_ids).execute()
