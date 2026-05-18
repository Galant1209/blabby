"""
End-to-end smoke test for the Reading module.

This is an *integration* test — it hits a real Blabby backend, which in turn
hits Supabase and Anthropic. It is intentionally gated on environment
variables so that:
  - CI without credentials skips the entire module cleanly
  - A developer with credentials runs it explicitly

Required env vars (all three must be set, else every test in this module
is skipped):

    READING_E2E_BASE_URL        e.g. http://localhost:10000
                                or   https://blabby-backend.onrender.com

    READING_E2E_USER_TOKEN      Supabase access_token (JWT) of a dedicated
                                FREE-tier test user (e.g. e2e-free@blabby.test).
                                Never granted Pro. Get the token by signing
                                in via the frontend and copying it from
                                supabase.auth.getSession().

    READING_E2E_PRO_USER_TOKEN  (Optional) Supabase access_token of a
                                separate, dedicated PRO-tier test user
                                (e.g. e2e-pro@blabby.test) that has been
                                permanently granted is_pro_grant via the
                                admin tool. The Pro-bypass test uses this
                                token; it never calls admin endpoints, never
                                mutates is_pro_grant, and never affects the
                                free user. Test is skipped if unset.

                                The free and pro tokens MUST belong to
                                different users — sharing them would cause
                                the quota-blocked test to flap.

Run only this file:
    cd blabby/backend
    READING_E2E_BASE_URL=http://localhost:10000 \\
    READING_E2E_USER_TOKEN=... \\
    pytest tests/test_reading_e2e.py -v

Cost note: each full run consumes Anthropic tokens for one passage + one
question pack + a couple of vocab lookups. Estimate: a few cents per run.
"""

import os
import re
import uuid

import pytest
import requests


BASE_URL = os.environ.get("READING_E2E_BASE_URL", "").rstrip("/")
USER_TOKEN = os.environ.get("READING_E2E_USER_TOKEN", "").strip()
PRO_USER_TOKEN = os.environ.get("READING_E2E_PRO_USER_TOKEN", "").strip()

SKIP_REASON = (
    "READING_E2E_BASE_URL and READING_E2E_USER_TOKEN must be set to run "
    "reading e2e tests against a live backend"
)

pytestmark = pytest.mark.skipif(
    not (BASE_URL and USER_TOKEN),
    reason=SKIP_REASON,
)


# ────────────────────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────────────────────
def _auth_headers(token: str = "") -> dict:
    return {"Authorization": f"Bearer {token or USER_TOKEN}"}


def _is_uuid(s) -> bool:
    if not isinstance(s, str):
        return False
    try:
        uuid.UUID(s)
        return True
    except ValueError:
        return False


# ────────────────────────────────────────────────────────────────────────────
#  Module-scoped fixture: one passage + attempt shared across tests so we
#  only spend Anthropic tokens once per pytest run.
# ────────────────────────────────────────────────────────────────────────────
@pytest.fixture(scope="module")
def generated_passage():
    res = requests.post(
        f"{BASE_URL}/reading/passage/generate",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        json={},
        timeout=120,
    )
    assert res.status_code == 200, f"generate {res.status_code}: {res.text[:300]}"
    body = res.json()
    # Shape assertions belong in the dedicated test below; here we only
    # surface the most catastrophic failure (broken contract) early.
    assert "passage_id" in body
    assert "questions" in body
    return body


@pytest.fixture(scope="module")
def started_attempt(generated_passage):
    res = requests.post(
        f"{BASE_URL}/reading/attempt/start",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        json={"passage_id": generated_passage["passage_id"]},
        timeout=30,
    )
    assert res.status_code == 200, f"start {res.status_code}: {res.text[:300]}"
    body = res.json()
    assert _is_uuid(body.get("attempt_id"))
    return body


# ────────────────────────────────────────────────────────────────────────────
#  Tests
# ────────────────────────────────────────────────────────────────────────────
def test_generate_returns_well_shaped_passage(generated_passage):
    body = generated_passage
    assert _is_uuid(body["passage_id"])
    assert isinstance(body.get("title"), str) and body["title"].strip()
    assert isinstance(body.get("body"), str) and len(body["body"]) > 200

    questions = body.get("questions")
    assert isinstance(questions, list)
    assert len(questions) == 9

    order_seen = set()
    for q in questions:
        assert _is_uuid(q.get("id"))
        assert q.get("question_type") in {"mcq", "tfng", "heading"}
        assert isinstance(q.get("question_text"), str)
        assert isinstance(q.get("order_idx"), int) and 1 <= q["order_idx"] <= 9
        order_seen.add(q["order_idx"])
        # CRITICAL: correct_answer/explanation/evidence_quote must NOT leak
        # before the user submits. If any of these are present, the user
        # could cheat by reading the API response.
        for forbidden in ("correct_answer", "explanation", "evidence_quote"):
            assert forbidden not in q, f"{forbidden} leaked in pre-submit payload"
    assert order_seen == set(range(1, 10))


def test_start_attempt_returns_uuid(started_attempt):
    assert _is_uuid(started_attempt["attempt_id"])
    assert isinstance(started_attempt.get("started_at"), str)


def test_start_attempt_is_idempotent(generated_passage, started_attempt):
    """A second call with the same passage_id returns the same attempt id."""
    res = requests.post(
        f"{BASE_URL}/reading/attempt/start",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        json={"passage_id": generated_passage["passage_id"]},
        timeout=30,
    )
    assert res.status_code == 200
    assert res.json()["attempt_id"] == started_attempt["attempt_id"]


def test_submit_returns_score_and_reveals_explanations(
    generated_passage, started_attempt,
):
    # Deliberately mix answers: half garbage, half "True" for variety. We
    # don't care which ones are right; we only care that:
    #   - the score and total are returned
    #   - the reveal (correct_answer + explanation + evidence_quote) is present
    answers = []
    for q in generated_passage["questions"]:
        qtype = q["question_type"]
        if qtype == "mcq":
            user_answer = "A"
        elif qtype == "tfng":
            user_answer = "True"
        else:
            opts = q.get("options") or []
            user_answer = opts[0] if opts else ""
        answers.append({"question_id": q["id"], "user_answer": user_answer})

    res = requests.post(
        f"{BASE_URL}/reading/attempt/submit",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        json={"attempt_id": started_attempt["attempt_id"], "answers": answers},
        timeout=30,
    )
    assert res.status_code == 200, f"submit {res.status_code}: {res.text[:300]}"
    body = res.json()
    assert isinstance(body.get("score"), int)
    assert body.get("total") == 9
    assert isinstance(body.get("band_estimate"), (int, float))
    results = body.get("results")
    assert isinstance(results, list) and len(results) == 9
    for r in results:
        assert _is_uuid(r.get("question_id"))
        assert "correct_answer" in r and r["correct_answer"]
        assert "explanation" in r and r["explanation"]
        # evidence_quote can be empty for some question types, but the key
        # itself must be present so the frontend can render it consistently.
        assert "evidence_quote" in r
        assert isinstance(r.get("is_correct"), bool)


def test_get_attempt_returns_same_results(started_attempt):
    res = requests.get(
        f"{BASE_URL}/reading/attempt/{started_attempt['attempt_id']}",
        headers=_auth_headers(),
        timeout=30,
    )
    assert res.status_code == 200
    body = res.json()
    assert body["attempt_id"] == started_attempt["attempt_id"]
    assert isinstance(body.get("results"), list) and len(body["results"]) == 9
    assert isinstance(body.get("passage_body"), str)


def test_history_includes_submitted_attempt(started_attempt):
    res = requests.get(
        f"{BASE_URL}/reading/history?limit=10",
        headers=_auth_headers(),
        timeout=30,
    )
    assert res.status_code == 200
    body = res.json()
    attempt_ids = [a["attempt_id"] for a in body.get("attempts", [])]
    assert started_attempt["attempt_id"] in attempt_ids


def test_quota_endpoint_shape():
    res = requests.get(
        f"{BASE_URL}/reading/quota",
        headers=_auth_headers(),
        timeout=30,
    )
    assert res.status_code == 200
    body = res.json()
    assert "used_today" in body
    assert "limit" in body
    assert "is_pro" in body
    assert isinstance(body["used_today"], int)
    assert isinstance(body["is_pro"], bool)


def test_second_attempt_same_day_blocks_free_user():
    """
    After the previous tests have submitted today's allowance, a fresh
    generate call should return 403 with the project's standard paywall
    error shape. Skipped if the test user is Pro (no way to enforce a
    quota for them).
    """
    quota_res = requests.get(
        f"{BASE_URL}/reading/quota",
        headers=_auth_headers(),
        timeout=30,
    )
    assert quota_res.status_code == 200
    if quota_res.json().get("is_pro"):
        pytest.skip("Test user is Pro; quota bypass test belongs in the Pro suite")

    res = requests.post(
        f"{BASE_URL}/reading/passage/generate",
        headers={**_auth_headers(), "Content-Type": "application/json"},
        json={},
        timeout=30,
    )
    assert res.status_code == 403, (
        f"expected 403 paywall, got {res.status_code}: {res.text[:300]}"
    )
    detail = res.json().get("detail")
    # detail can be a dict (FastAPI HTTPException) or string (rate limiter)
    if isinstance(detail, dict):
        assert detail.get("error") == "quota_exceeded"
        assert detail.get("redirect") == "/upgrade"
    # If it's a string it's still a valid 403, but the shape isn't ours —
    # fail loudly so we notice.
    else:
        pytest.fail(f"unexpected 403 detail shape: {detail!r}")


@pytest.mark.skipif(
    not PRO_USER_TOKEN,
    reason=(
        "READING_E2E_PRO_USER_TOKEN must be set to a dedicated permanent-Pro "
        "test user. Pro-bypass test does NOT grant Pro itself; see the "
        '"E2E test users" section of _QA_reading_module.md for setup.'
    ),
)
def test_pro_user_bypasses_quota():
    """
    Verify that a permanently-Pro user can summon passages without hitting
    the free quota.

    Felix's call (Prompt 5): this test no longer touches admin endpoints
    and no longer mutates is_pro_grant. The Pro user must already exist
    and be permanently provisioned via the admin tool, separately from the
    free test user. This keeps prod state stable across e2e runs.
    """
    # First confirm the configured user really is Pro — if someone wired
    # a free-tier token into READING_E2E_PRO_USER_TOKEN by mistake, fail
    # loudly rather than masquerading a quota miss as a "Pro works" pass.
    quota_res = requests.get(
        f"{BASE_URL}/reading/quota",
        headers=_auth_headers(PRO_USER_TOKEN),
        timeout=30,
    )
    assert quota_res.status_code == 200, (
        f"quota probe for pro user failed: {quota_res.status_code}: "
        f"{quota_res.text[:300]}"
    )
    quota_body = quota_res.json()
    assert quota_body.get("is_pro") is True, (
        "READING_E2E_PRO_USER_TOKEN must belong to a Pro user; "
        f"got is_pro={quota_body.get('is_pro')!r}. "
        "Refer to _QA_reading_module.md → E2E test users for setup."
    )

    res = requests.post(
        f"{BASE_URL}/reading/passage/generate",
        headers={**_auth_headers(PRO_USER_TOKEN), "Content-Type": "application/json"},
        json={},
        timeout=120,
    )
    assert res.status_code == 200, (
        f"Pro user blocked at quota: {res.status_code}: {res.text[:300]}"
    )
    body = res.json()
    assert _is_uuid(body.get("passage_id"))
