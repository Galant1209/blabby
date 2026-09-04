"""Static + behavioral contracts for the Vocabulary quota paywall."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


APP = Path(__file__).parents[2] / "frontend" / "app"
HARNESS = Path(__file__).with_name("frontend_vocabulary_paywall_behavior.mjs")


def _source() -> str:
    return (APP / "vocabulary.html").read_text(encoding="utf-8")


def _between(source: str, start: str, end: str) -> str:
    left = source.index(start)
    right = source.index(end, left)
    return source[left:right]


def test_vocabulary_posts_to_existing_save_endpoint():
    src = _source()
    assert "authJson('POST', '/api/vocabulary/my'" in src
    assert "vocabulary_item_id: id" in src
    assert "Authorization': `Bearer ${token}`" in src


def test_auth_json_parses_nested_backend_error_before_classifying_it():
    src = _source()
    auth = _between(src, "class ApiResponseError", "// Identify on auth ready")
    assert "if (r.status === 401)" in auth
    assert "detailObject?.error" in auth
    assert "detailObject?.message" in auth
    assert "if (r.status === 401 || r.status === 403)" not in auth
    assert "vocab_limit_reached" in src


def test_quota_403_opens_existing_paywall_and_uses_backend_limit():
    src = _source()
    add = _between(src, "async function onAddClick", "// Topic filter clicks")
    assert "isVocabularyQuotaError(e)" in add
    assert "showVocabProModal(e.detail && e.detail.limit)" in add
    assert "vocab-pro-modal-overlay" in src
    assert 'href="/upgrade.html?source=vocab_limit"' in src
    assert "解鎖無限單字" in src


def test_quota_failure_does_not_run_success_save_mutations():
    src = _source()
    add = _between(src, "async function onAddClick", "// Topic filter clicks")
    request_end = add.index("await authJson('POST', '/api/vocabulary/my'")
    before_request = add[:request_end]
    assert "mySetOfItemIds.add(id)" not in before_request
    assert "showAddedToast()" not in before_request
    assert "renderJournal(myItems, id)" not in before_request
    assert "if (isVocabularyQuotaError(e))" in add
    assert "return;" in add[add.index("if (isVocabularyQuotaError(e))") :]


def test_auth_and_permission_failures_do_not_use_quota_paywall_branch():
    src = _source()
    assert "error.status === 403 && error.code === 'vocab_limit_reached'" in src
    add = _between(src, "async function onAddClick", "// Topic filter clicks")
    quota_branch = _between(add, "if (isVocabularyQuotaError(e))", "alert(vocabularyAddErrorMessage")
    assert "showVocabProModal" in quota_branch
    assert "error.status === 401" in src
    assert "error.status === 403" in src
    assert "目前無法加入這個單字" in src
    assert "error.status === 404" in src
    assert "error.status >= 500" in src
    assert "error.name === 'TypeError'" in src


def test_paywall_is_hidden_accessible_dismissible_and_mobile_safe():
    src = _source()
    assert 'id="vocab-pro-modal-overlay" hidden' in src
    assert 'role="dialog"' in src
    assert 'aria-modal="true"' in src
    assert 'id="vocab-pro-modal-dismiss"' in src
    assert "event.key === 'Escape'" in src
    assert "overlay.hidden = true" in src
    assert "vocabPaywallReturnFocus" in src
    assert "@media (max-width: 390px)" in src
    assert ".pro-modal-actions { flex-direction: column; }" in src


def test_vocabulary_paywall_does_not_add_payment_or_analytics_schema():
    src = _source()
    assert "/api/payment/create-order" not in src
    assert "vocabulary_paywall" not in src
    assert "payment_funnel" not in src


def test_vocabulary_paywall_behavior_harness():
    if not HARNESS.exists():
        pytest.fail(f"missing harness: {HARNESS}")
    result = subprocess.run(
        ["node", str(HARNESS)],
        cwd=str(APP.parent.parent),
        capture_output=True,
        text=True,
        timeout=60,
    )
    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    assert result.returncode == 0, (
        f"vocabulary paywall behavior harness failed (exit {result.returncode}):\n"
        f"{result.stdout}\n{result.stderr}"
    )
