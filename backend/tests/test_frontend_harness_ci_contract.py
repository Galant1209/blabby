"""Keep every standalone frontend Node harness on a visible CI path."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).parents[2]
WORKFLOW = ROOT / ".github" / "workflows" / "test.yml"
HARNESS_DIR = Path(__file__).parent

DIRECT_CI_HARNESSES = (
    "frontend_active_vocabulary_behavior.mjs",
    "frontend_anonymous_conversion_behavior.mjs",
    "frontend_prescription_behavior.mjs",
    "frontend_progress_evidence_behavior.mjs",
    "frontend_resolution_cycle_behavior.mjs",
    "frontend_retention_behavior.mjs",
)

PYTEST_WRAPPED_HARNESSES = {
    "frontend_public_vocabulary_behavior.mjs": "test_public_vocabulary.py",
    "frontend_admin_member_behavior.mjs": "test_admin_member_contracts.py",
    "frontend_billing_behavior.mjs": "test_frontend_billing_contracts.py",
    "frontend_history_behavior.mjs": "test_frontend_history_contracts.py",
    "frontend_vocabulary_paywall_behavior.mjs": "test_frontend_vocabulary_paywall_contracts.py",
    "frontend_writing_restore_behavior.mjs": "test_writing_restore_contracts.py",
}


def _workflow() -> str:
    return WORKFLOW.read_text(encoding="utf-8")


def test_ci_pins_the_node_runtime_for_frontend_harnesses():
    source = _workflow()
    assert "uses: actions/setup-node@v4" in source
    assert "node-version: '20'" in source


def test_orphan_harnesses_are_each_explicitly_run_in_ci():
    source = _workflow()
    assert "name: Run orphan frontend Node contract harnesses" in source
    for harness in DIRECT_CI_HARNESSES:
        assert f"node backend/tests/{harness}" in source


def test_all_harnesses_have_a_current_ci_owner():
    source = _workflow()
    for harness in (*DIRECT_CI_HARNESSES, *PYTEST_WRAPPED_HARNESSES):
        assert (HARNESS_DIR / harness).is_file(), harness

    for harness, owner in PYTEST_WRAPPED_HARNESSES.items():
        assert f"node backend/tests/{harness}" not in source
        owner_source = (HARNESS_DIR / owner).read_text(encoding="utf-8")
        assert harness in owner_source
        assert "subprocess.run" in owner_source
