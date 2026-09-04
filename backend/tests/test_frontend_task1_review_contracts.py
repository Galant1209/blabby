"""Static guardrails for the admin Task 1 chart human-review surface."""

from pathlib import Path


ROOT = Path(__file__).parents[2]
ADMIN = (ROOT / "frontend/app/admin.html").read_text(encoding="utf-8")
WRITING = (ROOT / "frontend/app/writing.html").read_text(encoding="utf-8")
RENDERER = (ROOT / "frontend/app/task1-chart-renderer.js").read_text(encoding="utf-8")


def test_admin_has_review_entry_and_queue_fetch():
    assert "Task 1 chart review" in ADMIN
    assert "/admin/writing/task1-review?review_status=" in ADMIN
    assert "state.task1ReviewRows" in ADMIN


def test_admin_reuses_shared_true_renderer_for_preview():
    assert "task1-chart-renderer.js" in ADMIN
    assert "renderer.renderPieChart" in ADMIN
    assert "renderer.renderLegacyChartImage" in ADMIN
    assert "chart_data" in ADMIN and "chart_svg" in ADMIN


def test_admin_shows_safe_raw_source_and_review_filters():
    assert "task1-review-source" in ADMIN
    assert "escapeHtml(row.chart_description" in ADMIN
    for status in ("pending", "needs_fix", "approved", "retired", "all"):
        assert f"['{status}'" in ADMIN


def test_admin_patch_is_review_only_and_never_reactivates():
    assert "/admin/writing/task1-review/${encodeURIComponent(questionId)}" in ADMIN
    assert "review_status: status" in ADMIN
    assert "review_issue: issue" in ADMIN
    assert "review_note: note" in ADMIN
    assert "Serving state was not changed." in ADMIN
    assert "engineering reactivation" in ADMIN
    assert "is_pregenerated" not in ADMIN


def test_admin_handles_loading_empty_and_api_error_states():
    assert "Loading Task 1 review queue" in ADMIN
    assert "No questions in this review filter." in ADMIN
    assert "Failed to load (${res.status})" in ADMIN
    assert "Network error" in ADMIN


def test_student_and_admin_both_load_the_same_renderer_module():
    assert '<script src="./task1-chart-renderer.js"></script>' in WRITING
    assert '<script src="./task1-chart-renderer.js"></script>' in ADMIN
    assert "function calculatePieSlices" in RENDERER
    assert "function validatePieChartData" in RENDERER
    assert "function renderPieChart(raw, container)" in RENDERER
