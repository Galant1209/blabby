"""Source-level guardrails for the inline frontend renderers.

The application intentionally keeps JavaScript inline. These checks prevent
the two audited sinks from silently returning while browser-level verification
exercises the rendered behavior separately.
"""

from pathlib import Path


APP_DIR = Path(__file__).parents[2] / "frontend" / "app"


def test_part2_renderer_uses_text_nodes_not_html_templates():
    source = (APP_DIR / "index.html").read_text(encoding="utf-8")
    renderer = source.split("function renderPart2Score(data)", 1)[1].split(
        "// ── /Part 2 functions", 1
    )[0]
    assert "textContent" in renderer
    assert "replaceChildren" in renderer
    assert ".innerHTML" not in renderer
    for payload in (
        "<img src=x onerror=alert(1)>",
        "<svg onload=alert(1)>",
        "javascript:alert(1)",
    ):
        assert payload not in renderer


def test_writing_pie_chart_uses_canvas_and_text_only_legend():
    source = (APP_DIR / "writing.html").read_text(encoding="utf-8")
    renderer = source.split("function renderPieChart(raw)", 1)[1].split(
        "window.addEventListener('resize'", 1
    )[0]
    assert "createElement('canvas')" in renderer
    assert "calculatePieSlices" in renderer
    assert "text.textContent" in renderer
    assert "replaceChildren" in renderer
    assert "innerHTML" not in renderer
    assert "fetch(" not in renderer
    assert "raw.offset" not in renderer
    assert "raw.transform" not in renderer


def test_writing_legacy_svg_is_not_used_for_pie_or_injected_into_live_dom():
    source = (APP_DIR / "writing.html").read_text(encoding="utf-8")
    request_block = source.split("if (renderPieChart(data.chart_data))", 1)[1].split(
        "showElement('question-display')", 1
    )[0]
    assert "data.task1_subtype !== 'pie_chart' && data.chart_svg" in request_block
    assert "innerHTML" not in request_block
    assert "insertAdjacentHTML" not in source


def test_writing_legend_is_responsive_without_fixed_chart_width():
    source = (APP_DIR / "writing.html").read_text(encoding="utf-8")
    assert ".pie-chart-legend { display: grid" in source
    assert "auto-fit" in source
    assert "overflow-wrap: anywhere" in source
    assert "#chart-container { width: 100%; max-width: 100%; overflow: hidden" in source
