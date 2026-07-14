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


def test_writing_chart_is_rendered_as_image_not_live_svg_dom():
    source = (APP_DIR / "writing.html").read_text(encoding="utf-8")
    block = source.split("if (data.chart_svg)", 1)[1].split(
        "} else if (data.chart_description)", 1
    )[0]
    assert "createElement('img')" in block
    assert "encodeURIComponent" in block
    assert ".innerHTML" not in block
