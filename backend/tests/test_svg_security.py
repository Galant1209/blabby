"""Security regression tests for model-generated Writing chart SVG."""

from __future__ import annotations

import pytest

pytest.importorskip("defusedxml")

import main


SAFE_SVG = """<svg viewBox="0 0 600 420" xmlns="http://www.w3.org/2000/svg">
<rect width="600" height="420" fill="#ffffff"/>
<text x="20" y="30" font-family="Georgia, serif">正常 English 與繁體中文</text>
<path d="M10 10 L20 20" fill="none" stroke="#1A3550"/>
</svg>"""


@pytest.mark.parametrize(
    "payload",
    [
        '<svg xmlns="http://www.w3.org/2000/svg"><image href="x" onerror="alert(1)"/></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg" onload="alert(1)"></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg"><a href="javascript:alert(1)"><text>x</text></a></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg"><foreignObject><iframe src="x"/></foreignObject></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg"><g><svg></g></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg"><svg><text>x</text></svg></svg>',
        '<svg xmlns="http://www.w3.org/2000/svg"><rect fill="url(https://evil.example/a.svg#x)"/></svg>',
    ],
    ids=[
        "img-onerror", "svg-onload", "javascript-url", "nested-foreign-object",
        "malformed-nested-svg", "wellformed-nested-svg", "external-paint-reference",
    ],
)
def test_sanitize_chart_svg_rejects_active_or_malformed_content(payload):
    with pytest.raises(ValueError):
        main.sanitize_chart_svg(payload)


def test_sanitize_chart_svg_preserves_safe_english_and_traditional_chinese():
    sanitized = main.sanitize_chart_svg(SAFE_SVG)
    assert "正常 English 與繁體中文" in sanitized
    assert "<script" not in sanitized
    assert " onload=" not in sanitized


def test_sanitize_chart_svg_allows_only_local_marker_reference():
    safe = """<svg viewBox="0 0 10 10" xmlns="http://www.w3.org/2000/svg">
    <defs><marker id="arrow"><path d="M0 0 L1 1"/></marker></defs>
    <line x1="0" y1="0" x2="1" y2="1" marker-end="url(#arrow)"/>
    </svg>"""
    assert "url(#arrow)" in main.sanitize_chart_svg(safe)

    external = safe.replace("url(#arrow)", "url(https://evil.example/a.svg#arrow)")
    with pytest.raises(ValueError):
        main.sanitize_chart_svg(external)
