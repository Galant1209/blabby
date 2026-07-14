"""Regression coverage for deterministic Writing Task 1 pie charts."""

import asyncio
import json
import math
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import ValidationError
from starlette.requests import Request

import main


VALID = {
    "chart_type": "pie_chart",
    "title": "Household expenditure by category",
    "labels": ["Housing", "Transport", "Food", "Leisure", "Other"],
    "values": [38, 28, 18, 12, 4],
    "unit": "%",
}


def _chart(**updates):
    return main.PieChartData(**{**VALID, **updates})


def test_expected_percentage_values_pass_and_build_deterministic_artifacts():
    with patch.object(main, "generate_chart_svg") as svg_model:
        artifacts = main._pie_artifacts_from_response({"context": "", "chart": VALID})
    svg_model.assert_not_called()
    assert artifacts["chart_data"]["values"] == [38.0, 28.0, 18.0, 12.0, 4.0]
    assert artifacts["chart_svg"] is None
    assert artifacts["chart_description"].startswith("Category | Value (%)")
    assert artifacts["prompt"].startswith(
        "The pie chart below shows Household expenditure by category. "
    )
    assert "ofhousehold" not in artifacts["prompt"].lower()


@pytest.mark.parametrize(
    "values",
    [
        [38, 28, -1, 31, 4],
        [math.nan, 28, 18, 12, 42],
        [math.inf, 28, 18, 12, 4],
        [0, 0, 0, 0, 0],
    ],
    ids=["negative", "nan", "infinity", "all-zero"],
)
def test_invalid_values_are_rejected(values):
    with pytest.raises(ValidationError):
        _chart(values=values)


def test_labels_and_values_must_match():
    with pytest.raises(ValidationError):
        _chart(labels=["A", "B"])


@pytest.mark.parametrize("count", [1, 9])
def test_category_count_bounds(count):
    with pytest.raises(ValidationError):
        _chart(labels=[f"L{i}" for i in range(count)], values=[100 / count] * count)


@pytest.mark.parametrize("values", [[99, 1], [100, 0], [101, 0]])
def test_percentage_tolerance_accepts_99_through_101(values):
    assert sum(_chart(labels=["A", "B"], values=values).values) == sum(values)


@pytest.mark.parametrize("values", [[98, 0], [102, 0]])
def test_percentage_totals_outside_tolerance_are_rejected(values):
    with pytest.raises(ValidationError):
        _chart(labels=["A", "B"], values=values)


def test_malicious_label_remains_plain_data_and_geometry_fields_are_forbidden():
    payload = _chart(
        labels=["<img src=x onerror=alert(1)>", "Safe"], values=[50, 50]
    )
    assert payload.labels[0] == "<img src=x onerror=alert(1)>"
    for field, value in {
        "x": 10, "y": 10, "radius": 80, "offset": 20,
        "transform": "translate(10,10)", "color": "red",
        "url": "https://evil.example/", "style": "background:url(x)",
    }.items():
        with pytest.raises(ValidationError):
            main.PieChartData(**{**VALID, field: value})


def test_legacy_pipe_description_parses_through_same_schema():
    parsed = main.parse_legacy_chart_description(
        "Category | Share\nHousing | 38\nTransport | 28\nFood | 18\nLeisure | 12\nOther | 4",
        "Household expenditure",
    )
    assert parsed is not None
    assert parsed.values == [38, 28, 18, 12, 4]


@pytest.mark.parametrize(
    "description",
    ["not a table", "Category | A | B\nHousing | 50 | 50", "Category | Value\nA | nope\nB | 100"],
)
def test_malformed_or_multi_period_legacy_description_safely_falls_back(description):
    assert main.parse_legacy_chart_description(description, "Title") is None


def test_missing_word_boundary_is_rejected_instead_of_served():
    with pytest.raises(ValidationError):
        _chart(title="Distribution ofhousehold expenditure")


def test_pure_js_geometry_is_contiguous_and_finite():
    source = (Path(__file__).parents[2] / "frontend/app/writing.html").read_text()
    geometry = source.split("// PURE_GEOMETRY_START", 1)[1].split(
        "// PURE_GEOMETRY_END", 1
    )[0]
    script = geometry + "\n" + r"""
const slices = calculatePieSlices([38,28,18,12,4], 120, 120, 90);
if (slices.length !== 5) throw new Error('slice count');
for (let i = 0; i < slices.length; i++) {
  const slice = slices[i];
  if (slice.cx !== 120 || slice.cy !== 120 || slice.radius !== 90 || slice.offset !== 0) throw new Error('geometry');
  if (![slice.startAngle, slice.endAngle, slice.cx, slice.cy, slice.radius].every(Number.isFinite)) throw new Error('finite');
  if (i && Math.abs(slice.startAngle - slices[i - 1].endAngle) > 1e-12) throw new Error('gap');
}
if (Math.abs(slices[0].startAngle + Math.PI / 2) > 1e-12) throw new Error('start');
if (Math.abs(slices.at(-1).endAngle - (3 * Math.PI / 2)) > 1e-12) throw new Error('end');
if (Math.abs((slices.at(-1).endAngle - slices[0].startAngle) - 2 * Math.PI) > 1e-12) throw new Error('total');
"""
    result = subprocess.run(["node", "-e", script], capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr


def test_live_pie_endpoint_inserts_rebuildable_description_without_svg():
    inserted = []

    class Query:
        def __init__(self):
            self.operation = None

        def select(self, *args, **kwargs):
            self.operation = "select"
            return self

        def insert(self, payload):
            self.operation = "insert"
            inserted.append(payload)
            return self

        def eq(self, *args, **kwargs): return self
        def order(self, *args, **kwargs): return self
        def limit(self, *args, **kwargs): return self
        def not_(self, *args, **kwargs): return self

        def execute(self):
            if self.operation == "insert":
                return SimpleNamespace(data=[{"id": "question-id"}], count=None)
            return SimpleNamespace(data=[], count=0)

    class Supabase:
        def table(self, _name):
            return Query()

    model_response = SimpleNamespace(
        content=[SimpleNamespace(text=json.dumps({"context": "", "chart": VALID}))]
    )
    request = Request({
        "type": "http", "http_version": "1.1", "method": "GET",
        "scheme": "http", "path": "/api/writing/question", "root_path": "",
        "query_string": b"", "headers": [], "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    })
    with patch.object(main, "supabase_admin", Supabase()), \
         patch.object(main, "verify_token", return_value="user-a"), \
         patch.object(main, "is_user_pro", new=AsyncMock(return_value=True)), \
         patch.object(main.anthropic_client.messages, "create", return_value=model_response), \
         patch.object(main, "generate_chart_svg") as svg_model:
        result = asyncio.run(main.writing_get_question(
            request=request,
            task_type="task1",
            task1_subtype="pie_chart",
            authorization="Bearer test",
        ))

    svg_model.assert_not_called()
    assert result["chart_data"]["values"] == [38.0, 28.0, 18.0, 12.0, 4.0]
    assert result["chart_svg"] is None
    assert inserted[0]["chart_svg"] is None
    assert inserted[0]["chart_description"].startswith("Category | Value (%)")


def test_cached_legacy_pie_ignores_svg_and_returns_same_chart_contract():
    pool_reads = 0

    class Query:
        def __init__(self):
            self.operation = None

        def select(self, *args, **kwargs): self.operation = "select"; return self
        def update(self, _payload): self.operation = "update"; return self
        def eq(self, *args, **kwargs): return self
        def order(self, *args, **kwargs): return self
        def limit(self, *args, **kwargs): return self
        def not_(self, *args, **kwargs): return self

        def execute(self):
            nonlocal pool_reads
            if self.operation == "update":
                return SimpleNamespace(data=[{}], count=None)
            pool_reads += 1
            if pool_reads == 1:
                return SimpleNamespace(data=[{
                    "id": "legacy-id",
                    "prompt": "The pie chart below shows household expenditure.",
                    "chart_description": "Category | Share\nHousing | 38\nTransport | 28\nFood | 18\nLeisure | 12\nOther | 4",
                    "chart_svg": '<svg><script src="https://evil.example/x.js"/></svg>',
                    "essay_type": None,
                    "task1_subtype": "pie_chart",
                    "used_count": 0,
                }], count=1)
            return SimpleNamespace(data=[], count=10)

    class Supabase:
        def table(self, _name): return Query()

    request = Request({
        "type": "http", "http_version": "1.1", "method": "GET",
        "scheme": "http", "path": "/api/writing/question", "root_path": "",
        "query_string": b"", "headers": [], "client": ("127.0.0.1", 1234),
        "server": ("testserver", 80),
    })
    with patch.object(main, "supabase_admin", Supabase()), \
         patch.object(main, "verify_token", return_value="user-a"), \
         patch.object(main, "is_user_pro", new=AsyncMock(return_value=True)), \
         patch.object(main.anthropic_client.messages, "create") as model:
        result = asyncio.run(main.writing_get_question(
            request=request, task_type="task1", task1_subtype="pie_chart",
            authorization="Bearer test",
        ))
    model.assert_not_called()
    assert result["chart_svg"] is None
    assert result["chart_data"]["labels"][0] == "Housing"
    assert "evil.example" not in json.dumps(result)
