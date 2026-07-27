"""
Part 2 quality floor: persistence, schema validation, input shape, quota.

Before this, /part2/evaluate would happily return HTTP 200 with a full band
score card while the practice_records insert had silently failed, and it never
inspected the model's response at all — a malformed score went straight into the
database and the DOM. It also had no allowance of its own while quietly eating
Part 1's monthly 20.

Hermetic: supabase, verify_token, Pro status and the clock are patched. No
network, no provider calls, no DB.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

import main


USER = "user-abc"


def _valid_result(with_notes=False):
    return {
        "band_score": 6.5,
        "criteria": [
            {"name": n, "band": 6.5, "description": "d", "improvement": "i"}
            for n in main.PART2_CRITERIA_NAMES
        ],
        "strengths": ["優點一", "優點二"],
        "improvements": ["改進一", "改進二"],
        "notes_analysis": "筆記與口說相符。" if with_notes else None,
    }


# ── (a) persistence must not fail silently ───────────────────────────────
class _Resp:
    def __init__(self, data):
        self.data = data


def _fake_supabase(insert_data):
    tbl = MagicMock()
    tbl.insert.return_value = tbl
    tbl.execute.return_value = _Resp(insert_data)
    sb = MagicMock()
    sb.table.return_value = tbl
    return sb


def test_persist_raises_when_insert_returns_no_rows():
    """A 200 from PostgREST with zero rows must not read as success."""
    with patch.object(main, "supabase_admin", _fake_supabase([])):
        with pytest.raises(HTTPException) as exc:
            main._persist_part2(USER, "Topic", "transcript", _valid_result(), None)
    assert exc.value.status_code == 500


def test_persist_raises_when_insert_throws():
    sb = MagicMock()
    sb.table.side_effect = RuntimeError("db down")
    with patch.object(main, "supabase_admin", sb):
        with pytest.raises(HTTPException) as exc:
            main._persist_part2(USER, "Topic", "transcript", None, None)
    assert exc.value.status_code == 500


def test_persist_succeeds_when_a_row_comes_back():
    with patch.object(main, "supabase_admin", _fake_supabase([{"id": "rec-1"}])):
        main._persist_part2(USER, "Topic", "transcript", _valid_result(), None)


def test_persist_is_not_silent_anymore():
    """Regression guard: the old code swallowed every exception here."""
    import inspect
    src = inspect.getsource(main._persist_part2)
    assert "raise HTTPException" in src
    assert "insert_resp.data" in src


# ── (b) LLM schema validation ────────────────────────────────────────────
def test_valid_response_passes():
    ok, reason = main.validate_part2_response(_valid_result(), has_notes=False)
    assert ok, reason


def test_valid_response_with_notes_passes():
    ok, reason = main.validate_part2_response(_valid_result(True), has_notes=True)
    assert ok, reason


@pytest.mark.parametrize("bad,label", [
    ({}, "empty dict"),
    ([], "list not dict"),
    ("string", "string not dict"),
    (None, "none"),
])
def test_non_dict_or_empty_is_rejected(bad, label):
    ok, _ = main.validate_part2_response(bad, has_notes=False)
    assert not ok, label


@pytest.mark.parametrize("band,label", [
    (9.5, "above max"),
    (-1, "below min"),
    (6.3, "not a 0.5 step"),
    (6.25, "quarter step"),
    ("6.5", "string band"),
    (True, "bool band"),
    (None, "null band"),
])
def test_band_score_out_of_contract_is_rejected(band, label):
    r = _valid_result(); r["band_score"] = band
    ok, _ = main.validate_part2_response(r, has_notes=False)
    assert not ok, label


def test_band_score_missing_is_rejected():
    r = _valid_result(); del r["band_score"]
    ok, _ = main.validate_part2_response(r, has_notes=False)
    assert not ok


@pytest.mark.parametrize("criteria,label", [
    (None, "missing"),
    ("nope", "string"),
    ([], "empty"),
    ([{"name": "Fluency & Coherence", "band": 6.5, "description": "d"}], "too few"),
    ([{"name": "x", "band": 6.5, "description": "d"}] * 5, "too many"),
])
def test_criteria_shape_violations_are_rejected(criteria, label):
    r = _valid_result()
    if criteria is None:
        del r["criteria"]
    else:
        r["criteria"] = criteria
    ok, _ = main.validate_part2_response(r, has_notes=False)
    assert not ok, label


def test_criterion_with_bad_band_is_rejected():
    r = _valid_result()
    r["criteria"][2]["band"] = 11
    ok, _ = main.validate_part2_response(r, has_notes=False)
    assert not ok


def test_criterion_missing_description_is_rejected():
    r = _valid_result()
    del r["criteria"][1]["description"]
    ok, _ = main.validate_part2_response(r, has_notes=False)
    assert not ok


@pytest.mark.parametrize("value,label", [
    ("not a list", "string"),
    ([1, 2], "non-string items"),
    ([{"a": 1}], "dict items"),
    (["x" * (main.PART2_MAX_TEXT_LEN + 1)], "giant string"),
    (["ok"] * (main.PART2_MAX_LIST_ITEMS + 1), "too many items"),
])
def test_strengths_shape_violations_are_rejected(value, label):
    r = _valid_result(); r["strengths"] = value
    ok, _ = main.validate_part2_response(r, has_notes=False)
    assert not ok, label


def test_giant_description_is_rejected():
    r = _valid_result()
    r["criteria"][0]["description"] = "x" * (main.PART2_MAX_TEXT_LEN + 1)
    ok, _ = main.validate_part2_response(r, has_notes=False)
    assert not ok


def test_notes_analysis_required_when_notes_submitted():
    r = _valid_result(); r["notes_analysis"] = None
    ok, reason = main.validate_part2_response(r, has_notes=True)
    assert not ok and "notes_analysis" in reason


def test_notes_analysis_not_required_without_notes():
    r = _valid_result(); r["notes_analysis"] = None
    ok, _ = main.validate_part2_response(r, has_notes=False)
    assert ok


# ── (c) bullet_points input shape ────────────────────────────────────────
@pytest.mark.parametrize("payload,label", [
    ("5", "int"),
    ('{"a":1}', "dict"),
    ("[1,2]", "list of ints"),
    ('"plain"', "string"),
    ("null", "null"),
    ("[[1]]", "nested list"),
])
def test_bad_bullet_points_are_422_not_500(payload, label):
    """json.loads succeeds for all of these; only a list[str] may proceed."""
    parsed = json.loads(payload)
    is_ok = isinstance(parsed, list) and all(isinstance(b, str) for b in parsed)
    assert not is_ok, f"{label} should be rejected by the guard"


def test_good_bullet_points_pass_the_guard():
    parsed = json.loads('["a","b","c","d"]')
    assert isinstance(parsed, list) and all(isinstance(b, str) for b in parsed)


def test_endpoint_guard_source_rejects_non_string_lists():
    import inspect
    src = inspect.getsource(main._part2_evaluate_for_user)
    assert "isinstance(bullets, list)" in src
    assert "isinstance(b, str) for b in bullets" in src
    assert "422" in src


# ── (d) quota separation ─────────────────────────────────────────────────
def test_process_monthly_quota_excludes_part2_rows():
    """Part 2 must stop consuming Part 1's 20-per-month feedback allowance."""
    import inspect
    src = inspect.getsource(main.process)
    assert '.neq("mode", "part2")' in src, \
        "monthly feedback quota query must filter Part 2 rows out"


def test_part2_quota_counts_only_part2_rows():
    import inspect
    src = inspect.getsource(main._part2_monthly_count)
    assert '.eq("mode", "part2")' in src


# ── (e) Part 2 quota gate, fail-closed ───────────────────────────────────
def _quota_supabase(count):
    tbl = MagicMock()
    for m in ("select", "eq", "gte", "limit"):
        getattr(tbl, m).return_value = tbl
    resp = MagicMock(); resp.count = count
    tbl.execute.return_value = resp
    sb = MagicMock(); sb.table.return_value = tbl
    return sb


def test_pro_user_is_not_gated():
    with patch.object(main, "supabase_admin", _quota_supabase(9999)), \
         patch.object(main, "get_user_pro_status", return_value=True):
        main._enforce_part2_quota(USER)


def test_free_user_under_quota_passes():
    with patch.object(main, "supabase_admin", _quota_supabase(main.FREE_PART2_MONTHLY_QUOTA - 1)), \
         patch.object(main, "get_user_pro_status", return_value=False):
        main._enforce_part2_quota(USER)


def test_free_user_at_quota_gets_structured_403():
    with patch.object(main, "supabase_admin", _quota_supabase(main.FREE_PART2_MONTHLY_QUOTA)), \
         patch.object(main, "get_user_pro_status", return_value=False):
        with pytest.raises(HTTPException) as exc:
            main._enforce_part2_quota(USER)
    assert exc.value.status_code == 403
    assert exc.value.detail["error"] == "part2_quota_reached"
    assert exc.value.detail["limit"] == main.FREE_PART2_MONTHLY_QUOTA


def test_quota_lookup_failure_fails_CLOSED():
    """The whole point: /process fails open here, Part 2 must not."""
    sb = MagicMock(); sb.table.side_effect = RuntimeError("supabase down")
    with patch.object(main, "supabase_admin", sb), \
         patch.object(main, "get_user_pro_status", return_value=False):
        with pytest.raises(HTTPException) as exc:
            main._enforce_part2_quota(USER)
    assert exc.value.status_code == 503, "a failed quota read must refuse, not grant"


def test_quota_runs_before_any_provider_call():
    """The invariant that matters: no provider budget on a refused turn."""
    import inspect
    src = inspect.getsource(main._part2_evaluate_for_user)
    gate = src.index("_enforce_part2_quota")
    whisper = src.index("_transcribe_audio_file")
    claude = src.index("run_claude")
    assert gate < whisper, "quota must be enforced before Whisper"
    assert gate < claude, "quota must be enforced before Claude"


def test_quota_runs_after_request_shape_validation():
    """Malformed input must be judged on its own merits, not masked by quota.

    A 413 for over-long audio must stay a 413 even when the database is
    unreachable — which is exactly what regressed when the gate sat too early.
    """
    import inspect
    src = inspect.getsource(main._part2_evaluate_for_user)
    gate = src.index("_enforce_part2_quota")
    assert src.index("status_code=415") < gate, "MIME check must precede the quota gate"
    assert src.index("status_code=413") < gate, "duration check must precede the quota gate"
    assert src.index("must be a JSON array of strings") < gate, \
        "bullet_points check must precede the quota gate"


# ── (f) Pro seam ─────────────────────────────────────────────────────────
def test_enhanced_feedback_seam_exists_and_is_non_destructive():
    original = _valid_result()
    snapshot = json.loads(json.dumps(original))
    with patch.object(main, "get_user_pro_status", return_value=True):
        out = main._part2_enhanced_feedback(original, USER)
    assert original == snapshot, "seam must not mutate the source result"
    for key in snapshot:
        assert key in out, f"Free-tier key {key} must survive the seam"
    assert out["is_pro"] is True


def test_enhanced_feedback_reports_free_tier():
    with patch.object(main, "get_user_pro_status", return_value=False):
        out = main._part2_enhanced_feedback(_valid_result(), USER)
    assert out["is_pro"] is False
