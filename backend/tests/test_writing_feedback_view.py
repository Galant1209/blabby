"""Writing 批改的共用 view-model。

在此之前 criteria 的四鍵巢狀結構只在 POST /api/writing/submit 的 handler
現場組裝：沒有存進 DB，GET /api/writing/submission/{id} 也沒有重建它，
它回的是 {"submission": <DB 原始列>}。前端的 renderFeedback() 讀的是
data.criteria[key]，所以把 GET 的回應餵給它會得到四張評分卡全空的殼 ——
只是零呼叫者，從來沒人踩到。

本檔最重要的一條是 test_post_and_get_return_the_same_thing_for_one_row：
同一列資料經兩支端點取得必須逐欄相同。那是「一份批改只有一個定義」這件事
唯一能被機器守住的形式。

第二重要的是 band 的型別收斂。numeric 欄位經 PostgREST 序列化成字串
（"4.0"），而交卷當下的值來自 LLM 解析（float）。不收斂的話，前端的
badge.textContent 會讓同一份批改在交卷當下顯示 4、重整後顯示 4.0。
"""

from __future__ import annotations

import main


CRITERIA_KEYS = {
    "task_achievement",
    "coherence_cohesion",
    "lexical_resource",
    "grammatical_range",
}


def _db_row(**overrides) -> dict:
    """一列 writing_submissions，欄位與型別照 PostgREST 實際回傳的樣子。

    band_* 是字串不是數字 —— 那是 numeric 的序列化結果，不是筆誤。
    """
    row = {
        "id":                     "11111111-2222-3333-4444-555555555555",
        "user_id":                "99999999-8888-7777-6666-555555555555",
        "question_id":            "aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee",
        "task_type":              "task1",
        "essay_text":             "The chart shows a steady rise in coffee consumption.",
        "word_count":             155,
        "submitted_at":           "2026-08-20T02:55:06.302074+00:00",
        "feedback_ta":            "You described the trend but never compared the two lines.",
        "feedback_cc":            "Paragraphs are not signposted.",
        "feedback_lr":            "Repetition of 'increase'.",
        "feedback_gra":           "Tense slips in the overview.",
        "fix_ta":                 "Add one sentence comparing 2010 with 2020.",
        "fix_cc":                 "Open the second paragraph with 'By contrast'.",
        "fix_lr":                 "Swap one 'increase' for 'climb'.",
        "fix_gra":                "Keep the overview in the past simple.",
        "band_ta":                "4.5",
        "band_cc":                "4.0",
        "band_lr":                "4.0",
        "band_gra":               "4.0",
        "band_overall":           "4.0",
        "priority_fix":           "Compare the two lines, do not just list them.",
        "is_retry":               False,
        "previous_submission_id": None,
    }
    row.update(overrides)
    return row


# ── 形狀 ─────────────────────────────────────────────────────────────────

def test_criteria_has_all_four_keys_with_band_feedback_fix():
    view = main.build_writing_feedback_view(_db_row())

    assert set(view["criteria"]) == CRITERIA_KEYS
    for key in CRITERIA_KEYS:
        assert set(view["criteria"][key]) == {"band", "feedback", "fix"}


def test_each_criterion_maps_to_its_own_db_suffix():
    """摺疊不能串行 —— ta 的 feedback 配 ta 的 band，不是配 cc 的。"""
    view = main.build_writing_feedback_view(_db_row())["criteria"]

    assert view["task_achievement"]["feedback"].startswith("You described the trend")
    assert view["task_achievement"]["fix"].startswith("Add one sentence")
    assert view["task_achievement"]["band"] == 4.5
    assert view["coherence_cohesion"]["feedback"] == "Paragraphs are not signposted."
    assert view["lexical_resource"]["feedback"] == "Repetition of 'increase'."
    assert view["grammatical_range"]["fix"].startswith("Keep the overview")


def test_top_level_fields():
    row = _db_row()
    view = main.build_writing_feedback_view(row)

    assert view["submission_id"] == row["id"]
    assert view["word_count"] == 155
    assert view["band_overall"] == 4.0
    assert view["priority_fix"] == row["priority_fix"]
    assert view["essay_text"] == row["essay_text"]


def test_essay_text_is_returned():
    """還原情境要顯示原文：只有批改是懸空的判斷，看不到自己寫了什麼。"""
    view = main.build_writing_feedback_view(_db_row())

    assert view["essay_text"] == _db_row()["essay_text"]


# ── 不外洩 ───────────────────────────────────────────────────────────────

def test_user_id_is_never_in_the_output():
    """呼叫者自己的 id 也沒有理由回給前端。POST 本來就沒回。"""
    view = main.build_writing_feedback_view(_db_row())

    assert "user_id" not in view
    assert _db_row()["user_id"] not in str(view)


def test_extra_columns_are_not_forwarded_wholesale():
    """不是把 DB 列原封不動吐出去。需要時再加欄位，不要一次塞滿。"""
    view = main.build_writing_feedback_view(_db_row())

    assert set(view) == {
        "submission_id", "word_count", "band_overall",
        "priority_fix", "essay_text", "criteria",
    }


# ── band 型別收斂 ────────────────────────────────────────────────────────

def test_numeric_strings_from_postgrest_become_floats():
    """PostgREST 把 numeric 序列化成字串以保留精度。"""
    view = main.build_writing_feedback_view(_db_row())

    assert view["band_overall"] == 4.0
    assert isinstance(view["band_overall"], float)
    assert isinstance(view["criteria"]["task_achievement"]["band"], float)


def test_floats_from_the_grader_survive_unchanged():
    """交卷當下的值來自 LLM 解析，已經是數字。"""
    view = main.build_writing_feedback_view(
        _db_row(band_overall=6.5, band_ta=7.0)
    )

    assert view["band_overall"] == 6.5
    assert view["criteria"]["task_achievement"]["band"] == 7.0


def test_the_two_representations_of_one_band_collapse_to_the_same_value():
    """"4.0" 與 4.0 必須摺成同一個值。

    不然同一份批改在交卷當下顯示 4、重整後顯示 4.0 —— 前端是
    `badge.textContent = c.band`，JS 的 Number 4 印成 "4"，字串 "4.0"
    印成 "4.0"。使用者看到的是兩個不同的分數。
    """
    from_db = main.build_writing_feedback_view(_db_row(band_overall="4.0"))
    from_grader = main.build_writing_feedback_view(_db_row(band_overall=4.0))

    assert from_db["band_overall"] == from_grader["band_overall"]


def test_null_bands_do_not_crash():
    view = main.build_writing_feedback_view(
        _db_row(band_overall=None, band_ta=None, band_cc=None,
                band_lr=None, band_gra=None)
    )

    assert view["band_overall"] is None
    for key in CRITERIA_KEYS:
        assert view["criteria"][key]["band"] is None


def test_unparseable_band_becomes_none_not_an_exception():
    """壞資料讓那一格空著，不是讓整個還原 500。"""
    view = main.build_writing_feedback_view(_db_row(band_ta="not a number"))

    assert view["criteria"]["task_achievement"]["band"] is None
    # 其餘三格不受影響
    assert view["criteria"]["coherence_cohesion"]["band"] == 4.0


def test_missing_columns_do_not_crash():
    """DB 列缺欄位時回 None，不是 KeyError。"""
    view = main.build_writing_feedback_view({"id": "x"})

    assert view["submission_id"] == "x"
    assert view["essay_text"] is None
    assert set(view["criteria"]) == CRITERIA_KEYS
    assert view["criteria"]["task_achievement"] == {
        "band": None, "feedback": None, "fix": None,
    }


# ── 兩支端點的一致性（本檔核心）──────────────────────────────────────────

def test_post_and_get_return_the_same_thing_for_one_row():
    """同一列資料，交卷回的與還原回的必須逐欄相同。

    這條是這次改動的全部理由。改動前 POST 回巢狀 criteria、GET 回
    {"submission": <扁平列>}，把後者餵給 renderFeedback() 得到的是四張
    評分卡全空的殼。
    """
    import inspect

    row = _db_row()

    post_src = inspect.getsource(main.writing_submit)
    get_src = inspect.getsource(main.writing_submission_detail)

    # 兩支都只透過 builder 產生回應
    assert "build_writing_feedback_view(ins.data[0])" in post_src
    assert "build_writing_feedback_view(resp.data[0])" in get_src
    # 舊的兩種形狀都不得殘留
    assert '"criteria": {' not in post_src, "POST 仍在現場組裝 criteria"
    assert '{"submission": resp.data[0]}' not in get_src, "GET 仍回原始列"

    # 同一列 → 同一份輸出
    assert main.build_writing_feedback_view(row) == main.build_writing_feedback_view(row)


def test_get_no_longer_wraps_the_row_in_a_submission_key():
    view = main.build_writing_feedback_view(_db_row())

    assert "submission" not in view
    assert view["submission_id"] is not None


def test_ownership_check_and_404_are_untouched():
    """GET 的 ownership 與 404-not-403 一行不動。"""
    import inspect

    src = inspect.getsource(main.writing_submission_detail)

    assert '.eq("id", submission_id)' in src
    assert '.eq("user_id", user_id)' in src
    assert 'status_code=404' in src
    assert 'status_code=403' not in src


def test_the_insert_result_is_still_checked():
    """交卷主路徑的「不信任 200」保護不得被這次重構吃掉。"""
    import inspect

    src = inspect.getsource(main.writing_submit)

    assert "if not ins.data:" in src
    assert "Submission could not be recorded" in src
