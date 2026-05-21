# Free / Pro 功能分級 — 變現層

全站 Free vs Pro gate 上線。所有限制在 server 端執行，frontend 在收到 403 時顯示升級提示。

## 限制內容

- **每月批改額度**：Free 用戶每月上限 20 次，超出回傳 `feedback_quota_reached`
- **單字本上限**：Free 用戶最多儲存 30 個單字，超出回傳 `vocab_limit_reached`
- **練習紀錄**：Free 用戶只拿最近 10 筆，response 帶 `capped: true`
- **診斷頁**：Free 用戶收到精簡 response，只含 top-1 弱點。`weakness_timeline`、`recent_trend`、`weakness_first_seen/last_seen` 為 Pro 專屬
- **三層 tag**：`tag_secondary` 和 `tag_tertiary` 為 Pro 專屬，Free 用戶只看主要 tag

## Frontend 升級提示

- 批改額度用完 → 錄音狀態列 + 金邊升級 CTA，導向 `/upgrade.html?source=feedback_quota`
- 單字達上限 → 按鈕改為「已達上限」+ 行內提示
- 練習紀錄截斷 → 最後一筆之後追加升級 banner
- 診斷頁精簡靜默處理（Free 用戶無破版感）

## upgrade.html

- 加入 7 天 Pro 試用券誘因文案
- 送出成功訊息更新為確認試用券發送


# Speaking Feedback — 留存與診斷全面升級

## 見證語（witness note）

`/process` 新增 `witness_note` 欄位，server 端從 `total_practice_count` + `tag_counts` 計算，零 LLM call。

- **里程碑**（10/20/30/50/100 次）：金邊框、Libre Baskerville、完整不透明度
- **卡關模式**（同 tag ≥ 3 次）：「這個問題你已經碰了 N 次。卡住不是能力問題，是還沒找到那個說法。」
- **一般累積**：「你已累計練習 N 次。」

## 三層弱點 tag

`/process` 新增 `tag_secondary` 和 `tag_tertiary`，LLM 依嚴重程度選出最多三個不重複 tag。Frontend 以遞減透明度渲染三個 chip（Pro 專屬；Free 只看主要 chip）。

## 弱點 tag 說明抽屜

點擊主要弱點 chip 展開行內抽屜，顯示 tag 定義、常見症狀、修法。完全 client-side。

## Tier-B 觸發條件拓寬

答案含具體細節（數字、地點、人名、時間、感官描述）即觸發 `progress_note`，不依賴 weak word 歷史。新用戶第一次練習就有機會收到正向肯定。

## Persona 注入擴展到所有 mode

`PERSONA_PROMPTS` Tier A–D 原本只在 drill mode 注入，現在所有 mode 都有效。

## Few-shot 全覆蓋

五個 `weakness_tag` 全部有 few-shot 範例。新增 Example 8（`lack_detail`）、Example 9（`safe_answer`）、Example 10（`grammar_minor`）。Example 2 tag 從 `weak_vocab` 修正為 `safe_answer`。

## `lack_detail` 文案修正

Session insight 從「答案太短」改為「考官想知道為什麼」。

## Repeat reminder 修正

`shouldShowRepeatReminder` 改為以 `tag_counts[tag] >= 2` 為主要信號，適用全部五個 tag（之前只對 `weak_vocab` 有效）。

## 主題偏向下一題

`/process` 新增 `pick_next_question`，三層選題：同 topic 未練過 → 任意 topic 未練過 → 完全 fallback。Frontend nextBtn 優先用 backend 回傳的題目。

## Dead code 清理

`classify_quality` 和 `classify_quality_background` 兩層移除未使用的 `coach_response` 參數。


# Progress Comparison 頁面

新增 `/progress.html`，並排比較同一題的第一次與最新一次作答。

- Backend `GET /api/progress`：依 `question` 分組，保留最早與最新筆，計算 `word_delta` 和 `sentence_delta`
- 空狀態：「You have not yet trodden this ground twice.」
- Delta 顯示：`+12 words · +2 sentences`，零值省略
- History 頁加入「View Progress」入口


# 帳號頁升級

- 新增弱點分布橫條圖（從 `weakness_counts` 渲染）
- 新增三個行動按鈕：繼續練習 / 查看進步對比 / 練習紀錄
- SEO：index.html 加入 meta description、OG tags、canonical；account.html 加 noindex


# Speaking Feedback Panel 重排

Feedback panel 順序調整，讓視覺終點落在行動導向（下一題、學到什麼），不是失敗標籤（你又卡了）。


# Reading Module v1 — SSE Streaming

passage 生成改為 SSE 串流，用戶 5 秒內開始看到文章出現，不再等待 45 秒空白。

- Backend 新增 `/reading/passage/generate_stream` SSE endpoint
- Frontend SSE parser、streaming render、fallback 至 blocking path
- Questions 在 passage stream 完成後並行觸發
- `reading_passages` 加 `created_by` 欄位，ownership 驗證


# Reading Module v1 — final polish

- 新增 `profiles.reading_band_updated_at`
- Reading vocab 存入時自動補繁體中文翻譯（`/vocab/translate_zh`，LRU cache）
- E2E Pro-bypass 測試改用專屬永久 Pro 測試用戶


# Band-adaptive Feedback Persona 系統

四層 Persona（A/B/C/D）依 `user_band` 加權移動平均（0.8/0.2）自動切換，注入 drill prompt。

- Band < 4.5 → Persona A（70% 鼓勵）
- Band 5 → Persona B（50/50）
- Band 6 → Persona C（30% 鼓勵）
- Band 7+ → Persona D（10% 鼓勵，考官模式）
- `profiles` 新增 `user_band`、`band_updated_at` 欄位


# Paywall 系統

- Server 端 evidence stripping（Free 用戶 quota 耗盡後 HTTP 403）
- 三個定價方案：7 天 / 30 天 / 年訂閱
- `FREE_DRILL_QUOTA = 20`
- Repeat question detection：`previous_transcript` 注入 system prompt，顯示「You have trodden this ground before」banner


# Speaking Drill — Pro 弱點訓練

- 兩個可訓練 tag：`weak_vocab`、`lack_detail`
- Rubric-based 評分系統，三狀態 red/yellow/green 進度指示
- Drill prompt 經過 v1–v6 六次迭代，準確率從 30% 提升至 80%
- Evidence sub-object 加入 response schema


# Schema 保護層與 Feedback 結構重整

- `/process` endpoint 加入 validator + retry layer（最多 3 次，失敗回 503）
- `better_phrasing` 拆為 `better_phrasing_en` + `better_phrasing_zh`
- `correction` 強制為 dict object，不允許 array


# 初版用戶準備

- Privacy Policy + Terms of Service 頁（繁體中文，符合個資法）
- Google OAuth consent screen 完成
- Part 1 題庫上線（40 題，15 個 topic）
- `QUESTION_BANK` 從 hardcoded 前端陣列遷移至 Supabase 動態載入
- Topic lock 邏輯：同 topic 連續 3 題後切換，顯示提示


# Groq Whisper 取代 OpenAI Whisper

轉錄速度從 ~6 秒降至 <1 秒。使用 `whisper-large-v3-turbo`。


# 單字本 SRS flip flow

原本直接顯示答案，改為先看單字 → 點 Reveal 才看答案再評分。符合 SRS 正確學習流程。


# Recent Practice Resume — migrated from localStorage to Supabase

Moved the "你上次卡在：…" continuation state out of the browser and into
`practice_records`. Cross-device, cache-clear-proof, single source of truth.

## Files modified / added

### DB (applied manually in Supabase SQL editor; no file committed)
- `practice_records` — new column `resolved boolean NOT NULL default false`
- `practice_records` — new partial index `idx_practice_records_user_unresolved`
  on `(user_id, created_at DESC) WHERE resolved = false`
- Backfill: every pre-migration row set to `resolved = true` so the hub
  never surfaces stale year-old sessions

### `backend/main.py`
- `POST /process`
  - Captures the newly-inserted row's id so the resolution lookup can
    exclude it without relying on ordering races
  - **Auto-resolution rule** after a successful insert:
    pick the most-recent prior unresolved record for the same user
    (top-5 by `created_at DESC`, id ≠ the row we just inserted),
    match `question` with whitespace-normalised equality in Python
    (no SQL-layer string compare — prevents silent skips over
    curly-vs-straight apostrophes / trailing whitespace),
    then if the matched prior's `weakness_tag` is non-empty /
    in `ALLOWED_WEAKNESS_TAGS` / different from the new tag → flip to
    `resolved = true`. Failure logged, never propagated.
- `GET /api/practice-records/last-unresolved` (new, 10/minute)
  - Returns `{id, question, topic, weakness_tag, coach_response, created_at}`
    of the caller's most recent unresolved row
  - No unresolved record → HTTP 200 with body `null` (not 404)
- `PATCH /api/practice-records/{record_id}/resolve` (new, 60/minute)
  - 400 for bad UUID, 404 for missing row, 403 for wrong owner,
    204 on success (idempotent — already-resolved still returns 204)

### `frontend/app/index.html`
- Deleted `LAST_FEEDBACK_STORAGE_KEY`, `readLastFeedbackHook`,
  `saveLastFeedbackHook`, `summarizeCoachResponse` — all dead after
  the migration
- Removed the `saveLastFeedbackHook(data)` call inside `renderFeedback()`
- Added hard-coded `WEAKNESS_LABELS` map (`weak_vocab` / `safe_answer` /
  `lack_detail` / `grammar_minor` / `off_topic` → Chinese label)
- Added module-scoped `let pendingResume = null` — not on `window`,
  not in storage
- `showPracticeHub()` is now async: renders Scenario B first for instant
  paint, then `await loadPendingResume()` and re-renders
- `loadPendingResume()` fetches `/api/practice-records/last-unresolved`
  with the Supabase access token; any failure (missing session,
  non-OK response, network error) silently leaves `pendingResume = null`
  so the hub collapses to Scenario B instead of looking broken
- `updatePracticeHubState()` now derives scenario A/B from
  `pendingResume` + `WEAKNESS_LABELS[tag]`. Unknown tags defensively
  fall through to Scenario B (never renders a raw tag string)
- "繼續修這個問題" click handler: if a valid `pendingResume` exists,
  set `currentQuestion` / `currentTopic`, show the speaking UI, and
  set recorder hint to `"先把上一題講順"`. `coach_response` is
  deliberately not displayed anywhere in v1.

## Behaviour changes the user will notice

- Record on phone → log in on laptop → hub already knows where you left off
- Clear cache or switch browser → resume state survives
- Retry the same question with a different weakness → hub clears on next load
- Move to a different question without retrying → hub keeps showing the old one
- Fresh account → hub shows Scenario B, no blank flashes, no errors

## Not touched (per spec)

- Whisper / Groq call params, prompt contents, Soft Lock Next logic
- Admin endpoints, admin UI
- DB schema beyond `resolved` column + its index
- No analytics / tracking added
