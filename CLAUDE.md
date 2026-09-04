# CLAUDE.md — Blabby 執行環境事實

依 2026-09-04 勘查報告（`claude/CLAUDEMD_SURVEY_2026-09-04.md`）重寫。

---

## 0. 這份檔案怎麼用

這是 Claude Code 唯一自動載入的檔案，所以它的錯誤會被當成事實執行。

每一條都應該可以用一個指令驗證。寫不出驗證方式的，不要寫進來。

發現任何一條與實況不符時**停下回報**，不要自行調整程式碼去迎合這份文件，
也不要自行改這份文件。

---

## 1. Blabby 是什麼

IELTS 練習平台。五個練習模組 —— Speaking、Reading、Writing、Vocabulary、Admin ——
外加 ECPay 金流與練習前的入室契約（covenant）。

不是只有口說。舊版 CLAUDE.md 寫「Blabby = AI IELTS 口說訓練平台」，那是錯的。

---

## 2. 架構事實

### 後端

`backend/main.py`，單檔，11826 行，FastAPI，71 條路由。

其餘後端檔：`ecpay.py`、`reading_prompts.py`、`reading_validator.py`。

### 前端：15 支 html + 4 支獨立 .js

**「單檔前端」是錯的描述。** 舊版 CLAUDE.md 寫
「`frontend/app/index.html` / 所有 JS inline / 單檔架構 / 不要拆 module」。
`ls frontend/app/` 一次就會露餡。

這條錯誤描述已經污染過真實決策 —— 詳見 §4 的 Pro 判定那節。

| 檔案 | 服務的模組 |
|---|---|
| `index.html` | Speaking Part 1 + Part 2、入室契約、匿名試用、progress、部分 vocabulary |
| `reading.html` | Reading 全部 |
| `writing.html` | Writing 全部 |
| `vocabulary.html` | Vocabulary 主頁 |
| `admin.html` | Admin 全部 |
| `hub.html` | 練習入口／儀表：下一題、弱點摘要、今日複習、訂閱狀態 |
| `account.html` | 帳號頁：history、訂閱、診斷時間軸、配額 |
| `history.html` | 練習歷史 |
| `progress.html` | 進度與 evidence |
| `diagnosis.html` | AI 診斷 |
| `upgrade.html` | 付費頁：`create-order` + PostHog 漏斗 |
| `success.html` | 付款成功頁：訂閱狀態 |
| `debug_sse.html` | `/debug/sse_test` |
| `privacy.html` | 靜態；端點反查零命中 |
| `terms.html` | 靜態；端點反查零命中 |

四支獨立 `.js`：

| 檔案 | 大小 | 內容 |
|---|---|---|
| `config.js` | 349 B | `supabaseUrl` / `supabaseAnonKey` / `publicReviewMode`。已進版控，anon key 為公開金鑰 |
| `anonymous-conversion.js` | 2376 B | **待查** —— 內容未逐支讀，需要時自行 grep |
| `progress-evidence.js` | 2259 B | **待查** —— 內容未逐支讀，需要時自行 grep |
| `retention.js` | 15183 B | **待查** —— 內容未逐支讀，需要時自行 grep |

後三支的檔名與 `backend/tests/` 裡三支孤兒 `.mjs` harness 同名。

### DB

Supabase。public schema 33 張表。**這個 project 是共用的**，見 §7。

### LLM / 語音

`run_claude()`（`backend/main.py`，搜函式名）。
provider client 在 **import 時**建構，不是 lazily —— 缺 key 會讓整個模組 collect 失敗。
相關環境變數：`ANTHROPIC_API_KEY`、`OPENAI_API_KEY`、`GROQ_API_KEY`、`GOOGLE_TTS_API_KEY`。

### 部署

後端 **Render**：auto-deploy on commit to `main`，Starter plan，無 idle sleep。

前端 **Vercel**：dashboard-only 設定，**無 `vercel.json`**，root 為 `frontend/app`。
檔案服務在根 URL —— `admin.html` → `/admin.html`。

### 分析

PostHog，`api_host: https://us.i.posthog.com`。
`index.html` 與 `upgrade.html` 各自初始化一份 `window.analytics` shim（`posthog.capture`）。

---

## 3. 模組全景

### Speaking

- 端點：`POST /process`、`GET /part2/topics`、`POST /part2/evaluate`、
  `GET /api/questions/bank`、`GET /api/questions/next`、`GET /api/drill/check_quota`
- 前端：`index.html`、`hub.html`
- Free/Pro 分界：後端 `FREE_DRILL_QUOTA`（20 次／7 天滾動，常數
  `DRILL_QUOTA_WINDOW_DAYS`）與 `FREE_PART2_MONTHLY_QUOTA`（10 次／月，
  gate 在 `_enforce_part2_quota()`）。前端搜 `drillQuotaState`
- 孤兒端點：無

**負向事實 —— 這兩條不寫會被預設存在：**

- **Part 3 零實作。** 後端、`index.html`、`hub.html` 三處搜 `part3` 皆零命中。
  「Part 1 / Part 2 / Part 3」是 IELTS 的標準結構，不明寫會讓人以為它在
- **Part 2 無 1 分鐘準備倒數。** `p2ShowOnly()` 只有 `p2-card` / `p2-record` /
  `p2-score` 三態，搜 `p2-prep` 零命中。筆記欄在 card 上但不計時，使用者可以
  無限時間準備。這件事在 2026-07-25 被列為「CLAUDE.md 宣稱已實作但不存在」，
  不要再製造同一個誤會

### Reading

- 端點：`GET /reading/quota`、`POST /reading/attempt/abandon`、
  `POST /reading/passage/generate`、`POST /reading/passage/generate_stream`、
  `POST /reading/questions/generate`、`GET /api/reading/passage`（pool-first）、
  `POST /reading/attempt/start`、`POST /reading/attempt/submit`、
  `GET /reading/attempt/{attempt_id}`、`GET /reading/history`、
  `POST /vocab/lookup`、`POST /vocab/translate_zh`、`POST /api/vocabulary/save_word`
- 前端：`reading.html`（`admin.html` 另讀 attempt 詳情）
- Free/Pro 分界：後端 `FREE_READING_DAILY_QUOTA`（1 篇／日），判準
  `get_user_pro_status()`。前端搜 `body.is_pro`
- 孤兒端點：`GET /reading/history` —— 交付失效，見 §5

### Writing

- 端點：`GET /api/writing/question`、`POST /api/writing/submit`、
  `GET /api/writing/history`、`GET /api/writing/submission/{submission_id}`、
  `POST /api/admin/writing/pregen`、`GET /admin/writing/submissions`
- 前端：`writing.html`（`admin.html` 讀 submissions）
- Free/Pro 分界：後端 `is_user_pro()`，出題與交卷各擋一次 Task 1。
  前端搜 `isProUser`
- Task 1 子題型：`TASK1_SERVED_SUBTYPES`（`bar_chart` / `line_graph` /
  `pie_chart` / `table`）。前端以註解手動同步，**無機械保證**
- 孤兒端點：`GET /api/writing/history` —— 交付失效，見 §5

### Vocabulary

- 端點：`GET /api/vocabulary/active-use/current`、`GET /api/vocabulary/items`、
  `POST /api/vocabulary/my`、`GET /api/vocabulary/my`、
  `GET /api/vocabulary/review/today`、`POST /api/vocabulary/review`、
  `POST /api/vocabulary/generate`、`POST /api/vocabulary/save_word`
- 前端：`vocabulary.html`、`index.html`、`hub.html`、`reading.html`
- **Free/Pro 分界只在後端，前端不處理。** gate 在 `POST /api/vocabulary/my`
  （搜 `vocab_limit_reached`）：判準 `get_user_pro_status()`，free 上限為
  `user_vocabulary` 總筆數 **30**。冪等重加不吃配額。
  上限是**硬編碼字面量**，沒有具名常數 —— 所以 `FREE_VOCAB*` grep 不到。
  `vocabulary.html` 搜 `vocab_limit_reached` 與 `upgrade` 皆零命中，
  前端不處理這個 403，也沒有付費牆 UI。列入 §5
- 孤兒端點：無

### Admin

- 端點 17 條：`/admin/recent`、`/admin/pro_breakdown`、`/admin/user/{id}/pro`（deprecated）、
  `/admin/user/{id}/pro_grant`、`DELETE /admin/user/{id}`、`/admin/waitlist`、
  `/admin/dashboard`、`/admin/activity`、`/admin/users`、`/api/admin/reclassify`、
  `/api/admin/student_brief/{id}`、`/admin/user/{id}`、`/admin/writing/submissions`、
  `/admin/reading/attempts`、`/admin/practice-volume`、`/admin/user/{id}/diagnosis`、
  `/api/admin/subscriptions`（+ extend / cancel）
- 前端：`admin.html`
- 授權分界：`verify_admin()`，email 白名單常數 `ADMIN_EMAILS`
- 孤兒端點 4 條：`/admin/pro_breakdown`、`/admin/waitlist`、`/admin/dashboard`、
  `/admin/activity` —— 半成品殘骸，見 §5

### 金流（ECPay）

- 端點：`POST /api/payment/create-order`、`GET /api/payment/_diag`、
  `POST /api/payment/callback`、`GET /api/user/subscription`、
  `POST /api/webhooks/lemonsqueezy`（LemonSqueezy 遺留）、
  `POST /api/track/upgrade_page_view`、`POST /api/track/upgrade_interest`
- 前端：`upgrade.html`（唯一呼叫 `create-order`）、`success.html`、
  `account.html`、`hub.html`、`writing.html`（讀訂閱狀態）
- 設定：`ecpay.load_config()` 於 startup；失敗記 CRITICAL 但不阻斷服務
- 背景 job：`expire_stale_pending_subscriptions`（每 6h :00）、
  `lapse_expired_active_subscriptions`（每 6h :30，刻意錯開）
- 孤兒端點 2 條：`/api/track/upgrade_page_view`、`/api/track/upgrade_interest`
  —— 半成品殘骸，見 §5

### Auth / Profile / 入室契約

- 端點：`GET /api/covenant/status`、`POST /api/covenant/sign`、
  `GET /api/anonymous-trial/status`、`GET /api/history`、`GET /api/progress`、
  `GET /api/progress/evidence`、`GET /api/diagnosis/timeline`
- 前端：契約 overlay 在 `index.html`（搜 `covenant-overlay`）
- 認證：`verify_token()`；admin 另過 `verify_admin()`
- 匿名試用：`ANONYMOUS_PROCESS_LIMIT`（10）、`ANONYMOUS_PROCESS_RATE_LIMIT`（3）
- 孤兒端點：無

---

## 4. 已建立的不變式 — 不要重新發明

以下都是已經修好、且有測試或實查佐證的。動它們之前先讀這一節。

### Reading 答案欄位

作答期間可見的欄位由 `READING_QUESTION_CLIENT_FIELDS` 定義（allowlist，非 denylist），
統一經 `_client_reading_questions()` 投影。`correct_answer` / `explanation` /
`evidence_quote` 刻意不在其中。

**新增任何出題路徑必須走 `_client_reading_questions()`。** 直接把 DB row 塞進 response
會繞過唯一的 gate —— 後端以 service_role 讀取，PostgREST 的欄位授權保護不到這條路徑。

答案欄位只在兩處進 response：
`POST /reading/attempt/submit`（交卷即揭曉），以及
`GET /reading/attempt/{attempt_id}`，後者 gate 為
`user_id` 相符 **且** `status == 'submitted'`，否則 404。

迴歸測試：`test_reading_questions_no_answer_leak.py`、`test_reading_e2e.py`、
`test_reading_pool_serving.py`、`test_content_lockdown_contract.py`、
`test_supabase_p1_permissions.py`（後者以真實角色驗 anon / 他人被拒、service_role 可讀）。

### RLS 與內容表

33 張表 `pg_tables.rowsecurity` 全為 **true**，無一例外。

內容表逐張：

| 表 | 機制 |
|---|---|
| `questions` | 零 policy，fail-closed |
| `reading_passages` | 零 policy，fail-closed |
| `writing_questions` | 零 policy，fail-closed |
| `reading_questions` | 兩層：row 層 policy `reading_questions_prompt_select`（`USING (true)`）+ column 層只授 7 個安全欄位給 `authenticated` |
| `vocabulary_items` | policy `Anyone can read vocabulary_items`，`roles={public}`、`qual=true`，anon 可讀全部 15 欄 |

前三張零 policy 是**刻意**，所有讀取經 FastAPI + service_role。
查詢回空陣列的原因是繞過了 FastAPI —— 修呼叫端，不加 policy。

`reading_questions` 的兩層是 RLS 單獨做不到的（RLS 過濾 row，column 授權是另一維度），
三個答案欄位從未授予任何人。它與 `_client_reading_questions()` 互為備援，不是重複。

anon/client 直接讀受保護答案欄位時，DB 必須拒絕而不是回傳可用資料；PostgreSQL
permission denial 的 SQLSTATE 是 `42501`（`insufficient_privilege`）。
`test_content_lockdown_contract.py` 與 replay harness 以這個拒絕結果驗證 fail-closed，
不得刪除或弱化這條 contract。

`vocabulary_items` **未決** —— 刻意公開還是遺漏。
收掉前必須先查前端有無 anon 直讀，猜錯會弄壞線上功能。
無答案類欄位，風險是題庫被整包抓走，不是答案外洩。

**Catalog 是索引，不是證據。** `has_table_privilege()` 對有欄位級授權的表回 `false`，
與「完全沒有授權」的答案一模一樣。2026-07-30 有一次稽核據此誤判為答案外洩，
並擬出會刪掉一層防護的「修補」方案。判定必須用實際角色嘗試讀取。

### auth.users 曝露

`user_lookup` view **已從 production 移除**。

`user_pro_status` 是 `{security_invoker=on}`，定義只讀 `profiles`、不碰 `auth.users`，
授權清單為 `postgres` / `authenticated` / `service_role` —— **`anon` 不在內**。

### Part 2

`validate_part2_response()` 在持久化與回傳**之前**呼叫。失敗走
`scoring_failed` fallback：只回 transcript，不寫分數，不渲染分數卡。

月配額用 `.eq("mode", "part2")` 與 `/process` 側的 `.neq("mode", "part2")`
對稱切分，兩個功能不會互吃配額。

配額 gate **fail-closed**：count 查詢失敗 raise 503，不是放行。
Part 2 是產品裡最貴的呼叫，查不到配額時的正確行為是拒絕。

### Pro 判定 — 兩類，不是「四份複製」

權威在後端。`is_user_pro()` 是 Postgres function，OR 兩條路徑：

- admin 授予：`profiles.is_pro_grant` 且（`pro_grant_expires_at IS NULL` 或未過期）
- 付費：`subscriptions.status = 'active'` **AND** `expires_at > now()`

付費那條是**雙條件時間窗，沒有裸 boolean**。
後端兩支同義包裝：`get_user_pro_status()`（同步）與 `is_user_pro()`（async），
都 fail-safe 回 `False`。

前端有兩類實作，處置完全不同：

**A. 客戶端日期運算（三份，判準被複製到瀏覽器）**

| 檔案 | 實作 | 形狀 |
|---|---|---|
| `success.html` | `isActiveSubscription()` | boolean |
| `writing.html` | `isActiveSubscription()`，逐字複製自 `success.html`，註解自承 | boolean |
| `hub.html` | `resolveIsPro()`，同判準但 inline | tri-state true/false/**null** |

`writing.html` 那份的註解是這個坑的現場紀錄：

> "Copied verbatim from success.html's isActiveSubscription()... this page was
> the one that got it wrong: **status alone was trusted, so a subscription that
> had run out still unlocked Task 1**."

**B. 伺服端權威（一份，判準留在後端）**

`account.html` 讀後端回傳的 `proBody.is_pro`。該值由 `is_user_pro()` 算出，
時間窗已在 server 判完。

**它不做日期比較是正確的，不是漏掉。
不要為了「統一」把 `account.html` 改成客戶端日期運算。**
方向若要收斂，是往伺服端權威那邊走，不是反過來。

（既有的三份 A 類不主動合併，見 §6。）

### 不做自動續訂

Pro 一律為使用者主動購買的 30 天期。不實作定期定額、不預留 recurring 欄位。
`is_user_pro()` 的 live 定義就是這個決定的實體形式。

### Phase 1 不整合電子發票

不加 `InvoiceMark` 或任何發票參數 —— 那會改動已通過 known-answer 驗證的
CheckMacValue 參數集。未來整合時必須重跑 `test_ecpay.py` 與
`test_ecpay_known_answer.py` 的兩組官方向量。

---

## 5. 已知的漂移與債

照實寫，不修飾。以下每一條都是本輪查到的，不是推測。

### 端點的三類，處置完全不同

**正常的零命中 —— 不是孤兒**

`/api/webhooks/lemonsqueezy`、`/api/payment/callback`、`/api/payment/_diag`、
`/api/admin/writing/pregen`。外部呼叫或手動觸發，本來就不該有前端 caller。

**交付失效 —— 後端做完、使用者拿不到**

`GET /api/writing/history`、`GET /reading/history`。
兩支同型，代表不是單次疏漏。**已排定，下一輪處理。**

**半成品殘骸 —— 三處互相錯開**

- `/admin/pro_breakdown`、`/admin/waitlist`、`/admin/dashboard`、`/admin/activity`
- `/api/track/upgrade_page_view`、`/api/track/upgrade_interest`
- `pro_waitlist` 表：0 列、零 policy、後端零寫入者
- `admin.html`（搜 `pro_waitlist`）用 anon key 直查那張零 policy 的表，
  **永遠回空陣列**
- 後端 `/admin/waitlist` 讀的其實是 `upgrade_intent`，不是 `pro_waitlist`，
  而前端不呼叫它 —— 兩邊讀不同的表，都讀不到
- `POST /api/track/upgrade_interest` 的 docstring 自承
  「Logs only — no persistence yet」，實作只有一行 `logger.info`，
  **且該行把 email 寫進 log**
- `upgrade_intent` 有 6 列，全部來自 2026-04-30，之後零新增
- `upgrade_intent` 掛著三條重複的 INSERT policy（`allow_anon_insert`、
  `anon_insert_upgrade_intent`、`authenticated_insert_upgrade_intent`），
  `with_check` 全為 `true`，**anon 可無限制寫入**

**等候名單是金流上線前的替代品。ECPay 已經活著、`upgrade.html` 走
`create-order`，這條路徑已無存在理由。處置方向是清除而非修復，獨立一輪。**

### Vocabulary 的 gate 只有後端

後端擋（`vocab_limit_reached`，30 筆上限），前端不接。
`vocabulary.html` 沒有付費牆 UI，也不處理那個 403 ——
free 使用者存到第 31 個字時會撞到一個前端沒準備接的錯誤。

上限 30 是硬編碼字面量，不是具名常數。

2026-07-25 的紀錄顯示這個 gate 當時在 `main.py:4228-4238`，位置已變、gate 還在。
搬遷的時點與原因**待查**。

### schema 有兩套事實

repo `supabase/migrations/` **34 支 `.sql`**（29 前進 + 5 rollback）。
production ledger `supabase_migrations.schema_migrations` **12 筆**。

兩筆 Blabby migration 的變更在 production 生效，repo 無對應檔案或無 ledger 記錄：

- `20260803111937_reading_pool_columns_and_backfill` — 在 ledger，repo 無檔
- `20260813_anonymous_process_quota` — 在 repo，ledger 無記錄，但
  `anonymous_process_usage` 表確實存在（代表被直接執行）

14 支 `202604`–`202605` 的舊檔早於 ledger 起點（`20260617`）。
ledger 裡的 `create_npc_relations` / `harden_npc_relations` 屬其他專案，見 §7。

修補 schema 漂移是獨立一輪。

### 其他

- `rec_log` 表與 `POST /api/debug/rec-log`：表註解自承為 iOS 診斷用臨時觀測，
  「確認 iOS 修復後連同 `/api/debug/rec-log` 一起移除」。是否可移除未判定
- `PATCH /admin/user/{user_id}/pro` 標記 `deprecated=True`，
  但 `admin.html` 與 `index.html` 仍有命中，是否仍在呼叫路徑上未確認

---

## 6. 紅線（不可跨）

- 不新增 streak / XP / level / gamification / 勵志型推播
- 不重構 `backend/main.py`
- 不為了抽象而抽象。`frontend/app/*.js` 是既有的共用模組位置，
  新的共用邏輯可以放那裡；但既有的重複實作不主動合併
  （舊版寫「單檔架構、不要拆 module」是錯的，且已造成過實際損害 ——
  見 §4 Pro 判定節）
- 不做視覺造假 —— 首頁 demo 的每個元素都必須存在於真實介面。
  省略允許，新增不允許
- 不動穩定的 Part 1，除非有具體 regression
- 不做大型 UI redesign
- 不做多期 pie renderer（11 題已軟退池，可逆、有文件：
  `docs/PIE_RETIRED_MULTIPERIOD_20260714.md`）
- 不補 Admin 批改佇列 —— 後端交卷即批改完，無 pending 狀態可掛，
  需先改 schema
- 面向使用者文字一律古老英式風格 —— 每一句都要值得被印在羊皮紙上
- 法務文字例外：條款內文要冷、清楚、可執行。
  「銀行退刷入帳約需 14–21 個工作天」不要改成文言。
  標題典雅、內文冷硬，兩種語域不混在同一段落
- Claude Code **永不 `git push`** —— Galant 自己執行
- 不會咬的測試等於沒有測試 —— 新增斷言要驗證它真的會紅

本輪掃描確認遊戲化紅線未被違反：`streak` / `\bXP\b` / `level-up` / `gamif`
全部零命中。`achievement` 的兩處命中是 IELTS 官方評分項 Task Achievement，
`badge` 的命中全是 UI 狀態標記（`pro-badge`、`upgrade-badge`…）。

決策規則：所有實作至少必須改善 conversion / retention / perceived progress
其中一項。三者都沒有就不要做。

文案調性 —— 物理治療師：描述可觀察的行為、可量化的弱點、修正路徑。
不要情緒性鼓勵、泛用稱讚、教練式說話。
（**這份 CLAUDE.md 本身不適用古英式。** 它面向 agent，要準確不要典雅。）

---

## 7. 環境陷阱

**`~/Desktop` 由 macOS iCloud Desktop & Documents / CloudDocs 管理，不是可靠的
canonical repo 路徑。** 問題不一定表現為 iCloud 的 `.icloud` 缺檔 placeholder；一般程式碼
檔案可能完全正常，但 Git 高頻改寫的 `.git/index`、`packed-refs`、reflog 可能在
同步協調期間 timeout、讀回 0 bytes，或造成 metadata corruption / unusable repo state。
跨 provider rename 也可能回 `ETIMEDOUT`。

Blabby canonical repo 固定在 `~/dev/Blabby/blabby`（`/Users/yichengchiu/dev/Blabby/blabby`）。
舊 `~/Desktop/Blabby` 只能當 read-only rescue source；CloudDocs-managed repo 禁止執行
`git gc` 或 `git repack`。

**git 指令一律加 `--no-pager`。** 本 repo 的 pager 會 hang。

**`git merge` 一律加 `--no-edit`。** `--no-ff` 會開 vim，在 Claude Code 的環境
會卡死，並留下 `.MERGE_MSG.swp` 要手動 `D` 掉。空的 merge message 會終止 merge；
若不慎中止，用 `git merge --abort` 或 `git commit --no-edit` 收尾，變更不會遺失。

**分類器過載會讓 Bash 與 Supabase MCP 同時不可用。** 發生時改做不需要
工具的部分，並明確註記哪幾項未完成，不要用推測填空。

**Claude Code 的 browser pane 是 `document.hidden === true`。**
`IntersectionObserver` 完全不回呼，`setTimeout` 被節流到約 1 秒/次。
任何涉及 `IntersectionObserver` 或牆鐘時間的驗證都要人工在真實瀏覽器做。

**本機 `http.server` 打 production backend 會被 CORS 擋** ——
`ALLOWED_ORIGINS` 只有 Vercel 域名。登入相關的驗證必須在 preview
或 production 上做。

**adblock 會擋 PostHog 與 `/health`。** 驗 tracking 時要關掉。

**Supabase project 是共用的。** 33 張表裡 13 張不屬於 Blabby。
migration ledger 不只有 Blabby 在寫；看到 `create_npc_relations`
這類項目是正常的。`EXPECTED_SUPABASE_PROJECT_REF` 保護的是 app→DB 方向，
擋不住別的 app 寫進同一個 DB。

---

## 8. 測試

### pytest

入口：`cd backend && pytest tests -q`（與 CI 的 `working-directory: backend` 一致）。

本機 venv 是 Python 3.10；CI 釘 3.11。

`backend/tests/conftest.py` 在 import 前清空 `SUPABASE_URL` /
`SUPABASE_SERVICE_KEY`，並對三個 provider key `setdefault('test-key')` ——
因為 `main.py` 在 import 時建構 client，缺 key 會讓整個模組 collect 失敗。

測試檔 41 支。**2026-09-04 未取得 passed/skipped 數字**：本機檔案系統病態緩慢
（見 §7），多次嘗試皆卡在 import 階段。當輪 diff 不含 Python 改動，故不阻擋。

### `.mjs` harness

8 支，全在 `backend/tests/`。

接進 CI 的 2 支：

```
frontend_billing_behavior.mjs          ← test_frontend_billing_contracts.py
frontend_writing_restore_behavior.mjs  ← test_writing_restore_contracts.py
```

孤兒 6 支（無 pytest 包裝，CI 不跑）：

```
frontend_active_vocabulary_behavior.mjs
frontend_anonymous_conversion_behavior.mjs
frontend_prescription_behavior.mjs
frontend_progress_evidence_behavior.mjs
frontend_resolution_cycle_behavior.mjs
frontend_retention_behavior.mjs
```

接進 CI 是獨立一輪，接進去很可能直接紅。

### Node

三處 `subprocess` 硬呼叫 `node`，無 `shutil.which` 保護、無 skip：
`test_writing_restore_contracts.py`、`test_frontend_billing_contracts.py`、
`test_writing_pie_chart.py`。

**2026-09-03 已拍板：Node 不可用時照既有 harness 的 fail，不改 skip。**
理由是兩支對同一件事有兩種反應，比兩支都紅更難查。
**不要重開這個討論，不要順手加 guard。**

CI 已顯式安裝 Node 20（`actions/setup-node@v4`），就是為了讓這個 fail 不再發生。

### CI

`.github/workflows/` 只有一支 `test.yml`，兩個獨立 job：

- `backend-tests`：checkout → setup-python 3.11 → setup-node 20 →
  `pip install` → `pytest tests -q`
- `migration-replay`：postgres:17 service container → `./supabase/replay/replay.sh`

兩個 job 互不依賴 —— 紅的 replay 不該遮住紅的測試套件，反之亦然。

### 測試最低標準

每個新 endpoint 要有 happy-path test。mock LLM calls。測試時不可真的打 API。
