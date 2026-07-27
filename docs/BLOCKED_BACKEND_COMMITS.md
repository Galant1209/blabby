# 封存的後端 Commits（等待 staging 驗證）

> **STATUS（2026-07-26 更正）：全數解除封存，本文件已成為歷史記錄。**
>
> 下表 9 筆全部已在 `main` 上並部署至 production。逐項查證：
>
> | 封存項 | 現況 |
> |---|---|
> | `3a265f9` SVG sanitizer | `backend/requirements.txt` 已釘 `defusedxml==0.7.1`；`_validate_chart_svg` 在 `main.py` |
> | `89f6b3c` / `39b7c08` 上傳邊界 | ASGI pre-parser 在 `main.py`，21 個 upload security 測試在 CI 內跑 |
> | `911c7a8` RLS migration | 已於 production 執行 —— ledger `20260714054716`（2026-07-14 13:47:16 +08） |
> | `06f87c1` 環境隔離 guard | `EXPECTED_SUPABASE_PROJECT_REF` 在 `main.py` |
> | `80b46d3` 確定性 pie | `PieChartData` 契約在 `main.py`，前後端已同步部署 |
> | `4998196` / `4f59723` 測試 | `backend/tests/` 在 `main`，CI 每次 push 都跑 |
> | `d2b862a` staging guide | `docs/STAGING_BOOTSTRAP.md` 在 `main`（本身標為 planned-only） |
>
> **解鎖條件實際上並未按本文件設想的方式滿足**：文中每一項都以「Render/Supabase
> staging 驗證後放行」為前提，而 staging 環境**從未建立**（見
> `docs/STAGING_BOOTSTRAP.md` 的 planned-only 標註）。實際放行途徑是直接上
> production 並以 row-count 覆核與 health check 驗證。記錄此事是為了讓後人知道
> 這份文件描述的流程沒有被走完，不要把它當成 staging 流程曾經運作的證據。
>
> 保留原因：這是 2026-07-14 那次拆包決策的唯一書面依據，說明了哪些 commit 為何
> 不能單獨上線。原文以下逐字保留，未作任何修改。
>
> 註：`security/backend-blocked-pending-staging` 只存在於本機，從未 push 到
> origin，所以「封存標記分支」在遠端是找不到的。

建立日期：2026-07-14
封存標記分支：`security/backend-blocked-pending-staging`（指向 `80b46d3`）

## 背景

2026-07-14 稽核判定 production 前端仍在跑含 DOM XSS 的舊版。純前端安全修復已抽出至
`security/frontend-xss-hotfix` 分支（Vercel preview 為其驗證環境），可獨立放行。

以下後端改動**不能單獨上線**：或依賴尚未存在的 staging 基礎設施，或改動 API contract
需與前端同步部署，或涉及 Supabase migration。全部封存於此，等待解鎖條件滿足。

## 封存清單

| Commit | 標題 | Blocker | 解鎖條件 |
|---|---|---|---|
| `3a265f9`（後端部分） | harden renderers against DOM XSS | 新依賴 `defusedxml==0.7.1` + 新增 `sanitize_chart_svg()` 伺服器端 SVG 白名單 | Render staging 部署後驗證 sanitizer 對現有題庫不誤殺 |
| `89f6b3c` | bound and validate Part 2 audio uploads | 音檔上傳大小/驗證限制含 process-local lock，行為隨 Render worker/instance 數而變 | 確認 Render worker/instance topology |
| `911c7a8` | forward-only RLS and Reading answer migration | Supabase migration（`supabase/migrations/20260714_p1_rls_and_reading_answers.sql`） | Supabase staging 上跑 permission matrix 驗證 RLS 不擋正常讀寫 |
| `39b7c08` | reject oversized multipart before parsing | ASGI pre-parser，與 89f6b3c 同屬上傳路徑 | 同 `89f6b3c`（Render topology 確認） |
| `06f87c1` | enforce Supabase environment isolation | 啟動時檢查 Supabase env 隔離，會依環境變數 fail-fast | 確認 Render 各環境 env vars 設定正確 |
| `80b46d3` | render pie charts deterministically | **前後端硬相依**：後端 pie 題改回傳 `chart_svg=None` 並新增 `chart_data` 欄位；前端 pie 只走 `renderPieChart(data.chart_data)` | 必須與後端同步部署，否則 pie 題無圖 |
| `4998196` | backend suite collect without Supabase creds | 測試基礎設施 | 隨對應 fix 一起放行 |
| `4f59723` | P1 regression and permission matrix coverage | 測試（`backend/tests/**`，含 RLS/upload/svg 覆蓋） | 隨對應 fix 一起放行 |
| `d2b862a` | staging environment bootstrap guide | 純 docs，無部署風險 | 隨 backend 堆一起，無獨立 blocker |

## 相依鏈備註

- `4f59723` 的測試覆蓋 `89f6b3c` / `911c7a8` / `39b7c08` / `3a265f9` 後端行為，需與這些 fix 同批。
- `4998196` 是讓 backend 測試在無本地 Supabase 憑證時能 collect 的前置，應排在其他測試 commit 之前。
- `80b46d3` 是唯一「前端也被封存」的 commit：其前端 pie renderer 抽出來也無法單獨運作，
  因為現行 production 後端不回傳 `chart_data`。整包留待與後端同步部署。

## 已放行（對照）

`security/frontend-xss-hotfix`（HEAD `68756f4`，基於 `origin/main`）只含 `3a265f9`
的純前端 hunks（`frontend/app/index.html` + `frontend/app/writing.html`），不含任何後端
依賴。搭配現行 production 後端相容（見放行驗證報告）。
