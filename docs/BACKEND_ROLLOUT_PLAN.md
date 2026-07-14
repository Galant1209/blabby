# Backend P1 Security — Rollout Plan

分支：`release/backend-p1-security`（= `origin/main` + 封存的 9 個 backend commit）
狀態：草案已備妥並 push。**migration 未執行、未 merge main。** 等 Galant 決定順序。

## ⚠️ 前提修正（重要）

原任務假設「`3a265f9` 的前端部分已 merge 進 main（透過 XSS hotfix）」。
**這不成立**：`origin/main` 仍是 `071695b`，XSS hotfix（`bfbbd68`）從未 push main。

因此本分支用 **merge**（非 cherry-pick）把 `security/backend-blocked-pending-staging`
(80b46d3) 整批 fast-forward 進來，**含 `3a265f9` 全部（前端 + 後端）**。後果：

- 本分支帶有 `3a265f9` 的前端 XSS renderer 硬化（index.html / writing.html）。
- 本分支**不含** hotfix 的 bullets escape（`bfbbd68`，只在 hotfix 分支上）。
- 若 hotfix 與本分支**都** merge 進 main → 前端檔案會衝突。

### 建議的 main 收斂路徑（二選一，避免雙重套用前端）

- **路徑 A（推薦，單一分支上 main）**：把 `release/backend-p1-security`(80b46d3)
  merge 進 main（前端 XSS + 全後端一次到位），再 cherry-pick `bfbbd68`
  （bullets escape，index.html 單行，乾淨）。**不要**再 merge hotfix 分支。
- **路徑 B**：先 merge hotfix（前端含 bullets escape）進 main，本分支改成
  **backend-only**（需拆 `3a265f9`、重建分支，且 `80b46d3` 的 writing.html 建在
  `3a265f9` 之上、拆分易衝突）。較繁瑣，不推薦。

**這一步需 Galant 拍板。** 本輪不動 main。

## Render 部署行為（已查證，當事實）

- Auto-Deploy: On Commit，branch = `main`。**merge 進 main 的當下 backend 就重啟。**
- Migration 是**手動**執行（Supabase），不會隨 deploy 自動跑。
- 1 worker / 1 instance / Starter 0.5CPU-512MB / Autoscaling OFF。
  → `_part2_active_users` 的 process-local lock 有效，不需 distributed lease。

## 上線順序建議：**方案乙（Backend 先）**

```
1. merge release 分支 → main（依上方路徑 A）→ Render auto-deploy
2. 確認 backend 起得來：env isolation guard 通過（已用 Render 正式值驗證會 PASS）
   確認 /healthz 或首頁 200、log 無 RuntimeError
3. 執行 RLS migration（production DB，手動）
4. 對照 docs/RLS_MIGRATION_BASELINE_20260714.md：
   - row count 8 張表完全不變
   - reading_questions RLS 已開、answer 欄位 anon/authenticated 讀不到
   - Reading 功能仍正常（backend service_role 不受影響）
```

**理由**：
- guard 若掛掉 → backend 起不來。方案乙讓我們**先確認 backend 活著**，
  此時 DB 還沒被改，風險最小、回滾面最小。
- 方案乙的過渡風險是「新 backend 短暫面對舊 RLS（答案仍公開）」——但這就是
  **現狀**，沒有變更糟；而方案甲的過渡風險是「舊 backend 面對新 RLS」，雖然
  舊 code 走 service_role 應不受影響，但那是「應該」，不如方案乙的風險已知且為現狀。
- migration 本身 forward-only / non-destructive / idempotent，最壞情況（policy 寫錯）
  用 baseline 文件第 5 節的 rollback 指令 drop policy 還原，資料不動。

## 驗證結果（release 分支上，本輪已跑）

- backend test suite：**99 passed / 10 skipped / 0 failed**（skip 全為需 live 憑證的 e2e）
- `py_compile`：全 backend OK
- `80b46d3` 前後端欄位一致性：`chart_data` = {chart_type,title,labels,values,unit}，
  frontend `validatePieChartData` allowed set 一字不差，雙方都拒絕多餘欄位
- `06f87c1` guard：用 Render 正式值（APP_ENV=production /
  EXPECTED_SUPABASE_PROJECT_REF=mkwywkwruyqzdhuzwnoa /
  SUPABASE_URL=…mkwywkwruyqzdhuzwnoa.supabase.co）**實跑通過**，錯 project / 缺 expected /
  APP_ENV 錯皆正確 fail-closed
- `defusedxml==0.7.1`：已 pin，Render `pip install -r requirements.txt` 會裝上
