# RLS Migration Baseline — 2026-07-14

Migration 前的完整快照。這是 `20260714_p1_rls_and_reading_answers.sql` 的
**唯一 rollback 依據**。此文件記錄的是 migration **執行前**的狀態。

> 狀態（2026-07-26 更正）：migration **已執行** —— Supabase ledger `20260714054716`
> （2026-07-14 13:47:16 +08），`20260714054836` 為冪等重跑。原文「migration 尚未執行」
> 自該時起即已過時。執行後的實數覆核見下方 §1.1。

Project: `mkwywkwruyqzdhuzwnoa`（blabby, ap-northeast-1, Postgres 17）
擷取時間：2026-07-14（migration 執行前）

## 1. Row counts（重點稽核表）

| 表 | rows |
|---|---|
| subscriptions | 0 |
| diagnosis_cache | 5 |
| reading_passages | 32 |
| reading_questions | 252 |
| reading_attempts | 26 |
| reading_answers | 18 |
| profiles | 17 |
| practice_records | 241 |

migration 後這些 count **必須完全不變**（migration 不含任何 DML）。

### 1.1 實查覆核 — 2026-07-25 20:36 (+08)

方法：對 production project `mkwywkwruyqzdhuzwnoa` 直接執行 `select count(*)`（**非** `pg_class.reltuples`
估計值，**非** Supabase `list_tables` 回報值）。同時取 `pg_stat_user_tables` 的
`n_live_tup` / `n_tup_del` / `last_analyze` / `last_autoanalyze` 作為佐證。全程唯讀，
未執行 `ANALYZE` / `VACUUM`（以免改寫正在調查的 `reltuples` 證據）。

結論：**上表 8 個 count 與 2026-07-25 實數逐項完全一致，無任何資料遺失。**

| 表 | baseline（07-14） | 實查 count（07-25） | `n_live_tup` | `n_tup_del` |
|---|---|---|---|---|
| subscriptions | 0 | 0 | 0 | 0 |
| diagnosis_cache | 5 | 5 | 2 | 0 |
| reading_passages | 32 | 32 | 8 | 0 |
| reading_questions | 252 | 252 | 63 | 0 |
| reading_attempts | 26 | 26 | 6 | 0 |
| reading_answers | 18 | 18 | 18 | 0 |
| profiles | 17 | 17 | 3 | 0 |
| practice_records | 241 | 241 | 241 | 0 |

`n_live_tup` 系統性偏小的原因：cumulative statistics 在 `pg_postmaster_start_time()`
= **2026-06-05 04:11:38 UTC** 的實例重啟時歸零，且這些表此後從未 `ANALYZE`
（`last_analyze` 與 `last_autoanalyze` 皆為 `null`，`reltuples` 為 `-1` 哨兵值），
因此 `n_live_tup` 只累計 06-05 之後的 insert 數，而非真實列數。以建立時間切分可完全驗證：

| 表 | `< 06-05` | `>= 06-05` | 實查 total |
|---|---|---|---|
| profiles | 14 | **3** | 17 |
| reading_passages | 24 | **8** | 32 |
| reading_questions | 189 | **63** | 252 |
| reading_attempts | 20 | **6** | 26 |
| diagnosis_cache | 3 | **2** | 5 |
| reading_answers | 0 | **18** | 18 |

`>= 06-05` 一欄與 `n_live_tup` 完全相同。`reading_answers` 之所以「看起來正確」，
是因為它 18 列全部建立於 06-05 之後；`practice_records` 之所以正確，是因為它在
2026-06-16 11:56 UTC 有一次 `autoanalyze` 把 `reltuples` 校正為 241。

`n_tup_del = 0` 涵蓋 2026-06-05 至今的整個窗口（含 07-14 migration 之後），
因此 migration「不含任何 DML」的斷言亦獲獨立驗證。

## 2. RLS status（migration 會碰到的表，執行前）

> 下表「migration 後預期」一欄寫於執行前，現已是**既成事實**：migration 於
> 2026-07-14 13:47:16 +08 執行完畢，各表最終狀態與該欄一致。

| 表 | rls_enabled（前） | migration 後預期 |
|---|---|---|
| subscriptions | true | true（不變；只新增 policy + 收斂 grant） |
| diagnosis_cache | true | true（新增 4 條 own policy + 收斂 grant） |
| reading_questions | **false** | **true**（本 migration 開啟） |
| reading_passages | false | false（**本 migration 不碰**） |
| reading_answers | true | true（不碰） |
| reading_attempts | true | true（不碰） |

> 注意：`reading_passages`、`questions`、`writing_questions` 的 RLS 仍為 false，
> 本 migration **不處理**它們（只收斂 reading_questions 的答案欄位）。

## 3. 現有 policies（migration 會碰到的表，執行前）

- `subscriptions`：**無 policy**（RLS 開啟但零 policy → authenticated 讀不到，讀取靠 service_role 繞過）
- `diagnosis_cache`：**無 policy**（同上）
- `reading_questions`：**無 policy**（RLS 為 false，anon/authenticated 靠 grant 全表可讀）
- `reading_answers`：`reading_answers_owner`（ALL, public, `EXISTS reading_attempts WHERE a.id=attempt_id AND a.user_id=auth.uid()`）
- `reading_attempts`：`reading_attempts_owner`（ALL, public, `auth.uid()=user_id`）

（全庫其餘 policy 見擷取當日 `pg_policies` dump；與本 migration 無關的表未列。）

## 4. 現有 grants（migration 會碰到的表，執行前）

migration 會 `REVOKE ALL ... FROM anon, authenticated` 再重新 grant，故記錄執行前基準：

| 表 | anon | authenticated | service_role |
|---|---|---|---|
| subscriptions | ALL（DELETE,INSERT,REFERENCES,SELECT,TRIGGER,TRUNCATE,UPDATE） | ALL | ALL |
| diagnosis_cache | ALL | ALL | ALL |
| reading_questions | ALL | ALL | ALL |
| reading_attempts | ALL | ALL | ALL |
| reading_answers | ALL | ALL | ALL |
| reading_passages | ALL | ALL | ALL |

> 這是問題根源：`reading_questions` 目前 anon/authenticated 有 **SELECT 全欄位**
> （含 `correct_answer`/`explanation`/`evidence`），且無 RLS → 答案公開可讀。

## 5. Rollback 指令（若 migration 後要回滾）

migration 是 forward-only / non-destructive / idempotent。回滾方式（不涉及資料）：

```sql
-- reading_questions：關 RLS、還原全 grant
DROP POLICY IF EXISTS reading_questions_prompt_select ON public.reading_questions;
ALTER TABLE public.reading_questions DISABLE ROW LEVEL SECURITY;
GRANT ALL ON TABLE public.reading_questions TO anon, authenticated;

-- diagnosis_cache：移除新 policy、還原全 grant（RLS 本就開著，維持）
DROP POLICY IF EXISTS diagnosis_cache_select_own ON public.diagnosis_cache;
DROP POLICY IF EXISTS diagnosis_cache_insert_own ON public.diagnosis_cache;
DROP POLICY IF EXISTS diagnosis_cache_update_own ON public.diagnosis_cache;
DROP POLICY IF EXISTS diagnosis_cache_delete_own ON public.diagnosis_cache;
GRANT ALL ON TABLE public.diagnosis_cache TO anon, authenticated;

-- subscriptions：移除新 policy、還原全 grant（RLS 維持開）
DROP POLICY IF EXISTS subscriptions_select_own ON public.subscriptions;
GRANT ALL ON TABLE public.subscriptions TO anon, authenticated;
```

> 回滾只動 policy/grant，**不動任何一列資料**。上面第 1 節的 row count
> 是驗證「回滾後資料無損」的對照基準。
