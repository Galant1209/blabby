# RLS Migration Baseline — 2026-07-14

Migration 前的完整快照。這是 `20260714_p1_rls_and_reading_answers.sql` 的
**唯一 rollback 依據**。migration 尚未執行；此文件記錄執行前的狀態。

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

## 2. RLS status（migration 會碰到的表，執行前）

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
