-- ============================================================================
-- 20260630033125_add_retry_of_to_practice_records.sql
--
-- ALREADY EXECUTED IN PRODUCTION — DO NOT RUN AGAINST PRODUCTION.
--
-- Reconstructed from supabase_migrations.schema_migrations.statements on
-- 2026-07-25. In the production ledger (version 20260630033125) but had no file in this
-- repo. Recorded verbatim so ledger and repo agree and a clean environment
-- can replay the directory.
-- ============================================================================

-- NOTE: the original ledger statement is `ADD COLUMN retry_of ...` with no
-- IF NOT EXISTS. Guarded here so this file is replay-safe after the baseline
-- (which already carries the column). The FK below is the source of
-- practice_records_retry_of_fkey — see the delivery notes on retry_of.
ALTER TABLE public.practice_records
  ADD COLUMN IF NOT EXISTS retry_of uuid REFERENCES public.practice_records(id);

CREATE INDEX IF NOT EXISTS idx_practice_records_retry_of
  ON public.practice_records(retry_of)
  WHERE retry_of IS NOT NULL;

COMMENT ON COLUMN public.practice_records.retry_of IS
  'Speaking 修復閉環:指向被重講的原始 record。NULL = 首次作答;非 NULL = 此筆為某筆的 retry。前後對比與 Pro 跨 session 記憶依賴此鏈路。';
