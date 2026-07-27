-- ============================================================================
-- 20260617120832_create_rec_log_table.sql
--
-- ALREADY EXECUTED IN PRODUCTION — DO NOT RUN AGAINST PRODUCTION.
--
-- Reconstructed from supabase_migrations.schema_migrations.statements on
-- 2026-07-25. This migration is in the production ledger (version 20260617120832) but had
-- no file in this repo. Recorded here verbatim so the ledger and the repo
-- agree and so a clean environment can replay the directory.
-- ============================================================================

create table if not exists public.rec_log (
  id            bigint generated always as identity primary key,
  created_at    timestamptz not null default now(),
  part          smallint,
  mime          text,
  recorder_mime text,
  ext           text,
  size          integer,
  chunk_count   integer,
  forced_mp4    boolean,
  status        text,
  error         text,
  ua            text
);

comment on table public.rec_log is '臨時觀測:前端錄音回報(iOS blob 格式/大小診斷)。確認 iOS 修復後連同 /api/debug/rec-log 一起移除。';

-- 只有 service_role(後端 supabase_admin)能讀寫;不開放任何 anon/authenticated policy。
alter table public.rec_log enable row level security;
