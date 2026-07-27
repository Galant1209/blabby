-- ============================================================================
-- 20260618045448_writing_questions_add_svg_pregen.sql
--
-- ALREADY EXECUTED IN PRODUCTION — DO NOT RUN AGAINST PRODUCTION.
--
-- Reconstructed from supabase_migrations.schema_migrations.statements on
-- 2026-07-25. In the production ledger (version 20260618045448) but had no file in this
-- repo. Recorded verbatim so ledger and repo agree and a clean environment
-- can replay the directory.
-- ============================================================================

ALTER TABLE writing_questions
  ADD COLUMN IF NOT EXISTS chart_svg text,
  ADD COLUMN IF NOT EXISTS is_pregenerated boolean DEFAULT false,
  ADD COLUMN IF NOT EXISTS used_count integer DEFAULT 0;

CREATE INDEX IF NOT EXISTS idx_writing_questions_pregenerated
  ON writing_questions(task_type, task1_subtype, is_pregenerated);
