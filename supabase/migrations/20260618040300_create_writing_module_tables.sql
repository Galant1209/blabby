-- ============================================================================
-- 20260618040300_create_writing_module_tables.sql
--
-- ALREADY EXECUTED IN PRODUCTION — DO NOT RUN AGAINST PRODUCTION.
--
-- Reconstructed from supabase_migrations.schema_migrations.statements on
-- 2026-07-25. In the production ledger (version 20260618040300) but had no file in this
-- repo. Recorded verbatim so ledger and repo agree and a clean environment
-- can replay the directory.
-- ============================================================================

-- writing_questions: AI-generated prompts, reusable cache
CREATE TABLE IF NOT EXISTS writing_questions (
  id               uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  task_type        text NOT NULL CHECK (task_type IN ('task1', 'task2')),
  task1_subtype    text CHECK (task1_subtype IN ('bar_chart','line_graph','pie_chart','table','process','map')),
  prompt           text NOT NULL,
  chart_description text,
  essay_type       text,
  created_at       timestamptz DEFAULT now()
);
ALTER TABLE writing_questions DISABLE ROW LEVEL SECURITY;

-- writing_submissions: user essays + AI grading results
CREATE TABLE IF NOT EXISTS writing_submissions (
  id                     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id                uuid NOT NULL REFERENCES auth.users(id),
  question_id            uuid REFERENCES writing_questions(id),
  task_type              text NOT NULL CHECK (task_type IN ('task1', 'task2')),
  essay_text             text NOT NULL,
  word_count             int NOT NULL,
  submitted_at           timestamptz DEFAULT now(),
  feedback_ta            text,
  feedback_cc            text,
  feedback_lr            text,
  feedback_gra           text,
  fix_ta                 text,
  fix_cc                 text,
  fix_lr                 text,
  fix_gra                text,
  band_ta                numeric(3,1),
  band_cc                numeric(3,1),
  band_lr                numeric(3,1),
  band_gra               numeric(3,1),
  band_overall           numeric(3,1),
  priority_fix           text,
  is_retry               boolean DEFAULT false,
  previous_submission_id uuid REFERENCES writing_submissions(id)
);
ALTER TABLE writing_submissions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "users_own_writing_submissions" ON writing_submissions;
CREATE POLICY "users_own_writing_submissions"
  ON writing_submissions FOR ALL
  USING (auth.uid() = user_id);
