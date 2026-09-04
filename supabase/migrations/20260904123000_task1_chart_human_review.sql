-- Task 1 chart human-review metadata.
--
-- Review state is deliberately separate from is_pregenerated: saving an
-- engineering review must never change the serving pool or reactivate a row.
-- Existing rows, including the eleven soft-retired multi-period pies, start in
-- a safe pending state.

ALTER TABLE public.writing_questions
  ADD COLUMN IF NOT EXISTS review_status text NOT NULL DEFAULT 'pending',
  ADD COLUMN IF NOT EXISTS review_issue text,
  ADD COLUMN IF NOT EXISTS review_note text,
  ADD COLUMN IF NOT EXISTS reviewed_at timestamptz,
  ADD COLUMN IF NOT EXISTS reviewed_by uuid;

DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conrelid = 'public.writing_questions'::regclass
      AND conname = 'writing_questions_review_status_check'
  ) THEN
    ALTER TABLE public.writing_questions
      ADD CONSTRAINT writing_questions_review_status_check
      CHECK (review_status IN ('pending', 'approved', 'needs_fix', 'retired'));
  END IF;

  IF NOT EXISTS (
    SELECT 1
    FROM pg_constraint
    WHERE conrelid = 'public.writing_questions'::regclass
      AND conname = 'writing_questions_review_issue_check'
  ) THEN
    ALTER TABLE public.writing_questions
      ADD CONSTRAINT writing_questions_review_issue_check
      CHECK (
        review_issue IS NULL OR review_issue IN (
          'renderer_unsupported',
          'data_shape_invalid',
          'misleading_visual',
          'label_collision',
          'unreadable',
          'content_issue',
          'other'
        )
      );
  END IF;
END
$$;

CREATE INDEX IF NOT EXISTS idx_writing_questions_review_status
  ON public.writing_questions(task_type, review_status, created_at);
