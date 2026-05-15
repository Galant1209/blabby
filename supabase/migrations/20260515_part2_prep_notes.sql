-- Add preparation notes to practice records. Part 2 students jot keywords
-- and structure during the prepare phase; we store them alongside the
-- transcript so the scoring prompt can diagnose gaps between intent and
-- speech, and so the UI can show the notes back to the student.
--
-- Nullable: pre-existing records and Part 1 records both leave it NULL.

ALTER TABLE practice_records
ADD COLUMN IF NOT EXISTS notes TEXT;
