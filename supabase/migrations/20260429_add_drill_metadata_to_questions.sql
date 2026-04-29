-- Sprint 2B: drill metadata for the questions bank.
--
-- drill_tags : array of weakness tags this question is suitable for drilling
--              against. Empty array means "generic, fits any drill".
--              Existing 28 rows are intentionally NOT backfilled here —
--              backfill is a separate task outside Sprint 2B.
-- difficulty : reserved for future tiering. 'standard' is the only value
--              used today; reserved values are 'easy' and 'hard'.

ALTER TABLE questions
  ADD COLUMN IF NOT EXISTS drill_tags TEXT[] NOT NULL DEFAULT '{}',
  ADD COLUMN IF NOT EXISTS difficulty TEXT NOT NULL DEFAULT 'standard';
