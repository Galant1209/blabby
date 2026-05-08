-- Per-record quality classification used by the admin triage UI.
-- Populated by classify_quality() in backend/main.py at insert time;
-- backfilled by /api/admin/reclassify for historical rows.
--
-- grade values: 'valid' | 'partial' | 'invalid' | 'unknown'
-- (no enum/check constraint — keep it loose so the LLM swap can introduce
--  new buckets without a schema change)

ALTER TABLE practice_records
ADD COLUMN IF NOT EXISTS quality_grade text DEFAULT NULL,
ADD COLUMN IF NOT EXISTS quality_reason text DEFAULT NULL;

CREATE INDEX IF NOT EXISTS practice_records_quality_grade_idx
ON practice_records (quality_grade)
WHERE quality_grade IS NOT NULL;
