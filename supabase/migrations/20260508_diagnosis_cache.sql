-- Cache /api/diagnosis/me output keyed by user_id. Reused as long as
-- practice_count matches — any new practice invalidates by mismatch.
-- One row per user; UPSERT on user_id.

CREATE TABLE IF NOT EXISTS diagnosis_cache (
    user_id        uuid PRIMARY KEY,
    content        text NOT NULL,
    practice_count integer NOT NULL,
    created_at     timestamptz DEFAULT now(),
    updated_at     timestamptz DEFAULT now()
);
