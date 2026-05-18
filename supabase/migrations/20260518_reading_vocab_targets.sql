-- Sprint Reading-2: curated vocab targets on each passage.
--
-- Stores the 6–10 words Claude flags as "worth looking up" relative to the
-- band of the user who triggered generation. Frontend uses this list to
-- decide which words get the dotted-underline affordance. Words not in
-- this list are rendered as plain text and are not clickable.
--
-- Apply manually via Supabase SQL Editor BEFORE deploying the backend
-- changes that read/write this column (same pattern as 20260518_reading_module.sql
-- and 20260518_reading_band_updated_at.sql).

alter table public.reading_passages
    add column if not exists vocab_targets jsonb;

-- Shape: ["aristocracy", "disturbances", "nominalisation", ...]
-- Lowercased, deduplicated, in passage-appearance order.
-- Existing rows are left NULL — the frontend treats NULL as
-- "legacy passage, fall back to no-targets / disable click-to-look-up".
