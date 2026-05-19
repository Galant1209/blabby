-- Migration: add created_by to reading_passages
-- Rationale: ownership tracking. The two-endpoint split (passage generate +
-- questions generate) introduced a new attack surface where a user could
-- spend another user's Sonnet tokens by passing their passage_id to the new
-- /reading/questions/generate endpoint. created_by lets the questions
-- endpoint enforce "you can only generate questions for passages you
-- created". Backfilled from reading_attempts for historical passages.

-- Step 1: add the column (nullable to allow backfill on existing rows)
alter table public.reading_passages
  add column if not exists created_by uuid;

-- Step 2: backfill from reading_attempts
-- For each existing passage, find the earliest attempt and assign that
-- user_id as the owner. Passages with zero attempts (extremely rare —
-- only happens if a passage was generated but the attempt-start step
-- failed) remain NULL.
update public.reading_passages p
   set created_by = sub.user_id
  from (
    select distinct on (passage_id)
           passage_id,
           user_id
      from public.reading_attempts
     order by passage_id, started_at asc
  ) sub
 where p.id = sub.passage_id
   and p.created_by is null;

-- Step 3: index for ownership lookups
-- Frequent query pattern: "is this passage owned by this user?" (questions
-- endpoint) and "list this user's passages" (future history feature).
create index if not exists idx_reading_passages_created_by
  on public.reading_passages(created_by);

-- Verification queries (run manually, do not include in migration):
--
--   -- Count rows by backfill status
--   select count(*) filter (where created_by is null) as orphans,
--          count(*) filter (where created_by is not null) as owned,
--          count(*) as total
--     from public.reading_passages;
--
--   -- Sanity-check: no passage should have a created_by that doesn't
--   -- match any of its attempts' user_ids
--   select p.id, p.created_by, a.user_id
--     from public.reading_passages p
--     join public.reading_attempts a on a.passage_id = p.id
--    where p.created_by is not null
--      and p.created_by <> a.user_id
--    limit 10;
