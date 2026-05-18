-- Sprint Reading-1: Reading module schema.
--
-- Adds 4 new tables for the Reading practice flow plus a profiles.user_band_reading
-- column. Does NOT touch any Speaking-module table or the existing profiles.user_band
-- column (which now implicitly represents Speaking).
--
-- Quota: Reading's "1 new attempt per calendar day for free users" is NOT enforced
-- via a generic daily_usage table (none exists). Speaking currently enforces its
-- own quota inline in Python by counting drill_usage rows over a rolling window
-- (see main.py:_drill_quota_state). The Reading endpoint should follow the same
-- pattern — count reading_attempts rows with started_at >= today (UTC) — so no
-- separate usage table is created here. Flagged in the migration deliverable.

-- =========================================================================
-- 1. Passages (AI-generated IELTS Reading articles)
-- =========================================================================
create table public.reading_passages (
    id                uuid          primary key default gen_random_uuid(),
    title             text          not null,
    body              text          not null,
    difficulty_band   numeric(3,1)  not null check (difficulty_band between 4.0 and 9.0),
    topic             text,
    source            text          not null default 'ai_generated'
                                    check (source in ('ai_generated','curated')),
    word_count        int,
    created_at        timestamptz   not null default now()
);

-- =========================================================================
-- 2. Questions belonging to a passage
-- =========================================================================
create table public.reading_questions (
    id              uuid          primary key default gen_random_uuid(),
    passage_id      uuid          not null references public.reading_passages(id) on delete cascade,
    question_type   text          not null check (question_type in ('mcq','tfng','heading')),
    question_text   text          not null,
    options         jsonb,                       -- mcq: ["A...","B...","C...","D..."]; heading: candidate headings; tfng: null
    correct_answer  text          not null,     -- mcq: "A"/"B"/...; tfng: "True"/"False"/"Not Given"; heading: matched heading text or its index
    explanation     text          not null,
    evidence_quote  text,                        -- exact span from passage supporting the answer
    order_idx       int           not null,
    created_at      timestamptz   not null default now()
);
create index on public.reading_questions (passage_id, order_idx);

-- =========================================================================
-- 3. A user's attempt at one passage
-- =========================================================================
create table public.reading_attempts (
    id              uuid          primary key default gen_random_uuid(),
    user_id         uuid          not null references auth.users(id) on delete cascade,
    passage_id      uuid          not null references public.reading_passages(id),
    started_at      timestamptz   not null default now(),
    submitted_at    timestamptz,
    score           int,                          -- correct count
    total           int,                          -- total questions
    band_estimate   numeric(3,1),                 -- derived band from score
    status          text          not null default 'in_progress'
                                  check (status in ('in_progress','submitted','abandoned'))
);
create index on public.reading_attempts (user_id, started_at desc);

-- =========================================================================
-- 4. Individual answers within an attempt
-- =========================================================================
create table public.reading_answers (
    id            uuid          primary key default gen_random_uuid(),
    attempt_id    uuid          not null references public.reading_attempts(id) on delete cascade,
    question_id   uuid          not null references public.reading_questions(id),
    user_answer   text,
    is_correct    boolean,
    answered_at   timestamptz   not null default now(),
    unique (attempt_id, question_id)
);

-- =========================================================================
-- 5. Extend the existing profile table
-- =========================================================================
-- Confirmed via main.py:3072 — Speaking writes to public.profiles.user_band.
-- We add a separate column for Reading so the two bands stay independent.
alter table public.profiles
    add column if not exists user_band_reading numeric(3,1);

-- =========================================================================
-- 6. RLS
-- =========================================================================
-- Shared content — RLS disabled (matches the Speaking-module posture for
-- `questions`, per project memory). Service-role bypasses RLS for inserts;
-- authenticated users only ever read.
alter table public.reading_passages  disable row level security;
alter table public.reading_questions disable row level security;

-- User-owned rows — RLS enabled, owner-scoped policy.
alter table public.reading_attempts enable row level security;
create policy "reading_attempts_owner" on public.reading_attempts
    for all using (auth.uid() = user_id);

alter table public.reading_answers enable row level security;
create policy "reading_answers_owner" on public.reading_answers
    for all using (
        exists (
            select 1 from public.reading_attempts a
            where a.id = reading_answers.attempt_id
              and a.user_id = auth.uid()
        )
    );
