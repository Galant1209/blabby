-- ============================================================================
-- 00000000_baseline_from_production.sql
--
-- BASELINE SNAPSHOT — DO NOT RUN AGAINST PRODUCTION.
--
-- These eight tables exist in production but have no CREATE TABLE anywhere in
-- supabase/migrations/. They were created by hand in the SQL editor, so a clean
-- replay of this directory used to fail: get_admin_users_full() joins
-- practice_records and sub-queries upgrade_intent, get_admin_user_activity()
-- joins practice_records and drill_usage — none of which the migration set
-- created. The RPCs could not be built, and /admin/users, /admin/activity and
-- /admin/waitlist would all be dead in any environment built from this repo.
--
-- Captured read-only from production project mkwywkwruyqzdhuzwnoa on
-- 2026-07-25 (pg_attribute / pg_constraint / pg_indexes / pg_policies /
-- information_schema.role_table_grants). Production already has every object
-- below, which is exactly why running this there is pointless and unsafe.
--
-- Its only purpose is to make `supabase/migrations/` replayable from empty.
-- Numbered 00000000 so it sorts before every real migration; every later
-- ALTER in this directory uses ADD COLUMN IF NOT EXISTS and is therefore a
-- no-op against the already-current shape below.
--
-- Everything here is idempotent: CREATE ... IF NOT EXISTS, DROP POLICY IF
-- EXISTS before CREATE POLICY, guarded constraint adds.
--
-- NOT captured here on purpose:
--   * RLS gaps. questions and writing_questions have RLS disabled in
--     production and are reproduced that way. Turning them on belongs to
--     20260726_billing_identity_containment.sql (TASK 2), not to a snapshot.
--   * pro_waitlist / upgrade_intent policy cleanup — belongs to TASK 5B.
--   * The permissive anon grants below are production's real state, warts and
--     all. A baseline that "fixes" things silently is not a baseline.
-- ============================================================================


-- ── practice_records — 241 rows, RLS enabled ────────────────────────────────
CREATE TABLE IF NOT EXISTS public.practice_records (
    id                   uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id              uuid NOT NULL REFERENCES auth.users(id),
    topic                text NOT NULL,
    question             text NOT NULL,
    user_transcript      text,
    coach_response       text,
    better_expression    text,
    better_expression_zh text,
    next_question        text,
    created_at           timestamptz DEFAULT now(),
    weakness_tag         text,
    memory_snapshot      jsonb,
    resolved             boolean NOT NULL DEFAULT false,
    mode                 text DEFAULT 'normal'::text,
    drill_tag            text,
    drill_score          jsonb,
    evidence             jsonb,
    quality_grade        text,
    quality_reason       text,
    notes                text,
    retry_of             uuid REFERENCES public.practice_records(id)
);

CREATE INDEX IF NOT EXISTS idx_practice_records_drill_lookup
    ON public.practice_records (user_id, mode, drill_tag, created_at DESC)
    WHERE (mode = 'drill'::text);
CREATE INDEX IF NOT EXISTS idx_practice_records_retry_of
    ON public.practice_records (retry_of) WHERE (retry_of IS NOT NULL);
CREATE INDEX IF NOT EXISTS idx_practice_records_user_unresolved
    ON public.practice_records (user_id, created_at DESC) WHERE (resolved = false);

ALTER TABLE public.practice_records ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "users can insert own records" ON public.practice_records;
CREATE POLICY "users can insert own records" ON public.practice_records
    FOR INSERT WITH CHECK (auth.uid() = user_id);
DROP POLICY IF EXISTS "users can read own records" ON public.practice_records;
CREATE POLICY "users can read own records" ON public.practice_records
    FOR SELECT USING (auth.uid() = user_id);


-- ── profiles — 17 rows, RLS enabled ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS public.profiles (
    id                      uuid PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    is_pro                  boolean NOT NULL DEFAULT false,
    created_at              timestamptz NOT NULL DEFAULT now(),
    updated_at              timestamptz NOT NULL DEFAULT now(),
    is_pro_grant            boolean NOT NULL DEFAULT false,
    pro_grant_reason        text,
    pro_grant_at            timestamptz,
    pro_grant_by            text,
    pro_grant_expires_at    timestamptz,
    reading_band_updated_at timestamptz,
    user_band_reading       numeric(3,1),
    covenant_signed_at      timestamptz,
    covenant_name           text
);

ALTER TABLE public.profiles ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "users can read own profile" ON public.profiles;
CREATE POLICY "users can read own profile" ON public.profiles
    FOR SELECT TO authenticated USING (auth.uid() = id);


-- ── questions — 0 rows, RLS DISABLED in production ──────────────────────────
-- drill_tags / difficulty are added again by 20260429_add_drill_metadata_to_
-- questions.sql with IF NOT EXISTS; both are included here because the
-- snapshot reflects 2026-07-25, i.e. after that migration ran.
CREATE TABLE IF NOT EXISTS public.questions (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    text       text NOT NULL,
    topic      text NOT NULL,
    part       integer NOT NULL DEFAULT 2,
    created_at timestamptz NOT NULL DEFAULT now(),
    drill_tags text[] NOT NULL DEFAULT '{}'::text[],
    difficulty text NOT NULL DEFAULT 'standard'::text
);

CREATE INDEX IF NOT EXISTS idx_questions_part ON public.questions (part);


-- ── writing_questions — 95 rows, RLS DISABLED in production ─────────────────
CREATE TABLE IF NOT EXISTS public.writing_questions (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    task_type         text NOT NULL,
    task1_subtype     text,
    prompt            text NOT NULL,
    chart_description text,
    essay_type        text,
    created_at        timestamptz DEFAULT now(),
    chart_svg         text,
    is_pregenerated   boolean DEFAULT false,
    used_count        integer DEFAULT 0
);

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conrelid = 'public.writing_questions'::regclass
                     AND conname = 'writing_questions_task_type_check') THEN
        ALTER TABLE public.writing_questions ADD CONSTRAINT writing_questions_task_type_check
            CHECK (task_type = ANY (ARRAY['task1'::text, 'task2'::text]));
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conrelid = 'public.writing_questions'::regclass
                     AND conname = 'writing_questions_task1_subtype_check') THEN
        ALTER TABLE public.writing_questions ADD CONSTRAINT writing_questions_task1_subtype_check
            CHECK (task1_subtype = ANY (ARRAY['bar_chart'::text, 'line_graph'::text,
                   'pie_chart'::text, 'table'::text, 'process'::text, 'map'::text]));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_writing_questions_pregenerated
    ON public.writing_questions (task_type, task1_subtype, is_pregenerated);


-- ── writing_submissions — 2 rows, RLS enabled ───────────────────────────────
CREATE TABLE IF NOT EXISTS public.writing_submissions (
    id                     uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id                uuid NOT NULL REFERENCES auth.users(id),
    question_id            uuid REFERENCES public.writing_questions(id),
    task_type              text NOT NULL,
    essay_text             text NOT NULL,
    word_count             integer NOT NULL,
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
    previous_submission_id uuid REFERENCES public.writing_submissions(id)
);

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint
                   WHERE conrelid = 'public.writing_submissions'::regclass
                     AND conname = 'writing_submissions_task_type_check') THEN
        ALTER TABLE public.writing_submissions ADD CONSTRAINT writing_submissions_task_type_check
            CHECK (task_type = ANY (ARRAY['task1'::text, 'task2'::text]));
    END IF;
END $$;

ALTER TABLE public.writing_submissions ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "users_own_writing_submissions" ON public.writing_submissions;
CREATE POLICY "users_own_writing_submissions" ON public.writing_submissions
    FOR ALL USING (auth.uid() = user_id);


-- ── upgrade_intent — 0 rows, RLS enabled ────────────────────────────────────
-- Four policies, three of which are overlapping INSERT grants with
-- WITH CHECK (true), and no UPDATE policy at all — which is why the opt-out
-- PATCH at upgrade.html:781 fails silently. Reproduced as-is; cleaning it up
-- is TASK 5B, not this snapshot.
CREATE TABLE IF NOT EXISTS public.upgrade_intent (
    id             uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id        uuid REFERENCES auth.users(id),
    email          text NOT NULL,
    reserved_price integer NOT NULL,
    reserved_at    timestamptz DEFAULT now(),
    source         text DEFAULT 'upgrade_page'::text,
    user_agent     text,
    metadata       jsonb DEFAULT '{}'::jsonb
);

CREATE INDEX IF NOT EXISTS idx_upgrade_intent_price ON public.upgrade_intent (reserved_price);
CREATE INDEX IF NOT EXISTS idx_upgrade_intent_user  ON public.upgrade_intent (user_id);

ALTER TABLE public.upgrade_intent ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS allow_anon_insert ON public.upgrade_intent;
CREATE POLICY allow_anon_insert ON public.upgrade_intent
    FOR INSERT TO anon, authenticated WITH CHECK (true);
DROP POLICY IF EXISTS anon_insert_upgrade_intent ON public.upgrade_intent;
CREATE POLICY anon_insert_upgrade_intent ON public.upgrade_intent
    FOR INSERT TO anon WITH CHECK (true);
DROP POLICY IF EXISTS authenticated_insert_upgrade_intent ON public.upgrade_intent;
CREATE POLICY authenticated_insert_upgrade_intent ON public.upgrade_intent
    FOR INSERT TO authenticated WITH CHECK (true);
DROP POLICY IF EXISTS allow_owner_select ON public.upgrade_intent;
CREATE POLICY allow_owner_select ON public.upgrade_intent
    FOR SELECT TO authenticated USING (auth.uid() = user_id);


-- ── rec_log — 40 rows, RLS enabled with ZERO policies (service_role only) ────
-- Also created by 20260617120832_create_rec_log_table.sql (the ledger
-- reconstruction). Both are IF NOT EXISTS so order does not matter.
CREATE TABLE IF NOT EXISTS public.rec_log (
    id            bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at    timestamptz NOT NULL DEFAULT now(),
    part          smallint,
    mime          text,
    recorder_mime text,
    ext           text,
    size          integer,
    chunk_count   integer,
    forced_mp4    boolean,
    status        text,
    error         text,
    ua            text
);

ALTER TABLE public.rec_log ENABLE ROW LEVEL SECURITY;


-- ── pro_waitlist — 0 rows, RLS enabled with ZERO policies ───────────────────
-- Zero policies + RLS on means the anon upsert at index.html:4155 can never
-- succeed, which is why this table is empty. Reproduced as-is; the fix is
-- TASK 5B (consolidate onto upgrade_intent).
CREATE TABLE IF NOT EXISTS public.pro_waitlist (
    email      text PRIMARY KEY,
    created_at timestamptz DEFAULT now()
);

ALTER TABLE public.pro_waitlist ENABLE ROW LEVEL SECURITY;


-- ── Grants, exactly as production has them ──────────────────────────────────
-- Production grants ALL to anon and authenticated on every one of these tables.
-- That is genuinely the live state (Supabase's default table privileges were
-- never narrowed). RLS is what actually restrains the six tables that have it
-- enabled; questions and writing_questions have neither, which is the hole
-- TASK 2's migration closes.
DO $$
DECLARE t text;
BEGIN
    FOREACH t IN ARRAY ARRAY['practice_records','profiles','questions','writing_questions',
                             'writing_submissions','upgrade_intent','rec_log','pro_waitlist']
    LOOP
        EXECUTE format('GRANT ALL ON TABLE public.%I TO anon, authenticated, service_role', t);
    END LOOP;
END $$;
