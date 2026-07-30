-- ============================================================================
-- 20260731_content_access_lockdown.sql
--
-- Closes anon's direct read access to the three content tables:
--     questions (71 rows) / reading_passages (32) / writing_questions (95)
--
-- Verified 2026-07-30 by reading them as the anon role, not by reading
-- catalogs: all three returned their full row counts. The anon key is
-- published in frontend/app/config.js:3 and this repo is public, so "anon"
-- means anyone with a browser.
--
-- Two layers, because RLS and GRANT are independent and doing one is doing
-- nothing:
--     ENABLE ROW LEVEL SECURITY  + zero policies   → deny-all at the row layer
--     REVOKE SELECT              from anon/authenticated → nothing at the grant layer
--
-- service_role has BYPASSRLS and keeps its grant, so FastAPI is unaffected.
-- Verified 2026-07-30: backend has exactly one Supabase client
-- (main.py:263, SUPABASE_SERVICE_KEY) and all 153 call sites use it; the
-- frontend issues zero queries against these tables and zero .rpc() calls.
--
-- ── ZERO POLICY IS THE DESIGN, NOT AN OVERSIGHT ─────────────────────────────
-- The failure mode of a deny-all table is that someone later sees an empty
-- array, assumes a bug, and "fixes" it with USING (true). That single line
-- reverts this entire migration.
--
--     Content tables are RLS on + ZERO policies, permanently.
--     Every read goes through FastAPI + service_role.
--     An empty array means the caller bypassed FastAPI.
--     Fix the caller. Do not add a policy.
--     A new policy requires an explicit product decision.
--
-- ── reading_questions IS DELIBERATELY NOT IN THIS FILE ──────────────────────
-- It is the asymmetry someone will eventually try to "tidy up". Do not.
--
-- reading_questions holds correct_answer / explanation / evidence_quote, and is
-- already protected by a different, correct mechanism installed by
-- 20260714_p1_rls_and_reading_answers.sql:68-71 —
--
--     row layer:    policy reading_questions_prompt_select, USING (true)
--                   → every row visible, which is right; a question bank is
--                     not per-user
--     column layer: GRANT SELECT (id, passage_id, question_type, question_text,
--                                 options, order_idx, created_at)
--                   → the three answer columns are never granted to anyone
--
-- Together they express "any logged-in user may read any question, but nobody
-- may read its answer" — which RLS alone cannot say, because RLS filters rows
-- and this is a column constraint. It mirrors the FastAPI allowlist at
-- main.py:7047-7056, so the two are mutual backups rather than duplicates.
--
-- WHY SOMEONE WILL WANT TO REMOVE IT, AND WHY THEY WILL BE WRONG:
--
-- On 2026-07-30 an audit concluded that reading_questions was leaking every
-- answer to every registered user, and that TASK 1's fix had been ineffective.
-- The remediation drafted from that conclusion was to drop the policy and the
-- grants so that all four content tables would share one uniform rule.
--
-- The conclusion was false. It came from asking
--     has_table_privilege('authenticated', 'reading_questions', 'SELECT')
-- which returns FALSE for a table carrying column-level grants — identical to
-- the answer for a table with no grants at all. The audit saw
-- "policy USING (true)" next to "answer columns present" and inferred a leak
-- that the grant layer had already prevented.
--
-- What settled it was attempting the read:
--     SET LOCAL ROLE authenticated;
--     SELECT count(*) FROM reading_questions WHERE correct_answer IS NOT NULL;
--     ERROR: 42501: permission denied for table reading_questions
-- while the seven safe columns returned 252 rows normally.
--
-- Had the remediation shipped, it would have deleted a working defence and
-- replaced two independent layers with one. The uniformity would have been
-- real and the security worse.
--
-- The asymmetry is load-bearing: these three tables have no answer columns,
-- reading_questions does. One rule cannot cover both without losing something.
-- scripts/rls_exposure_audit.sql was rewritten to probe per column and to
-- re-derive every verdict by actually attempting the read, so this specific
-- misreading cannot recur silently.
--
-- The preflight below asserts those grants are still exactly as described. This
-- file changes nothing about reading_questions; it proves nothing else has.
--
-- ── RELATIONSHIP TO 20260726 §4 ─────────────────────────────────────────────
-- 20260726_billing_identity_containment.sql §4 targeted these same three tables
-- and was never applied. It planned RLS plus three permissive policies granting
-- authenticated SELECT. This file REPLACES that design with fail-closed — it
-- does not merely complete it. Recorded explicitly because someone comparing
-- against the original 20260726 SQL will find three policies that were never
-- created, and that absence is a decision, not a missed statement:
--
--     20260726 §4 planned          this file ships
--     ─────────────────────────    ──────────────────────────────
--     RLS on x3                    RLS on x3                      (same)
--     3 permissive SELECT policies zero policies                  (REPLACED)
--     REVOKE anon                  REVOKE ALL, anon+authenticated (widened)
--     GRANT SELECT authenticated   no grant                       (REPLACED)
--
-- The reason: the frontend reads none of these tables directly, so granting
-- authenticated buys nothing and leaves a second path to the content.
--
-- §0 below DROPs those three policies by name. That matters because 20260726
-- has not been applied to production but IS applied on every clean replay, so
-- CI reaches this file with all three policies live. Without §0 the lockdown
-- would be a no-op exactly where it gets tested — RLS enabled with a
-- USING (true) policy attached is not a lockdown at all.
--
-- ── INDEPENDENCE FROM THE ECPay CHAIN ───────────────────────────────────────
-- Zero object overlap. This file touches only questions / reading_passages /
-- writing_questions. The ECPay chain touches payment_events, subscriptions,
-- is_user_pro(), get_admin_pro_breakdown() and accept_ecpay_payment() — no
-- table, function, policy or index appears in both.
--
-- Therefore this migration may be applied BEFORE 20260730_0 and 20260730, or
-- after, or without them ever running. The filename sorting after them is
-- irrelevant: neither reads anything the other writes, and neither preflight
-- inspects an object the other touches.
--
-- Properties: forward-only, idempotent (safe to re-run), non-destructive
-- (drops no table, column or row). Rollback in
-- 20260731_content_access_lockdown_rollback.sql.
--
-- NOT YET EXECUTED against production.
-- ============================================================================

BEGIN;

-- ── EXECUTABLE PRE-MUTATION GATE ────────────────────────────────────────────
-- Accepts exactly three states, and nothing else:
--
--   (A) production today   — 20260726 §4 never applied: RLS off, zero policies,
--                            anon and authenticated read every column
--   (B) replay / any DB where 20260726 §4 DID run — RLS on, its three named
--                            policies present, anon revoked, authenticated granted
--   (C) already locked     — RLS on, zero policies, neither role granted
--
-- (B) is not hypothetical: a clean replay of supabase/migrations/ executes
-- 20260726 in full, so CI always arrives here in state (B) while production is
-- in state (A). A gate that accepted only (A) and (C) would pass in production
-- and fail in CI — the worst possible split, since CI is where it gets tested.
--
-- Any policy whose name is not one of 20260726 §4's three is an unknown
-- re-opening: abort and name it rather than dropping something deliberate.
DO $preflight$
DECLARE
    tbl              text;
    rls              boolean;
    n_policies       integer;
    n_known          integer;
    anon_cols        integer;
    auth_cols        integer;
    total_cols       integer;
    stray            text;
    safe_granted     integer;
    answers_granted  integer;
    anon_rq_cols     integer;
    -- The exact policies 20260726 §4 creates on these tables. This file
    -- replaces them with fail-closed, so it must be able to recognise and
    -- remove them — but only these.
    known_policies   CONSTANT text[] := ARRAY[
        'questions_read_authenticated',
        'writing_questions_read_authenticated',
        'reading_passages_read_own'
    ];
BEGIN
    ---------------------------------------------------------------- three tables
    FOREACH tbl IN ARRAY ARRAY['questions', 'reading_passages', 'writing_questions'] LOOP
        IF NOT EXISTS (
            SELECT 1 FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE n.nspname = 'public' AND c.relname = tbl AND c.relkind = 'r'
        ) THEN
            RAISE EXCEPTION
                'content lockdown: required table public.% is missing', tbl;
        END IF;

        SELECT c.relrowsecurity INTO rls
          FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = 'public' AND c.relname = tbl;

        SELECT count(*), count(*) FILTER (WHERE p.policyname = ANY(known_policies))
          INTO n_policies, n_known
          FROM pg_policies p
         WHERE p.schemaname = 'public' AND p.tablename = tbl;

        -- Unknown policies are never dropped silently.
        IF n_policies <> n_known THEN
            SELECT string_agg(p.policyname, ', ' ORDER BY p.policyname) INTO stray
              FROM pg_policies p
             WHERE p.schemaname = 'public' AND p.tablename = tbl
               AND NOT (p.policyname = ANY(known_policies));
            RAISE EXCEPTION
                'content lockdown: public.% carries unrecognised policy/policies (%). '
                'This file only removes 20260726 §4''s own policies. Decide what '
                'these are for before locking down.', tbl, stray;
        END IF;

        SELECT count(*),
               count(*) FILTER (WHERE has_column_privilege('anon', c.oid, a.attnum, 'SELECT')),
               count(*) FILTER (WHERE has_column_privilege('authenticated', c.oid, a.attnum, 'SELECT'))
          INTO total_cols, anon_cols, auth_cols
          FROM pg_class c
          JOIN pg_namespace n ON n.oid = c.relnamespace
          JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
         WHERE n.nspname = 'public' AND c.relname = tbl;

        IF NOT (
          -- (A) production today
             (rls = false AND n_policies = 0
                          AND anon_cols = total_cols AND auth_cols = total_cols)
          -- (B) 20260726 §4 applied (replay/CI)
          OR (rls = true  AND n_policies > 0
                          AND anon_cols = 0          AND auth_cols = total_cols)
          -- (C) already locked by this file
          OR (rls = true  AND n_policies = 0
                          AND anon_cols = 0          AND auth_cols = 0)
        ) THEN
            RAISE EXCEPTION
                'content lockdown: public.% is in an unrecognised state '
                '(rls=%, policies=%, anon readable %/%, authenticated readable %/%). '
                'Expected the pre-lockdown state, the 20260726 §4 state, or the '
                'locked state produced by this file.',
                tbl, rls, n_policies, anon_cols, total_cols, auth_cols, total_cols;
        END IF;
    END LOOP;

    ------------------------------------------------- reading_questions, untouched
    -- This file does not modify reading_questions. It asserts that the
    -- two-layer protection described in the header is still intact, so that a
    -- lockdown cannot be recorded as "done" while the answer columns were
    -- opened up by something else in the meantime.
    IF NOT EXISTS (
        SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relname = 'reading_questions' AND c.relkind = 'r'
    ) THEN
        RAISE EXCEPTION 'content lockdown: public.reading_questions is missing';
    END IF;

    SELECT
        count(*) FILTER (
            WHERE a.attname IN ('id','passage_id','question_type','question_text',
                                'options','order_idx','created_at')
              AND has_column_privilege('authenticated', c.oid, a.attnum, 'SELECT')),
        count(*) FILTER (
            WHERE a.attname IN ('correct_answer','explanation','evidence_quote')
              AND has_column_privilege('authenticated', c.oid, a.attnum, 'SELECT')),
        count(*) FILTER (
            WHERE has_column_privilege('anon', c.oid, a.attnum, 'SELECT'))
      INTO safe_granted, answers_granted, anon_rq_cols
      FROM pg_class c
      JOIN pg_namespace n ON n.oid = c.relnamespace
      JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum > 0 AND NOT a.attisdropped
     WHERE n.nspname = 'public' AND c.relname = 'reading_questions';

    IF answers_granted <> 0 THEN
        RAISE EXCEPTION
            'content lockdown: reading_questions has granted % answer column(s) to '
            'authenticated. The column-level GRANT from 20260714 has been altered; '
            'that is an answer leak and must be fixed before anything else.',
            answers_granted;
    END IF;
    IF safe_granted <> 7 THEN
        RAISE EXCEPTION
            'content lockdown: reading_questions grants authenticated only %/7 of the '
            'answer-free columns. The answering path is broken; investigate before '
            'locking down further tables.', safe_granted;
    END IF;
    IF anon_rq_cols <> 0 THEN
        RAISE EXCEPTION
            'content lockdown: reading_questions is readable by anon on % column(s); '
            'expected none.', anon_rq_cols;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
         WHERE schemaname = 'public' AND tablename = 'reading_questions'
           AND policyname = 'reading_questions_prompt_select'
    ) THEN
        RAISE EXCEPTION
            'content lockdown: policy reading_questions_prompt_select is missing. '
            'Without it authenticated cannot read any row and Reading breaks.';
    END IF;
END
$preflight$;


-- ── §0  Remove 20260726 §4's permissive policies ────────────────────────────
-- This is where "REPLACES §4" stops being a comment and becomes a statement.
--
-- In production these three do not exist and every DROP is a no-op. In a clean
-- replay 20260726 §4 has just created them, and leaving them would make the
-- lockdown a no-op there — RLS on with a USING (true) policy is RLS on with
-- everything visible, which is precisely the failure mode this file warns about
-- in its header.
--
-- IF EXISTS on all three, so this is re-runnable from any accepted state.

DROP POLICY IF EXISTS questions_read_authenticated          ON public.questions;
DROP POLICY IF EXISTS writing_questions_read_authenticated  ON public.writing_questions;
DROP POLICY IF EXISTS reading_passages_read_own             ON public.reading_passages;


-- ── §1  Row layer — deny-all ────────────────────────────────────────────────
-- ENABLE on an already-enabled table is a no-op, so this is re-runnable.
-- No policy is created. That is the point; see the header.

ALTER TABLE public.questions          ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.reading_passages   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.writing_questions  ENABLE ROW LEVEL SECURITY;


-- ── §2  Grant layer — remove the privilege entirely ─────────────────────────
-- RLS without this is theatre: a role still holding SELECT would read every
-- row the moment any policy appears. REVOKE on an already-revoked privilege is
-- a no-op, so this is re-runnable too.
--
-- ALL, not just SELECT: these tables are written exclusively by service_role,
-- so anon/authenticated have no business holding INSERT/UPDATE/DELETE either.

REVOKE ALL ON TABLE public.questions          FROM anon, authenticated;
REVOKE ALL ON TABLE public.reading_passages   FROM anon, authenticated;
REVOKE ALL ON TABLE public.writing_questions  FROM anon, authenticated;


COMMENT ON TABLE public.questions IS
    'Part 1 question bank. RLS on with ZERO policies, by design: reads go '
    'through FastAPI + service_role only. An empty array means the caller '
    'bypassed FastAPI — fix the caller, do not add a policy.';
COMMENT ON TABLE public.reading_passages IS
    'Reading passages. RLS on with ZERO policies, by design: reads go through '
    'FastAPI + service_role only. An empty array means the caller bypassed '
    'FastAPI — fix the caller, do not add a policy.';
COMMENT ON TABLE public.writing_questions IS
    'Writing question bank. RLS on with ZERO policies, by design: reads go '
    'through FastAPI + service_role only. An empty array means the caller '
    'bypassed FastAPI — fix the caller, do not add a policy.';

COMMIT;


-- ── verification (run after applying; all four must hold) ───────────────────
-- Prefer scripts/rls_exposure_audit.sql, which performs the reads rather than
-- inspecting catalogs. Minimum manual checks:
--
-- 1. SELECT relname, relrowsecurity FROM pg_class
--     WHERE relname IN ('questions','reading_passages','writing_questions');
--    → relrowsecurity = true for all three
--
-- 2. SELECT count(*) FROM pg_policies WHERE schemaname='public'
--     AND tablename IN ('questions','reading_passages','writing_questions');
--    → 0
--
-- 3. BEGIN; SET LOCAL ROLE anon;
--    SELECT count(*) FROM public.questions;   → ERROR 42501
--    ROLLBACK;
--
-- 4. reading_questions must be UNCHANGED:
--    BEGIN; SET LOCAL ROLE authenticated;
--    SELECT count(*) FROM reading_questions WHERE correct_answer IS NOT NULL;
--      → ERROR 42501   (answers still denied)
--    SELECT count(*) FROM (SELECT id, passage_id, question_type, question_text,
--                                 options, order_idx, created_at
--                            FROM reading_questions) q;
--      → 252           (answering path still works)
--    ROLLBACK;
