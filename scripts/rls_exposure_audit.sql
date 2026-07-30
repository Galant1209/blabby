-- ============================================================================
-- rls_exposure_audit.sql — read-only exposure audit for the content tables.
--
-- Run BEFORE and AFTER any lockdown migration and diff the two outputs. Re-run
-- whenever a table is added, so a new table cannot arrive unprotected.
--
--   psql "$PGURI" -f scripts/rls_exposure_audit.sql
--
-- Read-only: catalog SELECTs, has_column_privilege(), and role-switched read
-- attempts inside a transaction that is never committed. Writes nothing.
--
-- ── WHY THIS SCRIPT IS COLUMN-LEVEL ─────────────────────────────────────────
-- An earlier version asked has_table_privilege(role, tbl, 'SELECT'). That is
-- the wrong question, and it misled a real audit on 2026-07-30.
--
-- Postgres GRANT works at column granularity. For a table with column-level
-- grants, has_table_privilege returns FALSE even though the role can read most
-- of the table. The failure runs in both directions:
--
--   * false "safe"      — a role with grants on 14 of 15 columns reads almost
--                         everything, and the table-level probe says "no grant"
--   * false "exposed"   — a table whose sensitive columns are individually
--                         withheld looks identical to one with no protection
--
-- Neither direction raises an error. On 2026-07-30 the second one nearly caused
-- a working column-level GRANT on reading_questions to be deleted as "dead
-- policy". See §3 below and the header of
-- supabase/migrations/20260731_content_access_lockdown.sql.
--
-- Catalogs are an index, not evidence. §4 therefore re-derives every verdict by
-- actually attempting the read as anon and as authenticated.
-- ============================================================================

\echo ''
\echo '════════ §1  Per-table summary (column-level aware) ════════'

WITH audited(tbl, kind) AS (
    VALUES ('questions',         'content'),
           ('reading_passages',  'content'),
           ('writing_questions', 'content'),
           ('reading_questions', 'content+answers'),
           ('vocabulary_items',  'shared dictionary')
),
cols AS (
    SELECT a.tbl, a.kind, c.oid, att.attname, att.attnum,
           has_column_privilege('anon',          c.oid, att.attnum, 'SELECT') AS anon_ok,
           has_column_privilege('authenticated', c.oid, att.attnum, 'SELECT') AS auth_ok
      FROM audited a
      JOIN pg_class c     ON c.relname = a.tbl
      JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = 'public'
      JOIN pg_attribute att ON att.attrelid = c.oid
                           AND att.attnum > 0 AND NOT att.attisdropped
)
SELECT
    tbl                                   AS "table",
    kind,
    (SELECT c2.relrowsecurity FROM pg_class c2 WHERE c2.oid = cols.oid) AS rls_on,
    (SELECT count(*) FROM pg_policies p
      WHERE p.schemaname='public' AND p.tablename = cols.tbl)           AS policies,
    count(*)                              AS n_cols,
    count(*) FILTER (WHERE anon_ok)       AS anon_readable_cols,
    count(*) FILTER (WHERE auth_ok)       AS auth_readable_cols,
    CASE
        WHEN count(*) FILTER (WHERE anon_ok) = 0
         AND count(*) FILTER (WHERE auth_ok) = 0 THEN 'LOCKED (service_role only)'
        WHEN count(*) FILTER (WHERE anon_ok) = count(*) THEN 'ANON READS EVERYTHING'
        WHEN count(*) FILTER (WHERE anon_ok) > 0        THEN 'anon partial'
        WHEN count(*) FILTER (WHERE auth_ok) = count(*) THEN 'authenticated reads everything'
        ELSE 'authenticated partial (column-level grant in force)'
    END                                   AS verdict
FROM cols
GROUP BY tbl, kind, oid
ORDER BY tbl;

\echo ''
\echo '════════ §2  Per-column detail — which columns are actually reachable ════════'

SELECT
    c.relname AS "table",
    att.attname AS column_name,
    has_column_privilege('anon',          c.oid, att.attnum, 'SELECT') AS anon,
    has_column_privilege('authenticated', c.oid, att.attnum, 'SELECT') AS authenticated,
    has_column_privilege('service_role',  c.oid, att.attnum, 'SELECT') AS service_role,
    att.attacl::text AS column_acl
  FROM pg_class c
  JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = 'public'
  JOIN pg_attribute att ON att.attrelid = c.oid AND att.attnum > 0 AND NOT att.attisdropped
 WHERE c.relname IN ('questions', 'reading_passages', 'writing_questions',
                     'reading_questions', 'vocabulary_items')
 ORDER BY c.relname, att.attnum;

\echo ''
\echo '════════ §3  reading_questions — the two-layer invariant ════════'
\echo 'row layer:    policy reading_questions_prompt_select, USING (true) — all rows'
\echo 'column layer: GRANT on 7 safe columns only — answers never granted'
\echo 'Both layers are deliberate. Removing either one is a regression.'
\echo 'Source: supabase/migrations/20260714_p1_rls_and_reading_answers.sql:68-71'
\echo ''

SELECT
    CASE WHEN att.attname IN ('correct_answer','explanation','evidence_quote')
         THEN 'ANSWER   ' ELSE 'safe     ' END || att.attname AS column_name,
    has_column_privilege('authenticated', c.oid, att.attnum, 'SELECT') AS authenticated,
    CASE
        WHEN att.attname IN ('correct_answer','explanation','evidence_quote')
             AND has_column_privilege('authenticated', c.oid, att.attnum, 'SELECT')
            THEN '*** ANSWER LEAK ***'
        WHEN att.attname NOT IN ('correct_answer','explanation','evidence_quote')
             AND NOT has_column_privilege('authenticated', c.oid, att.attnum, 'SELECT')
            THEN '!! answering path broken !!'
        ELSE 'ok'
    END AS invariant
  FROM pg_class c
  JOIN pg_namespace n ON n.oid = c.relnamespace AND n.nspname = 'public'
  JOIN pg_attribute att ON att.attrelid = c.oid AND att.attnum > 0 AND NOT att.attisdropped
 WHERE c.relname = 'reading_questions'
 ORDER BY att.attnum;

\echo ''
\echo '════════ §4  Evidence — actual reads as anon and authenticated ════════'
\echo 'Catalogs are an index. These are the reads themselves.'
\echo ''

BEGIN;

DO $audit$
DECLARE
    tbl        text;
    n          bigint;
    answer_tbl CONSTANT text := 'reading_questions';
BEGIN
    FOREACH tbl IN ARRAY ARRAY['questions', 'reading_passages',
                               'writing_questions', 'vocabulary_items'] LOOP
        BEGIN
            SET LOCAL ROLE anon;
            EXECUTE format('SELECT count(*) FROM public.%I', tbl) INTO n;
            RAISE NOTICE 'anon          SELECT * %-20s -> ALLOWED, % row(s)', tbl, n;
        EXCEPTION WHEN insufficient_privilege THEN
            RAISE NOTICE 'anon          SELECT * %-20s -> DENIED (42501)', tbl;
        END;
        RESET ROLE;
    END LOOP;

    -- reading_questions: answers must be denied, safe columns must work.
    BEGIN
        SET LOCAL ROLE anon;
        EXECUTE format('SELECT count(*) FROM public.%I', answer_tbl) INTO n;
        RAISE NOTICE 'anon          SELECT * %-20s -> ALLOWED, % row(s)', answer_tbl, n;
    EXCEPTION WHEN insufficient_privilege THEN
        RAISE NOTICE 'anon          SELECT * %-20s -> DENIED (42501)', answer_tbl;
    END;
    RESET ROLE;

    BEGIN
        SET LOCAL ROLE authenticated;
        EXECUTE 'SELECT count(*) FROM public.reading_questions WHERE correct_answer IS NOT NULL' INTO n;
        RAISE NOTICE 'authenticated correct_answer                     -> ALLOWED, % row(s)  *** LEAK ***', n;
    EXCEPTION WHEN insufficient_privilege THEN
        RAISE NOTICE 'authenticated correct_answer                     -> DENIED (42501)  [expected]';
    END;
    RESET ROLE;

    BEGIN
        SET LOCAL ROLE authenticated;
        EXECUTE 'SELECT count(*) FROM (SELECT id, passage_id, question_type, question_text,'
                ' options, order_idx, created_at FROM public.reading_questions) q' INTO n;
        RAISE NOTICE 'authenticated safe columns                       -> ALLOWED, % row(s)  [expected]', n;
    EXCEPTION WHEN insufficient_privilege THEN
        RAISE NOTICE 'authenticated safe columns                       -> DENIED  !! answering path broken !!';
    END;
    RESET ROLE;

    FOREACH tbl IN ARRAY ARRAY['questions', 'reading_passages', 'writing_questions'] LOOP
        BEGIN
            SET LOCAL ROLE authenticated;
            EXECUTE format('SELECT count(*) FROM public.%I', tbl) INTO n;
            RAISE NOTICE 'authenticated SELECT * %-20s -> ALLOWED, % row(s)', tbl, n;
        EXCEPTION WHEN insufficient_privilege THEN
            RAISE NOTICE 'authenticated SELECT * %-20s -> DENIED (42501)', tbl;
        END;
        RESET ROLE;
    END LOOP;
END
$audit$;

ROLLBACK;

\echo ''
\echo 'Audit complete. Nothing was written.'
\echo ''
