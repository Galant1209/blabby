-- Round N: PENDING_BLABBY / LOCAL ONLY. No production apply authorized.
-- Deploy/verify the bounded anonymous backend API before revoking raw reads.
-- Public browsing remains anonymous through the backend service_role client.
BEGIN;

ALTER TABLE public.vocabulary_items ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Anyone can read vocabulary_items" ON public.vocabulary_items;
REVOKE SELECT ON TABLE public.vocabulary_items FROM PUBLIC, anon, authenticated;

-- Table REVOKE does not remove independent column grants, including PUBLIC's.
DO $$
DECLARE columns text;
BEGIN
    SELECT string_agg(quote_ident(attname), ', ' ORDER BY attnum) INTO columns
      FROM pg_attribute
     WHERE attrelid = 'public.vocabulary_items'::regclass
       AND attnum > 0 AND NOT attisdropped;
    EXECUTE format('REVOKE SELECT (%s) ON TABLE public.vocabulary_items FROM PUBLIC, anon, authenticated', columns);
END $$;

GRANT SELECT ON TABLE public.vocabulary_items TO service_role;

-- Fail closed on unexpected inherited privileges; do not rewrite shared roles.
DO $$
DECLARE browser_role text;
BEGIN
    FOREACH browser_role IN ARRAY ARRAY['anon', 'authenticated'] LOOP
        IF has_table_privilege(browser_role, 'public.vocabulary_items', 'SELECT')
           OR has_any_column_privilege(browser_role, 'public.vocabulary_items', 'SELECT') THEN
            RAISE EXCEPTION 'Unexpected inherited vocabulary SELECT for %', browser_role;
        END IF;
    END LOOP;
    IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'service_role' AND rolbypassrls) THEN
        RAISE EXCEPTION 'Expected service_role BYPASSRLS is absent';
    END IF;
END $$;

-- user_vocabulary, review logs, and save_vocabulary_atomic are untouched.
COMMIT;
