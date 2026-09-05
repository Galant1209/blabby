-- LOCAL PENDING ONLY. No production apply/backfill authorization.
-- Publication controls discovery, not user_vocabulary ownership or save quota.
BEGIN;

ALTER TABLE public.vocabulary_items
    ADD COLUMN IF NOT EXISTS is_public boolean NOT NULL DEFAULT false;

-- Refuse incompatible pre-existing/manual schema instead of guessing intent.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_attribute a
        JOIN pg_attrdef d ON d.adrelid = a.attrelid AND d.adnum = a.attnum
        WHERE a.attrelid = 'public.vocabulary_items'::regclass
          AND a.attname = 'is_public' AND NOT a.attisdropped
          AND a.atttypid = 'boolean'::regtype AND a.attnotnull
          AND pg_get_expr(d.adbin, d.adrelid) = 'false'
    ) THEN
        RAISE EXCEPTION 'incompatible vocabulary publication column';
    END IF;
END $$;

COMMENT ON COLUMN public.vocabulary_items.is_public IS
    'Explicit trusted publication decision. False means unpublished, not owner-private. No automatic seed/generated approval.';

-- Browser roles cannot self-publish, even if a permissive write policy exists.
-- Remove table and independent column grants; preserve service-role and owned ACLs.
REVOKE INSERT, UPDATE ON TABLE public.vocabulary_items FROM PUBLIC, anon, authenticated;
DO $$
DECLARE
    columns text;
    browser_role text;
BEGIN
    SELECT string_agg(quote_ident(attname), ', ' ORDER BY attnum) INTO columns
    FROM pg_attribute WHERE attrelid = 'public.vocabulary_items'::regclass
        AND attnum > 0 AND NOT attisdropped;
    EXECUTE format('REVOKE INSERT (%s), UPDATE (%s) ON TABLE public.vocabulary_items FROM PUBLIC, anon, authenticated', columns, columns);
    FOREACH browser_role IN ARRAY ARRAY['anon', 'authenticated'] LOOP
        IF has_any_column_privilege(browser_role, 'public.vocabulary_items', 'INSERT')
           OR has_any_column_privilege(browser_role, 'public.vocabulary_items', 'UPDATE') THEN
            RAISE EXCEPTION 'unexpected inherited vocabulary mutation privilege: %', browser_role;
        END IF;
    END LOOP;
END $$;

-- Deliberately NO publication backfill. Existing/seed/source-unknown rows stay
-- false until a separately reviewed exact-ID/content-hash publication decision.
-- No index for the historical 83-row corpus without query-plan justification.
COMMIT;
