-- LOCAL PENDING MIGRATION — NOT APPLIED / NOT AUTHORIZED FOR PRODUCTION.
-- Serialize both vocabulary save routes per owner, including Reading's optional
-- corpus insert. Reuse is_user_pro(uuid); never accept a client entitlement flag.
BEGIN;

DO $$
BEGIN
    IF to_regprocedure('public.is_user_pro(uuid)') IS NULL THEN
        RAISE EXCEPTION 'atomic vocabulary save requires canonical is_user_pro(uuid)';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint c
        WHERE c.conrelid = 'public.user_vocabulary'::regclass
          AND c.contype IN ('u', 'p')
          AND c.conkey = ARRAY[
              (SELECT attnum FROM pg_attribute WHERE attrelid = c.conrelid AND attname = 'user_id'),
              (SELECT attnum FROM pg_attribute WHERE attrelid = c.conrelid AND attname = 'vocabulary_item_id')
          ]::smallint[]
    ) THEN
        RAISE EXCEPTION 'atomic vocabulary save requires existing owner/item uniqueness';
    END IF;
END $$;

CREATE OR REPLACE FUNCTION public.save_vocabulary_atomic(
    p_user_id uuid,
    p_vocabulary_item_id uuid DEFAULT NULL,
    p_word text DEFAULT NULL,
    p_zh_meaning text DEFAULT '',
    p_source text DEFAULT 'manual_added',
    p_source_practice_record_id uuid DEFAULT NULL
)
RETURNS jsonb
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, pg_temp
AS $function$
DECLARE
    v_item_id uuid := p_vocabulary_item_id;
    v_saved_id uuid;
    v_word text;
    v_count bigint;
    v_limit CONSTANT integer := 30;
BEGIN
    -- After waiting for a lock, each SQL command needs a fresh committed
    -- snapshot. Fail closed if a caller changes the RPC transaction isolation.
    IF current_setting('transaction_isolation') <> 'read committed' THEN
        RAISE EXCEPTION 'atomic vocabulary save requires read committed isolation';
    END IF;
    IF p_user_id IS NULL OR
       ((p_vocabulary_item_id IS NULL) = (p_word IS NULL)) THEN
        RAISE EXCEPTION 'provide owner and exactly one item id or word';
    END IF;
    IF p_word IS NOT NULL AND
       (length(p_word) NOT BETWEEN 1 AND 60 OR p_word !~ '^[a-z''-]+$') THEN
        RAISE EXCEPTION 'invalid normalized vocabulary word';
    END IF;

    -- Same key across routes/workers; hash collisions only serialize extra owners.
    PERFORM pg_advisory_xact_lock(hashtextextended('blabby:vocabulary-save:' || p_user_id::text, 0));

    IF p_word IS NOT NULL THEN
        -- Prefer an already-owned spelling, even if the historical corpus has
        -- duplicate spellings. No new corpus-wide word uniqueness is imposed.
        SELECT uv.id, vi.id, vi.word INTO v_saved_id, v_item_id, v_word
        FROM public.user_vocabulary uv
        JOIN public.vocabulary_items vi ON vi.id = uv.vocabulary_item_id
        WHERE uv.user_id = p_user_id AND vi.word = p_word
        ORDER BY uv.id LIMIT 1;
        IF v_saved_id IS NULL THEN
            SELECT vi.id, vi.word INTO v_item_id, v_word
            FROM public.vocabulary_items vi WHERE vi.word = p_word
            ORDER BY vi.id LIMIT 1;
        END IF;
    ELSE
        SELECT uv.id INTO v_saved_id
        FROM public.user_vocabulary uv
        WHERE uv.user_id = p_user_id AND uv.vocabulary_item_id = v_item_id;
        SELECT vi.word INTO v_word FROM public.vocabulary_items vi WHERE vi.id = v_item_id;
    END IF;

    IF v_saved_id IS NOT NULL THEN
        RETURN jsonb_build_object('status', 'existing', 'user_vocabulary_id', v_saved_id,
                                  'vocabulary_item_id', v_item_id, 'word', v_word);
    END IF;

    IF NOT coalesce(public.is_user_pro(p_user_id), false) THEN
        SELECT count(*) INTO v_count FROM public.user_vocabulary WHERE user_id = p_user_id;
        IF v_count >= v_limit THEN
            RETURN jsonb_build_object('status', 'quota_reached', 'limit', v_limit);
        END IF;
    END IF;

    IF p_word IS NOT NULL AND v_item_id IS NULL THEN
        INSERT INTO public.vocabulary_items (word, zh_meaning)
        VALUES (p_word, left(coalesce(p_zh_meaning, ''), 30))
        RETURNING id, word INTO v_item_id, v_word;
    ELSIF v_word IS NULL THEN
        RETURN jsonb_build_object('status', 'not_found');
    END IF;

    INSERT INTO public.user_vocabulary (user_id, vocabulary_item_id, source, source_practice_record_id)
    VALUES (p_user_id, v_item_id, p_source, p_source_practice_record_id)
    RETURNING id INTO v_saved_id;

    RETURN jsonb_build_object('status', 'inserted', 'user_vocabulary_id', v_saved_id,
                              'vocabulary_item_id', v_item_id, 'word', v_word);
END;
$function$;

REVOKE ALL ON FUNCTION public.save_vocabulary_atomic(uuid, uuid, text, text, text, uuid)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.save_vocabulary_atomic(uuid, uuid, text, text, text, uuid)
    TO service_role;

-- New owned rows must enter through the function, including backend saves.
-- Preserve SELECT, review UPDATE/DELETE, and all corpus policies/privileges.
-- Remove column grants too: revoking only table INSERT would leave a bypass.
REVOKE INSERT ON public.user_vocabulary FROM PUBLIC, anon, authenticated, service_role;
DO $$
DECLARE
    v_columns text;
    v_role text;
BEGIN
    SELECT string_agg(quote_ident(attname), ', ' ORDER BY attnum) INTO v_columns
    FROM pg_attribute
    WHERE attrelid = 'public.user_vocabulary'::regclass AND attnum > 0 AND NOT attisdropped;
    EXECUTE format('REVOKE INSERT (%s) ON public.user_vocabulary FROM PUBLIC, anon, authenticated, service_role', v_columns);
    FOREACH v_role IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
        IF has_any_column_privilege(v_role, 'public.user_vocabulary', 'INSERT') THEN
            RAISE EXCEPTION 'unexpected inherited owned-vocabulary INSERT privilege for %', v_role;
        END IF;
    END LOOP;
END $$;

COMMIT;
