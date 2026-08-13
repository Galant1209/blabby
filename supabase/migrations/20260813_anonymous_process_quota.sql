-- Durable, privacy-safe quota for the temporary public Speaking Part 1 trial.
-- Stores only HMAC-SHA256 identifiers produced by the backend. It does not
-- create anonymous Auth users, profiles, practice history, or entitlements.

BEGIN;

CREATE TABLE IF NOT EXISTS public.anonymous_process_usage (
    visitor_hash text PRIMARY KEY CHECK (visitor_hash ~ '^[0-9a-f]{64}$'),
    ip_hash text NOT NULL CHECK (ip_hash ~ '^[0-9a-f]{64}$'),
    used integer NOT NULL DEFAULT 0 CHECK (used >= 0),
    updated_at timestamptz NOT NULL DEFAULT now()
);

ALTER TABLE public.anonymous_process_usage ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON public.anonymous_process_usage FROM PUBLIC, anon, authenticated;

CREATE OR REPLACE FUNCTION public.get_anonymous_process_quota(
    p_visitor_hash text,
    p_ip_hash text,
    p_limit integer DEFAULT 10
)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $function$
DECLARE
    visitor_used integer := 0;
BEGIN
    IF p_visitor_hash !~ '^[0-9a-f]{64}$'
       OR p_ip_hash !~ '^[0-9a-f]{64}$'
       OR p_limit < 1 THEN
        RAISE EXCEPTION 'invalid anonymous quota input';
    END IF;

    SELECT used INTO visitor_used
      FROM public.anonymous_process_usage
     WHERE visitor_hash = p_visitor_hash;
    visitor_used := COALESCE(visitor_used, 0);
    RETURN jsonb_build_object(
        'allowed', visitor_used < p_limit,
        'used', LEAST(visitor_used, p_limit),
        'remaining', GREATEST(p_limit - visitor_used, 0)
    );
END;
$function$;

CREATE OR REPLACE FUNCTION public.consume_anonymous_process_quota(
    p_visitor_hash text,
    p_ip_hash text,
    p_limit integer DEFAULT 10
)
RETURNS jsonb
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $function$
DECLARE
    visitor_used integer;
BEGIN
    IF p_visitor_hash !~ '^[0-9a-f]{64}$'
       OR p_ip_hash !~ '^[0-9a-f]{64}$'
       OR p_limit < 1 THEN
        RAISE EXCEPTION 'invalid anonymous quota input';
    END IF;

    INSERT INTO public.anonymous_process_usage (visitor_hash, ip_hash)
    VALUES (p_visitor_hash, p_ip_hash)
    ON CONFLICT (visitor_hash) DO NOTHING;

    SELECT used INTO visitor_used
      FROM public.anonymous_process_usage
     WHERE visitor_hash = p_visitor_hash
     FOR UPDATE;

    IF visitor_used >= p_limit THEN
        RETURN jsonb_build_object(
            'allowed', false,
            'used', p_limit,
            'remaining', 0
        );
    END IF;

    UPDATE public.anonymous_process_usage
       SET used = used + 1, ip_hash = p_ip_hash, updated_at = now()
     WHERE visitor_hash = p_visitor_hash;

    visitor_used := visitor_used + 1;
    RETURN jsonb_build_object(
        'allowed', true,
        'used', visitor_used,
        'remaining', GREATEST(p_limit - visitor_used, 0)
    );
END;
$function$;

REVOKE ALL ON FUNCTION public.get_anonymous_process_quota(text, text, integer)
    FROM PUBLIC, anon, authenticated;
REVOKE ALL ON FUNCTION public.consume_anonymous_process_quota(text, text, integer)
    FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.get_anonymous_process_quota(text, text, integer)
    TO service_role;
GRANT EXECUTE ON FUNCTION public.consume_anonymous_process_quota(text, text, integer)
    TO service_role;

COMMIT;
