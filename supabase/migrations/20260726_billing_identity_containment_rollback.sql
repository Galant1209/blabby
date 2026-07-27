-- ============================================================================
-- ROLLBACK for 20260726_billing_identity_containment.sql
--
-- Sections mirror the forward file and are INDEPENDENT — run only the ones you
-- need, in reverse order (§5 → §4 → §3 → §2 → §1) if reverting everything.
--
-- Non-destructive by design: payment_events is NOT dropped by default. It is an
-- evidentiary ledger; if you have taken even one real callback, dropping it
-- destroys the only record of it. The DROP is provided but commented out.
-- ============================================================================


-- ── §5 rollback — restore anon EXECUTE ──────────────────────────────────────
GRANT EXECUTE ON FUNCTION public.is_user_pro(uuid) TO anon;

DO $$ BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'public' AND p.proname = 'handle_new_user'
    ) THEN
        EXECUTE 'GRANT EXECUTE ON FUNCTION public.handle_new_user() TO anon';
    END IF;
END $$;


-- ── §4 rollback — disable RLS on the three shared-content tables ────────────
-- Restores the pre-migration state exactly: RLS off, anon able to read.
DROP POLICY IF EXISTS questions_read_authenticated         ON public.questions;
DROP POLICY IF EXISTS writing_questions_read_authenticated ON public.writing_questions;
DROP POLICY IF EXISTS reading_passages_read_own            ON public.reading_passages;

ALTER TABLE public.questions         DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.reading_passages  DISABLE ROW LEVEL SECURITY;
ALTER TABLE public.writing_questions DISABLE ROW LEVEL SECURITY;

GRANT SELECT ON public.questions         TO anon, authenticated;
GRANT SELECT ON public.reading_passages  TO anon, authenticated;
GRANT SELECT ON public.writing_questions TO anon, authenticated;


-- ── §3 rollback — restore the previous views verbatim ───────────────────────
-- These are the pre-migration definitions, email column included. Restoring
-- them REOPENS the anon email exposure; only do so if a real consumer is found.
DROP VIEW IF EXISTS public.user_pro_status;

CREATE OR REPLACE VIEW public.user_pro_status AS
SELECT
  p.id,
  u.email,
  p.is_pro                          AS is_pro_paid,
  p.is_pro_grant,
  p.pro_grant_reason,
  p.pro_grant_at,
  p.pro_grant_by,
  p.pro_grant_expires_at,
  (
    COALESCE(p.is_pro, FALSE)
    OR (
      COALESCE(p.is_pro_grant, FALSE)
      AND (p.pro_grant_expires_at IS NULL OR p.pro_grant_expires_at > NOW())
    )
  )                                 AS is_pro_effective
FROM public.profiles p
JOIN auth.users u ON u.id = p.id;

-- public.user_lookup had no definition anywhere in the repo, so it cannot be
-- restored verbatim. If a consumer is discovered, recreate it from the
-- Supabase dashboard's stored definition BEFORE re-running §3 forward.


-- ── §2 rollback — restore the bare-boolean is_user_pro ──────────────────────
-- Reverts to 20260515_pro_grant_expiry.sql:22-28. Reinstates the "one write =
-- Pro forever" behaviour, so treat this as an emergency measure only.
CREATE OR REPLACE FUNCTION public.is_user_pro(user_id uuid)
 RETURNS boolean
 LANGUAGE sql
 STABLE SECURITY DEFINER
 SET search_path TO 'public'
AS $function$
  SELECT
    COALESCE(is_pro, FALSE)
    OR (
      COALESCE(is_pro_grant, FALSE)
      AND (pro_grant_expires_at IS NULL OR pro_grant_expires_at > NOW())
    )
  FROM profiles
  WHERE id = user_id;
$function$;


-- ── §1 rollback — payment_events ────────────────────────────────────────────
-- The trigger must go first; it forbids DELETE and would block any cleanup.
DROP TRIGGER  IF EXISTS payment_events_immutable_trg ON public.payment_events;
DROP FUNCTION IF EXISTS public.payment_events_immutable();

-- Deliberately commented out. Dropping this table destroys the audit trail for
-- every payment ever received. Uncomment ONLY if no real callback has landed.
--   DROP TABLE IF EXISTS public.payment_events;
--
-- Reverting §1 while leaving the table in place is the safe default: the
-- backend simply stops writing to it once the application code is rolled back.
