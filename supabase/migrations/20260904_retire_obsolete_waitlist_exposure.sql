-- Retire the pre-ECPay anonymous waitlist write path without deleting its
-- historical rows. The application no longer accepts waitlist submissions;
-- admin user cleanup and legacy reporting RPCs may still need the data.
--
-- The table was historically created outside the ordered migration set in
-- some environments, so every operation is guarded for a missing relation.
DO $$
BEGIN
  IF to_regclass('public.upgrade_intent') IS NULL THEN
    RETURN;
  END IF;

  EXECUTE 'DROP POLICY IF EXISTS allow_anon_insert ON public.upgrade_intent';
  EXECUTE 'DROP POLICY IF EXISTS anon_insert_upgrade_intent ON public.upgrade_intent';
  EXECUTE 'DROP POLICY IF EXISTS authenticated_insert_upgrade_intent ON public.upgrade_intent';
  EXECUTE 'REVOKE INSERT ON TABLE public.upgrade_intent FROM anon';
END
$$;
