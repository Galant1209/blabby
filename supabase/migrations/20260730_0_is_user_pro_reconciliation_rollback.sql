-- ############################################################################
-- ##  WARNING — ENTITLEMENT-CRITICAL ROLLBACK                              ##
-- ##                                                                        ##
-- ##  This reverts is_user_pro(uuid) to reading the bare profiles.is_pro    ##
-- ##  flag instead of the time-windowed grant/subscription definition.      ##
-- ##                                                                        ##
-- ##  If any subscription is currently status='active' and expires_at in    ##
-- ##  the future, that user is Pro ONLY because of the clause this rollback ##
-- ##  removes. Running it would silently revoke Pro from a paying customer  ##
-- ##  mid-subscription — the exact failure this file refuses to cause.      ##
-- ##                                                                        ##
-- ##  This is why, unlike 20260731's rollback, THIS ONE HAS A GATE: it      ##
-- ##  queries for active, unexpired subscriptions before touching anything, ##
-- ##  and ABORTS if any exist. There is no override. If you must revert     ##
-- ##  anyway, that is a product decision to knowingly downgrade paying      ##
-- ##  customers — make it explicitly, outside this file.                    ##
-- ############################################################################
--
-- ROLLBACK for 20260730_0_is_user_pro_reconciliation.sql
--
-- Single object changed forward, single object reverted here: only
-- public.is_user_pro(uuid). No table, column, row, grant or policy created by
-- the forward file exists outside that one function — there is nothing else
-- to undo.
-- ============================================================================

BEGIN;

DO $rollback_gate$
DECLARE
    at_risk integer;
BEGIN
    SELECT count(*) INTO at_risk
      FROM public.subscriptions s
     WHERE s.status = 'active' AND s.expires_at > now();

    IF at_risk > 0 THEN
        RAISE EXCEPTION
            'is_user_pro rollback REFUSED: % active, unexpired subscription(s) '
            'exist. Reverting to the bare is_pro body would silently revoke Pro '
            'from % paying customer(s) still inside their paid period. No '
            'override exists in this file — reverting anyway is a product '
            'decision to downgrade them, and must be made explicitly, not by '
            'running a schema rollback.', at_risk, at_risk;
    END IF;
END
$rollback_gate$;

-- Verbatim restore of the pre-reconciliation body
-- (source: production, captured 2026-07-30 before this migration applied).
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
      AND (pro_grant_expires_at IS NULL OR pro_grant_expires_at > now())
    )
  FROM profiles
  WHERE id = user_id;
$function$;

REVOKE EXECUTE ON FUNCTION public.is_user_pro(uuid) FROM PUBLIC, anon;

COMMIT;


-- ── verification after rollback ─────────────────────────────────────────────
-- SELECT prosrc FROM pg_proc WHERE oid = to_regprocedure('public.is_user_pro(uuid)');
--   → does NOT contain "FROM subscriptions s"; contains "COALESCE(is_pro, FALSE)"
