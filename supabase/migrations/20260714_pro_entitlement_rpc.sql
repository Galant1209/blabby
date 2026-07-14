-- Pro entitlement RPC — UNION grant-path OR paid-path (DRAFT, NOT executed).
--
-- Forward-only, idempotent (CREATE OR REPLACE). Does NOT drop profiles.is_pro
-- (kept as a bare boolean for one week of observation, then dropped separately).
-- Does NOT touch application code.
--
-- Change vs the current definition:
--   before: is_pro OR (is_pro_grant AND grant-not-expired)
--   after : (is_pro_grant AND grant-not-expired)          -- admin grant path
--        OR (active subscription with expires_at > now())  -- paid / billing path
--   i.e. the RPC STOPS reading the bare is_pro boolean and starts reading the
--   time-based subscriptions window (expires_at == current_period_end).
--
-- Migration-safety gate (verified 2026-07-14 against project mkwywkwruyqzdhuzwnoa):
--   All existing profiles have is_pro = false; nobody holds Pro via bare
--   is_pro=true with is_pro_grant=false. So dropping the is_pro read revokes
--   nobody. Grant-path users (is_pro_grant=true) are preserved unchanged.
--   subscriptions is currently empty, so the paid clause is a no-op today and
--   only starts granting once the billing flow writes active rows -> GO.
--
-- Notes on the signature (deliberate, do not "fix"):
--   * Parameter stays named user_id. CREATE OR REPLACE FUNCTION cannot rename
--     an input parameter, and the live RPC already uses user_id — renaming to
--     uid would error on replace.
--   * Because subscriptions has a column also named user_id, the parameter is
--     referenced as is_user_pro.user_id to force parameter (not column) binding.
--     Without this the paid clause could resolve user_id to the column and match
--     every row. This qualification is the correctness pivot of this migration.
--   * LANGUAGE sql / STABLE / SECURITY DEFINER / search_path='public' are copied
--     verbatim from the current definition.

CREATE OR REPLACE FUNCTION public.is_user_pro(user_id uuid)
 RETURNS boolean
 LANGUAGE sql
 STABLE SECURITY DEFINER
 SET search_path TO 'public'
AS $function$
  SELECT
       EXISTS (
         SELECT 1 FROM profiles p
         WHERE p.id = is_user_pro.user_id
           AND COALESCE(p.is_pro_grant, FALSE)
           AND (p.pro_grant_expires_at IS NULL OR p.pro_grant_expires_at > now())
       )
    OR EXISTS (
         SELECT 1 FROM subscriptions s
         WHERE s.user_id = is_user_pro.user_id
           AND s.status = 'active'
           AND s.expires_at > now()
       );
$function$;
