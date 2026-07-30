-- ============================================================================
-- ROLLBACK for 20260730_ecpay_backend.sql
--
-- Sections mirror the forward file and are INDEPENDENT — run only the ones you
-- need, in reverse order (§3 → §2 → §1) if reverting everything.
--
-- Non-destructive by default: subscriptions.merchant_trade_no is NOT dropped.
-- Once a single ECPay order has been created, that column is the only link
-- between a Blabby subscription and a real charge in the ECPay console;
-- dropping it destroys the audit trail. The DROP is provided but commented out.
--
-- ⚠ ORDER WITH THE CODE. §1 and §2 reverse the schema the backend depends on.
-- Roll the application back to a build without ECPay create-order/callback
-- BEFORE running §1 or §2, or the next callback will fail on a check violation
-- (§1) or an unknown column (§2).
-- ============================================================================


-- ── §3 rollback — paying_users back to the profiles.is_pro expression ───────
-- Verbatim restore of the definition from 20260515_pro_grant_expiry.sql:52-74.
-- Only meaningful alongside rolling 20260726 §2 back as well: with the
-- time-windowed is_user_pro() still live, this reports 0 paying users while
-- people are genuinely paying.

CREATE OR REPLACE FUNCTION public.get_admin_pro_breakdown()
RETURNS TABLE (
  total_pro_effective   BIGINT,
  paying_users          BIGINT,
  granted_users         BIGINT,
  both_paid_and_granted BIGINT
)
LANGUAGE SQL
SECURITY DEFINER
SET search_path = public
AS $$
  WITH grant_active AS (
    SELECT
      is_pro,
      (is_pro_grant AND (pro_grant_expires_at IS NULL OR pro_grant_expires_at > NOW())) AS is_grant_active
    FROM profiles
  )
  SELECT
    COUNT(*) FILTER (WHERE is_pro OR is_grant_active)        AS total_pro_effective,
    COUNT(*) FILTER (WHERE is_pro AND NOT is_grant_active)   AS paying_users,
    COUNT(*) FILTER (WHERE is_grant_active AND NOT is_pro)   AS granted_users,
    COUNT(*) FILTER (WHERE is_pro AND is_grant_active)       AS both_paid_and_granted
  FROM grant_active;
$$;

REVOKE EXECUTE ON FUNCTION public.get_admin_pro_breakdown() FROM public, anon, authenticated;
GRANT  EXECUTE ON FUNCTION public.get_admin_pro_breakdown() TO service_role;


-- ── §2 rollback — drop the uniqueness, keep the data ────────────────────────
DROP INDEX IF EXISTS public.subscriptions_merchant_trade_no_uniq;

-- Destructive. Only if you are certain no ECPay order was ever created:
--   ALTER TABLE public.subscriptions DROP COLUMN IF EXISTS merchant_trade_no;


-- ── §1 rollback — restore the original source whitelist ─────────────────────
-- ⚠ WILL FAIL if any payment_events row already has source='ecpay_callback',
-- which is correct: the constraint cannot be narrowed below the data. Check
-- first, and if rows exist, either leave §1 applied or reclassify them (the
-- immutability trigger forbids updating source, so reclassifying means a new
-- ledger row, not an edit).
--
--   SELECT count(*) FROM public.payment_events WHERE source = 'ecpay_callback';
--   → must be 0 before running the statements below.

ALTER TABLE public.payment_events
    DROP CONSTRAINT IF EXISTS payment_events_source_check;

ALTER TABLE public.payment_events
    ADD CONSTRAINT payment_events_source_check
    CHECK (source IN ('return_url', 'period_return_url',
                      'reconciliation', 'lemonsqueezy'));
