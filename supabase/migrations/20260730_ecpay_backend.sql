-- ============================================================================
-- 20260730_ecpay_backend.sql
--
-- The schema delta the ECPay backend (feat/ecpay-backend) needs before it can
-- take a single real payment:
--   §1  payment_events.source     — allow 'ecpay_callback' (HARD BLOCKER)
--   §2  subscriptions.merchant_trade_no — the idempotency anchor, UNIQUE
--   §3  get_admin_pro_breakdown() — count paying users from subscriptions
--   §4  subscriptions.ecpay_trade_no — ECPay's own transaction number
--
-- ── TWO PRODUCT DECISIONS THIS SCHEMA ENCODES (2026-07-30) ──────────────────
-- Both are deliberate choices, not features awaiting implementation. Changing
-- either must be an explicit product decision, never a "while we're here".
--
-- 1. NO AUTO-RENEWAL. Blabby Pro is a 30-day period the buyer chooses to
--    purchase, every time. No 定期定額, no recurring mandate, and no reserved
--    columns for one. A product whose red lines forbid retaining users through
--    gamification does not get to retain them through "forgot to cancel"
--    either; IELTS is a 1-3 month use case where subscription LTV is thin; and
--    it removes cancellation flows, dunning, involuntary churn and chargeback
--    disputes outright.
--
--    Consequence for this file: payment_events.total_success_times stays in the
--    idempotency key and stays nullable. Under one-off billing it is always
--    NULL, which is exactly why the unique index needs NULLS NOT DISTINCT — a
--    resent callback must collide rather than insert a second row.
--
-- 2. NO E-INVOICE INTEGRATION IN PHASE 1. The company is obliged to issue 統一
--    發票, but with 0 paying users the first few are issued by hand. No
--    InvoiceMark and no other invoice parameter is sent, because adding one
--    changes the signed AioCheckOut parameter set that the known-answer vectors
--    currently validate. Integrating later REQUIRES re-running that validation.
--
-- Properties: forward-only, idempotent (safe to re-run), non-destructive
-- (drops no table, drops no column, deletes no row). Rollback lives in
-- 20260730_ecpay_backend_rollback.sql, section-for-section.
--
-- NOT YET EXECUTED against production.
--
-- ── DEPENDENCY ──────────────────────────────────────────────────────────────
-- This file deliberately does NOT redefine is_user_pro(). That change already
-- exists, unapplied, as §2 of 20260726_billing_identity_containment.sql — the
-- same UNION-of-time-windows body drafted on origin/feat/billing-schema
-- (9960654), plus a pre-flight gate that ABORTS rather than silently revoking
-- anyone. Writing a second definition here would leave two competing sources of
-- truth for the one function that decides who is Pro.
--
-- Required order:  20260726 §2   →   this file §3
-- §1 and §2 of this file are independent of 20260726 and may run at any time.
--
-- §3 stops counting profiles.is_pro, and 20260726 §2 stops reading it for
-- entitlement. Applying §3 before 20260726 §2 would leave the admin dashboard
-- and the actual paywall disagreeing about who has paid.
--
-- ── MIGRATION GATE (verified 2026-07-30, read-only) ─────────────────────────
--   profiles          = 17 rows,  is_pro = true on 0 of them
--   is_pro_grant      = true on 7 (all reachable through the grant clause)
--   subscriptions     = 0 rows
--   payment_events    = 0 rows, payment_events_idem_uniq NULLS NOT DISTINCT live
-- Nobody holds Pro through the bare is_pro boolean, so dropping that read
-- revokes nobody, and §1's constraint swap cannot conflict with existing data.
-- profiles.is_pro is NOT dropped by this file.
-- ============================================================================


-- ────────────────────────────────────────────────────────────────────────────
-- §1  payment_events.source — admit 'ecpay_callback'
--
-- HARD BLOCKER, and the reason this file exists. §1 of 20260726 shipped
-- payment_events with
--     CHECK (source IN ('return_url','period_return_url','reconciliation',
--                       'lemonsqueezy'))
-- and that constraint IS live in production. The backend now writes
-- 'ecpay_callback', because the row records the signed server-to-server
-- notification, not the browser-facing return URL that 'return_url' names.
-- Without this section every genuine callback fails on a 23514 check violation,
-- which the handler surfaces as a 500 — so ECPay would retry, fail, and give
-- up, and no payment would ever grant anything.
--
-- 'return_url' is KEPT in the allowed set: production has 0 rows today, but
-- keeping it means this section stays non-destructive even if a row lands
-- between now and execution.
-- ────────────────────────────────────────────────────────────────────────────

ALTER TABLE public.payment_events
    DROP CONSTRAINT IF EXISTS payment_events_source_check;

ALTER TABLE public.payment_events
    ADD CONSTRAINT payment_events_source_check
    CHECK (source IN ('ecpay_callback',      -- signed server-to-server callback
                      'return_url',          -- legacy name, retained for old rows
                      'period_return_url',   -- vestigial: from the abandoned
                                             -- recurring draft. Kept only
                                             -- because narrowing a live CHECK
                                             -- is destructive; auto-renewal is
                                             -- decided against, see header.
                      'reconciliation',      -- manual/scheduled repair
                      'lemonsqueezy'));


-- ────────────────────────────────────────────────────────────────────────────
-- §2  subscriptions.merchant_trade_no
--
-- The callback locates the subscription by MerchantTradeNo, not by the legacy
-- order_id, and the UNIQUE constraint is half the idempotency story: the ledger
-- key stops a replayed callback from re-granting, and this stops two rows from
-- ever claiming the same trade number in the first place.
--
-- NULL is allowed and the index is partial: LemonSqueezy rows and the pre-ECPay
-- rows have no trade number, and multiple NULLs must not collide. A partial
-- unique index rather than a table constraint is what makes that work.
-- ────────────────────────────────────────────────────────────────────────────

ALTER TABLE public.subscriptions
    ADD COLUMN IF NOT EXISTS merchant_trade_no text;

CREATE UNIQUE INDEX IF NOT EXISTS subscriptions_merchant_trade_no_uniq
    ON public.subscriptions (merchant_trade_no)
    WHERE merchant_trade_no IS NOT NULL;

COMMENT ON COLUMN public.subscriptions.merchant_trade_no IS
    'ECPay MerchantTradeNo for this order (BLB + yymmddHHMM + 7 alphanumeric, '
    '20 chars). The callback matches on this column. UNIQUE where not null; '
    'NULL for LemonSqueezy and pre-ECPay rows.';


-- ────────────────────────────────────────────────────────────────────────────
-- §3  get_admin_pro_breakdown() — paying_users from subscriptions
--
-- REQUIRES 20260726 §2 to have run first (see DEPENDENCY above).
--
--   before: paying_users = COUNT(profiles WHERE is_pro AND NOT grant_active)
--   after : paying_users = COUNT(profiles WITH an active, unexpired subscription)
--
-- The backend no longer writes profiles.is_pro on a successful payment — Pro is
-- the subscriptions window — so the old expression would report 0 paying users
-- forever, no matter how much money came in. Same four output columns, same
-- name and signature, so /api/admin/* needs no code change.
--
-- Column is not dropped: profiles.is_pro still holds the LemonSqueezy paid flag
-- and stays readable for reconciliation.
-- ────────────────────────────────────────────────────────────────────────────

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
  WITH state AS (
    SELECT
      (COALESCE(p.is_pro_grant, FALSE)
       AND (p.pro_grant_expires_at IS NULL OR p.pro_grant_expires_at > NOW()))
        AS is_grant_active,
      EXISTS (
        SELECT 1 FROM subscriptions s
        WHERE s.user_id = p.id
          AND s.status = 'active'
          AND s.expires_at > NOW()
      ) AS is_paying
    FROM profiles p
  )
  SELECT
    COUNT(*) FILTER (WHERE is_paying OR is_grant_active)      AS total_pro_effective,
    COUNT(*) FILTER (WHERE is_paying AND NOT is_grant_active) AS paying_users,
    COUNT(*) FILTER (WHERE is_grant_active AND NOT is_paying) AS granted_users,
    COUNT(*) FILTER (WHERE is_paying AND is_grant_active)     AS both_paid_and_granted
  FROM state;
$$;

-- Re-assert the grants: CREATE OR REPLACE keeps them, but 20260515 set them
-- explicitly and re-stating makes this file safe to run against a fresh DB.
REVOKE EXECUTE ON FUNCTION public.get_admin_pro_breakdown() FROM public, anon, authenticated;
GRANT  EXECUTE ON FUNCTION public.get_admin_pro_breakdown() TO service_role;


-- ────────────────────────────────────────────────────────────────────────────
-- §4  subscriptions.ecpay_trade_no
--
-- ECPay's own transaction number (TradeNo, String(20)), returned on every
-- callback. It is the only identifier their support desk recognises, so the
-- mapping from our MerchantTradeNo to theirs has to be queryable rather than
-- buried inside payment_events.raw_payload.
--
-- Not unique and not indexed: it is a lookup aid used by hand a few times a
-- year, and at this volume a sequential scan is free.
-- ────────────────────────────────────────────────────────────────────────────

ALTER TABLE public.subscriptions
    ADD COLUMN IF NOT EXISTS ecpay_trade_no text;

COMMENT ON COLUMN public.subscriptions.ecpay_trade_no IS
    'ECPay TradeNo from the payment callback. The shared vocabulary for any '
    'conversation with ECPay support; NULL for LemonSqueezy and unpaid orders.';


-- ────────────────────────────────────────────────────────────────────────────
-- §5  accept_ecpay_payment() — one transaction for ledger + entitlement
--
-- The callback performs read-only trust validation first. This function repeats
-- the order identity and amount checks while holding a row lock, then inserts
-- the accepted-success ledger row and activates the subscription in the same
-- PostgreSQL transaction. A process crash cannot leave only one of those writes.
--
-- Rejected callbacks never call this function, so they cannot consume the
-- accepted-success idempotency key and poison a later valid retry.
-- ────────────────────────────────────────────────────────────────────────────

CREATE OR REPLACE FUNCTION public.accept_ecpay_payment(
    p_merchant_trade_no   text,
    p_total_success_times integer,
    p_rtn_code            text,
    p_rtn_msg             text,
    p_raw_payload         jsonb,
    p_ecpay_trade_no      text,
    p_expected_user_id    uuid,
    p_expected_amount     integer,
    p_started_at          timestamptz,
    p_expires_at          timestamptz
)
RETURNS text
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public
AS $$
DECLARE
    target public.subscriptions%ROWTYPE;
    event_id uuid;
BEGIN
    SELECT *
      INTO target
      FROM public.subscriptions
     WHERE merchant_trade_no = p_merchant_trade_no
     FOR UPDATE;

    IF NOT FOUND
       OR target.user_id IS DISTINCT FROM p_expected_user_id
       OR target.amount IS DISTINCT FROM p_expected_amount THEN
        RETURN 'rejected';
    END IF;

    -- A duplicate success must not restart or extend the purchased period.
    IF target.status = 'active' THEN
        RETURN 'duplicate';
    END IF;
    IF target.status IS DISTINCT FROM 'pending' THEN
        RETURN 'rejected';
    END IF;

    INSERT INTO public.payment_events (
        source,
        merchant_trade_no,
        total_success_times,
        rtn_code,
        rtn_msg,
        checkmac_valid,
        user_id,
        subscription_id,
        raw_payload,
        processed_at
    ) VALUES (
        'ecpay_callback',
        p_merchant_trade_no,
        p_total_success_times,
        p_rtn_code,
        p_rtn_msg,
        TRUE,
        target.user_id,
        target.id,
        p_raw_payload,
        now()
    )
    ON CONFLICT (merchant_trade_no, total_success_times, source) DO NOTHING
    RETURNING id INTO event_id;

    IF event_id IS NULL THEN
        RETURN 'duplicate';
    END IF;

    UPDATE public.subscriptions
       SET status = 'active',
           started_at = p_started_at,
           expires_at = p_expires_at,
           updated_at = p_started_at,
           ecpay_trade_no = p_ecpay_trade_no
     WHERE id = target.id;

    RETURN 'activated';
END;
$$;

REVOKE EXECUTE ON FUNCTION public.accept_ecpay_payment(
    text, integer, text, text, jsonb, text, uuid, integer, timestamptz, timestamptz
) FROM public, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.accept_ecpay_payment(
    text, integer, text, text, jsonb, text, uuid, integer, timestamptz, timestamptz
) TO service_role;


-- ── verification (run after applying; all five must hold) ───────────────────
-- 1. SELECT conname, pg_get_constraintdef(oid) FROM pg_constraint
--     WHERE conrelid = 'public.payment_events'::regclass
--       AND conname  = 'payment_events_source_check';
--    → definition contains 'ecpay_callback'
--
-- 2. SELECT column_name FROM information_schema.columns
--     WHERE table_name = 'subscriptions' AND column_name = 'merchant_trade_no';
--    → 1 row
--
-- 3. SELECT indexdef FROM pg_indexes
--     WHERE indexname = 'subscriptions_merchant_trade_no_uniq';
--    → UNIQUE ... WHERE (merchant_trade_no IS NOT NULL)
--
-- 4. SELECT * FROM get_admin_pro_breakdown();
--    → granted_users = 7, paying_users = 0, total_pro_effective = 7
--      (matches the 2026-07-30 gate above; any other figure means stop)
--
-- 5. SELECT column_name FROM information_schema.columns
--     WHERE table_name = 'subscriptions' AND column_name = 'ecpay_trade_no';
--    → 1 row
