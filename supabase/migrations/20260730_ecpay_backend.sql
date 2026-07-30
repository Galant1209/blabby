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

BEGIN;

-- ── EXECUTABLE PRE-MUTATION GATE ────────────────────────────────────────────
-- Strategy A: this migration is safe to re-run. The gate accepts either the
-- exact 20260726 dependency state or the exact state produced by this file.
-- Unknown overloads, partially-created columns/indexes, or dependency drift
-- abort the transaction before the first schema mutation.
DO $preflight$
DECLARE
    expected_name text;
    expected_type text;
    expected_not_null boolean;
    actual_type text;
    actual_not_null boolean;
    source_values text[];
    source_constraint_count integer;
    fn_oid oid;
    fn_body text;
    expected_is_user_pro_body text;
    expected_accept_signature regprocedure;
BEGIN
    -- Dependency relations must be ordinary/partitioned tables in public.
    IF NOT EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relname = 'profiles'
          AND c.relkind IN ('r', 'p')
    ) THEN
        RAISE EXCEPTION 'ECPay preflight: required table public.profiles is missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relname = 'subscriptions'
          AND c.relkind IN ('r', 'p')
    ) THEN
        RAISE EXCEPTION 'ECPay preflight: required table public.subscriptions is missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relname = 'payment_events'
          AND c.relkind IN ('r', 'p')
    ) THEN
        RAISE EXCEPTION 'ECPay preflight: required table public.payment_events is missing';
    END IF;

    -- The columns this migration and its RPC depend on must have the exact
    -- types/nullability created by the repository migrations. Extra unrelated
    -- columns are allowed.
    FOR expected_name, expected_type, expected_not_null IN
        SELECT * FROM (VALUES
            ('id',         'uuid',                     true),
            ('user_id',    'uuid',                     true),
            ('order_id',   'text',                     false),
            ('plan',       'text',                     true),
            ('status',     'text',                     true),
            ('amount',     'integer',                  false),
            ('started_at', 'timestamp with time zone', false),
            ('expires_at', 'timestamp with time zone', false),
            ('created_at', 'timestamp with time zone', false),
            ('updated_at', 'timestamp with time zone', false)
        ) AS required(name, type_name, not_null)
    LOOP
        SELECT format_type(a.atttypid, a.atttypmod), a.attnotnull
          INTO actual_type, actual_not_null
          FROM pg_attribute a
         WHERE a.attrelid = 'public.subscriptions'::regclass
           AND a.attname = expected_name
           AND a.attnum > 0
           AND NOT a.attisdropped;
        IF NOT FOUND THEN
            RAISE EXCEPTION
                'ECPay preflight: subscriptions.% is missing', expected_name;
        END IF;
        IF actual_type <> expected_type OR actual_not_null <> expected_not_null THEN
            RAISE EXCEPTION
                'ECPay preflight: subscriptions.% has type/nullability %/%, expected %/%',
                expected_name, actual_type, actual_not_null,
                expected_type, expected_not_null;
        END IF;
    END LOOP;

    FOR expected_name, expected_type, expected_not_null IN
        SELECT * FROM (VALUES
            ('id',                  'uuid',                     true),
            ('received_at',         'timestamp with time zone', true),
            ('source',              'text',                     true),
            ('merchant_trade_no',   'text',                     false),
            ('total_success_times', 'integer',                  false),
            ('rtn_code',            'text',                     false),
            ('rtn_msg',             'text',                     false),
            ('checkmac_valid',      'boolean',                  true),
            ('user_id',             'uuid',                     false),
            ('subscription_id',     'uuid',                     false),
            ('raw_payload',         'jsonb',                    true),
            ('processed_at',        'timestamp with time zone', false)
        ) AS required(name, type_name, not_null)
    LOOP
        SELECT format_type(a.atttypid, a.atttypmod), a.attnotnull
          INTO actual_type, actual_not_null
          FROM pg_attribute a
         WHERE a.attrelid = 'public.payment_events'::regclass
           AND a.attname = expected_name
           AND a.attnum > 0
           AND NOT a.attisdropped;
        IF NOT FOUND THEN
            RAISE EXCEPTION
                'ECPay preflight: payment_events.% is missing', expected_name;
        END IF;
        IF actual_type <> expected_type OR actual_not_null <> expected_not_null THEN
            RAISE EXCEPTION
                'ECPay preflight: payment_events.% has type/nullability %/%, expected %/%',
                expected_name, actual_type, actual_not_null,
                expected_type, expected_not_null;
        END IF;
    END LOOP;

    -- Exactly one validated source CHECK must exist under the repository name.
    -- Extracting its string literals avoids depending on pg_get_constraintdef's
    -- whitespace/cast formatting across supported PostgreSQL versions.
    SELECT count(*)
      INTO source_constraint_count
      FROM pg_constraint c
     WHERE c.conrelid = 'public.payment_events'::regclass
       AND c.conname = 'payment_events_source_check'
       AND c.contype = 'c'
       AND c.convalidated;
    IF source_constraint_count <> 1 THEN
        RAISE EXCEPTION
            'ECPay preflight: expected one validated payment_events_source_check, found %',
            source_constraint_count;
    END IF;

    SELECT array_agg(match[1] ORDER BY match[1])
      INTO source_values
      FROM pg_constraint c
      CROSS JOIN LATERAL regexp_matches(
          pg_get_constraintdef(c.oid), $regex$'([^']+)'$regex$, 'g'
      ) AS match
     WHERE c.conrelid = 'public.payment_events'::regclass
       AND c.conname = 'payment_events_source_check';
    IF source_values <> ARRAY[
        'lemonsqueezy', 'period_return_url', 'reconciliation', 'return_url'
    ]::text[]
       AND source_values <> ARRAY[
        'ecpay_callback', 'lemonsqueezy', 'period_return_url',
        'reconciliation', 'return_url'
    ]::text[] THEN
        RAISE EXCEPTION
            'ECPay preflight: payment_events_source_check has unknown allowed values';
    END IF;

    -- The 20260726 append-only trigger must be enabled and attached to the
    -- expected function for BEFORE UPDATE OR DELETE.
    IF NOT EXISTS (
        SELECT 1
          FROM pg_trigger t
          JOIN pg_proc p ON p.oid = t.tgfoid
          JOIN pg_namespace n ON n.oid = p.pronamespace
         WHERE t.tgrelid = 'public.payment_events'::regclass
           AND t.tgname = 'payment_events_immutable_trg'
           AND NOT t.tgisinternal
           AND t.tgenabled IN ('O', 'A')
           AND t.tgtype = 27  -- ROW + BEFORE + DELETE + UPDATE
           AND n.nspname = 'public'
           AND p.proname = 'payment_events_immutable'
    ) THEN
        RAISE EXCEPTION
            'ECPay preflight: payment_events immutable trigger is missing or incompatible';
    END IF;

    -- The replay key must keep NULLS NOT DISTINCT. Validate structure through
    -- pg_index rather than trusting only the index name.
    IF NOT EXISTS (
        SELECT 1
          FROM pg_index i
          JOIN pg_class idx ON idx.oid = i.indexrelid
         WHERE i.indrelid = 'public.payment_events'::regclass
           AND idx.relnamespace = 'public'::regnamespace
           AND idx.relname = 'payment_events_idem_uniq'
           AND i.indisunique
           AND i.indisvalid
           AND i.indisready
           AND i.indnullsnotdistinct
           AND i.indpred IS NULL
           AND pg_get_indexdef(i.indexrelid) ILIKE
               '%(merchant_trade_no, total_success_times, source)%'
    ) THEN
        RAISE EXCEPTION
            'ECPay preflight: payment_events_idem_uniq is missing or incompatible';
    END IF;

    -- Exact dependency body: active grant OR active, unexpired subscription,
    -- never the legacy bare profiles.is_pro flag.
    fn_oid := to_regprocedure('public.is_user_pro(uuid)');
    IF fn_oid IS NULL THEN
        RAISE EXCEPTION
            'ECPay preflight: dependency function public.is_user_pro(uuid) is missing';
    END IF;
    SELECT regexp_replace(lower(p.prosrc), '\s+', ' ', 'g')
      INTO fn_body
      FROM pg_proc p
     WHERE p.oid = fn_oid
       AND p.prosecdef
       AND p.provolatile = 's'
       AND p.proconfig @> ARRAY['search_path=public'];
    expected_is_user_pro_body := regexp_replace(lower($expected$
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
    $expected$), '\s+', ' ', 'g');
    IF fn_body IS NULL OR btrim(fn_body) <> btrim(expected_is_user_pro_body) THEN
        RAISE EXCEPTION
            'ECPay preflight: public.is_user_pro(uuid) is not the expected 20260726 definition';
    END IF;

    -- Mechanical entitlement gate. Do not disclose identities in the error.
    IF EXISTS (
        SELECT 1
          FROM public.profiles p
         WHERE p.is_pro IS TRUE
           AND NOT (
                p.is_pro_grant IS TRUE
                AND (p.pro_grant_expires_at IS NULL
                     OR p.pro_grant_expires_at > now())
           )
           AND NOT EXISTS (
                SELECT 1
                  FROM public.subscriptions s
                 WHERE s.user_id = p.id
                   AND s.status = 'active'
                   AND s.expires_at > now()
           )
    ) THEN
        RAISE EXCEPTION
            'ECPay preflight: bare profiles.is_pro entitlement would be lost';
    END IF;

    -- A pre-existing column is accepted only when it exactly matches this
    -- migration's nullable text design and its unique partial index. This makes
    -- a complete rerun safe while rejecting partial/manual applications.
    IF EXISTS (
        SELECT 1 FROM pg_attribute
         WHERE attrelid = 'public.subscriptions'::regclass
           AND attname = 'merchant_trade_no'
           AND attnum > 0 AND NOT attisdropped
    ) THEN
        SELECT format_type(a.atttypid, a.atttypmod), a.attnotnull
          INTO actual_type, actual_not_null
          FROM pg_attribute a
         WHERE a.attrelid = 'public.subscriptions'::regclass
           AND a.attname = 'merchant_trade_no'
           AND a.attnum > 0 AND NOT a.attisdropped;
        IF actual_type <> 'text' OR actual_not_null THEN
            RAISE EXCEPTION
                'ECPay preflight: subscriptions.merchant_trade_no is incompatible';
        END IF;
        IF NOT EXISTS (
            SELECT 1
              FROM pg_index i
              JOIN pg_class idx ON idx.oid = i.indexrelid
             WHERE i.indrelid = 'public.subscriptions'::regclass
               AND idx.relnamespace = 'public'::regnamespace
               AND idx.relname = 'subscriptions_merchant_trade_no_uniq'
               AND i.indisunique AND i.indisvalid AND i.indisready
               AND i.indpred IS NOT NULL
               AND pg_get_indexdef(i.indexrelid) ILIKE
                   '%(merchant_trade_no) WHERE (merchant_trade_no IS NOT NULL)%'
        ) THEN
            RAISE EXCEPTION
                'ECPay preflight: existing merchant_trade_no lacks the expected unique partial index';
        END IF;
    ELSIF to_regclass('public.subscriptions_merchant_trade_no_uniq') IS NOT NULL THEN
        RAISE EXCEPTION
            'ECPay preflight: merchant trade index exists without its column';
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_attribute
         WHERE attrelid = 'public.subscriptions'::regclass
           AND attname = 'ecpay_trade_no'
           AND attnum > 0 AND NOT attisdropped
    ) THEN
        SELECT format_type(a.atttypid, a.atttypmod), a.attnotnull
          INTO actual_type, actual_not_null
          FROM pg_attribute a
         WHERE a.attrelid = 'public.subscriptions'::regclass
           AND a.attname = 'ecpay_trade_no'
           AND a.attnum > 0 AND NOT a.attisdropped;
        IF actual_type <> 'text' OR actual_not_null THEN
            RAISE EXCEPTION
                'ECPay preflight: subscriptions.ecpay_trade_no is incompatible';
        END IF;
    END IF;

    -- No unknown overload may be overwritten or left callable. The one expected
    -- signature is deliberately CREATE OR REPLACE below and its ACL converges.
    expected_accept_signature := to_regprocedure(
        'public.accept_ecpay_payment(text,integer,text,text,jsonb,text,uuid,integer,timestamp with time zone,timestamp with time zone)'
    );
    IF EXISTS (
        SELECT 1
          FROM pg_proc p
          JOIN pg_namespace n ON n.oid = p.pronamespace
         WHERE n.nspname = 'public'
           AND p.proname = 'accept_ecpay_payment'
           AND (expected_accept_signature IS NULL OR p.oid <> expected_accept_signature)
    ) THEN
        RAISE EXCEPTION
            'ECPay preflight: unknown public.accept_ecpay_payment overload exists';
    END IF;
    IF expected_accept_signature IS NOT NULL AND EXISTS (
        SELECT 1
          FROM pg_proc p
          CROSS JOIN LATERAL aclexplode(p.proacl) acl
         WHERE p.oid = expected_accept_signature
           AND acl.privilege_type = 'EXECUTE'
           AND acl.grantee NOT IN (
               0,
               p.proowner,
               'anon'::regrole::oid,
               'authenticated'::regrole::oid,
               'service_role'::regrole::oid
           )
    ) THEN
        RAISE EXCEPTION
            'ECPay preflight: existing accept_ecpay_payment has an unknown EXECUTE grantee';
    END IF;
    IF expected_accept_signature IS NOT NULL
       AND (SELECT p.proowner
              FROM pg_proc p
             WHERE p.oid = expected_accept_signature) <> current_user::regrole::oid THEN
        RAISE EXCEPTION
            'ECPay preflight: existing accept_ecpay_payment has an unexpected owner';
    END IF;
END
$preflight$;


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

COMMIT;
