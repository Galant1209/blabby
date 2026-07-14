-- ECPay 綠界信用卡定期定額 — billing schema (DRAFT, review before apply)
--
-- Forward-only, non-destructive, idempotent. This migration only shapes
-- storage; it does NOT change entitlement logic (is_user_pro / profiles.is_pro)
-- and does NOT touch any application code.
--
-- Design contract (from product handoff):
--   * Pro entitlement is time-based: now() < subscriptions.expires_at
--     (expires_at IS the "current_period_end" concept — see comment below).
--     We do NOT introduce a per-user is_pro boolean for the billing path.
--   * Entitlement is written ONLY from the server-to-server ReturnURL /
--     PeriodReturnURL handler, never from the browser OrderResultURL.
--   * Every ECPay callback is persisted raw in payment_events, success or
--     failure, signed or not. That table is the audit source of truth.
--   * Idempotent processing: ECPay retries. The same authorization must not
--     grant twice. See payment_events.gwsr unique index.
--
-- ECPay facts this schema depends on (verified against ECPay dev docs
-- 2026-07-14 — see accompanying report):
--   * First authorization result -> ReturnURL; each subsequent monthly
--     auto-charge result -> PeriodReturnURL. Both are server POST, both
--     carry MerchantTradeNo + RtnCode + gwsr + TotalSuccessTimes.
--   * MerchantTradeNo repeats across every period of one mandate; gwsr
--     (authorization number) is unique per execution -> gwsr is the
--     idempotency key, NOT MerchantTradeNo.
--   * To stop future charges we must call CreditCardPeriodAction
--     Action=Cancel keyed by MerchantTradeNo (= subscriptions.order_id).
--     "期末失效" therefore = call ECPay Cancel now (stops renewal) while
--     leaving expires_at untouched (user keeps Pro to period end, then it
--     lapses because no renewal arrives). gateway_canceled_at records that
--     ECPay actually accepted the cancel.

BEGIN;

-- ─────────────────────────────────────────────────────────────────────────
-- 1. subscriptions — extend the existing table (all additions nullable /
--    defaulted, so this is non-destructive to the 0 existing rows and to the
--    current code paths that write order_id / status / amount / expires_at).
--
--    Decision: orders is MERGED into subscriptions, not a separate table.
--    In ECPay 定期定額 one MerchantTradeNo == one recurring mandate == one
--    subscription row (subsequent charges reuse the same MerchantTradeNo).
--    A separate orders table would be strictly 1:1 with subscriptions and buy
--    nothing. Revisit only if one-time (non-recurring) products are added.
-- ─────────────────────────────────────────────────────────────────────────

ALTER TABLE subscriptions
    ADD COLUMN IF NOT EXISTS current_period_start  timestamptz,
    ADD COLUMN IF NOT EXISTS cancel_at_period_end   boolean NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS canceled_at            timestamptz,   -- user pressed cancel
    ADD COLUMN IF NOT EXISTS gateway_canceled_at    timestamptz,   -- ECPay accepted Cancel
    ADD COLUMN IF NOT EXISTS exec_times_used        integer NOT NULL DEFAULT 0,  -- mirrors TotalSuccessTimes
    ADD COLUMN IF NOT EXISTS exec_times_total       integer,       -- config ExecTimes (99)
    ADD COLUMN IF NOT EXISTS period_type            text,          -- config PeriodType (M)
    ADD COLUMN IF NOT EXISTS frequency              integer,       -- config Frequency (1)
    ADD COLUMN IF NOT EXISTS gwsr                   text,          -- latest ECPay auth number (convenience mirror)
    ADD COLUMN IF NOT EXISTS last_event_at          timestamptz;   -- last callback landed

COMMENT ON COLUMN subscriptions.order_id IS
    'ECPay MerchantTradeNo (our-generated, UNIQUE, <=20 chars). The identifier '
    'passed to CreditCardPeriodAction Action=Cancel to stop future charges.';
COMMENT ON COLUMN subscriptions.expires_at IS
    'current_period_end. Pro entitlement = (status = ''active'' AND now() < expires_at). '
    'NOT renamed to current_period_end because existing code writes expires_at; '
    'rename requires a coordinated code change (see report open question #1).';

-- status enum guard. pending -> active -> (past_due) -> canceled/expired.
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'subscriptions_status_check') THEN
        ALTER TABLE subscriptions
            ADD CONSTRAINT subscriptions_status_check
            CHECK (status IN ('pending', 'active', 'past_due', 'canceled', 'expired'));
    END IF;
END $$;

-- amount is TWD integer minor-unitless (綠界 TWD has no decimals); reject <= 0.
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'subscriptions_amount_positive') THEN
        ALTER TABLE subscriptions
            ADD CONSTRAINT subscriptions_amount_positive
            CHECK (amount IS NULL OR amount > 0);
    END IF;
END $$;

-- Find "cancel requested but still active" subscriptions cheaply (for the
-- reconciliation job that confirms ECPay actually stopped billing).
CREATE INDEX IF NOT EXISTS subscriptions_cancel_pending_idx
    ON subscriptions (cancel_at_period_end)
    WHERE cancel_at_period_end = TRUE AND gateway_canceled_at IS NULL;

-- ─────────────────────────────────────────────────────────────────────────
-- 2. payment_events — append-only raw ledger of every ECPay callback.
--    Never rewrite raw_payload, never DELETE. The only column meant to change
--    after insert is processed/processed_at (idempotency bookkeeping); see the
--    immutability trigger below which freezes everything else.
-- ─────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS payment_events (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    received_at         timestamptz NOT NULL DEFAULT now(),
    source              text NOT NULL DEFAULT 'ecpay',
    callback_type       text,            -- 'first' (ReturnURL) | 'period' (PeriodReturnURL)
    merchant_trade_no   text,            -- MerchantTradeNo (may repeat across periods)
    gwsr                text,            -- ECPay authorization number; unique per execution
    rtn_code            text,            -- '1' = success, else failure
    rtn_msg             text,
    total_success_times integer,         -- ECPay TotalSuccessTimes at this execution
    process_date        timestamptz,     -- ECPay ProcessDate
    check_mac_valid     boolean NOT NULL,-- CheckMacValue verification result (store even if false)
    user_id             uuid REFERENCES auth.users(id) ON DELETE SET NULL,   -- null when unresolved / unsigned
    subscription_id     uuid REFERENCES subscriptions(id) ON DELETE SET NULL,
    raw_payload         jsonb NOT NULL,  -- complete untruncated form body
    processed           boolean NOT NULL DEFAULT FALSE,
    processed_at        timestamptz,
    CONSTRAINT payment_events_callback_type_check
        CHECK (callback_type IS NULL OR callback_type IN ('first', 'period'))
);

-- Idempotency key: one row per ECPay authorization. A retried callback carries
-- the same gwsr and is rejected at insert time -> cannot grant twice. gwsr can
-- be null for a malformed/unsigned hit; those are still stored (no dedupe).
CREATE UNIQUE INDEX IF NOT EXISTS payment_events_gwsr_uniq
    ON payment_events (gwsr) WHERE gwsr IS NOT NULL;
CREATE INDEX IF NOT EXISTS payment_events_merchant_trade_no_idx
    ON payment_events (merchant_trade_no);
CREATE INDEX IF NOT EXISTS payment_events_unprocessed_idx
    ON payment_events (processed) WHERE processed = FALSE;

-- Freeze the audit record: allow flipping processed/processed_at once, forbid
-- any change to the raw payload or the identifying/verification fields, and
-- forbid DELETE. This is what makes "append-only" real rather than a comment.
CREATE OR REPLACE FUNCTION payment_events_immutable()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'payment_events is append-only: DELETE forbidden';
    END IF;
    IF NEW.raw_payload   IS DISTINCT FROM OLD.raw_payload
       OR NEW.gwsr              IS DISTINCT FROM OLD.gwsr
       OR NEW.merchant_trade_no IS DISTINCT FROM OLD.merchant_trade_no
       OR NEW.rtn_code          IS DISTINCT FROM OLD.rtn_code
       OR NEW.check_mac_valid   IS DISTINCT FROM OLD.check_mac_valid
       OR NEW.received_at       IS DISTINCT FROM OLD.received_at THEN
        RAISE EXCEPTION 'payment_events core fields are immutable';
    END IF;
    RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS payment_events_immutable_trg ON payment_events;
CREATE TRIGGER payment_events_immutable_trg
    BEFORE UPDATE OR DELETE ON payment_events
    FOR EACH ROW EXECUTE FUNCTION payment_events_immutable();

-- ─────────────────────────────────────────────────────────────────────────
-- 3. RLS
-- ─────────────────────────────────────────────────────────────────────────

-- payment_events: users must NOT read it at all. RLS on + zero policies =
-- deny-all to anon/authenticated; the callback handler uses the service role,
-- which bypasses RLS.
ALTER TABLE payment_events ENABLE ROW LEVEL SECURITY;

-- subscriptions: RLS is already enabled but has no policy today (reads go
-- through the service role). Add a read-own policy so the client MAY read its
-- own subscription directly. No INSERT/UPDATE/DELETE policy -> writes stay
-- service-role only.
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_policies
        WHERE schemaname = 'public' AND tablename = 'subscriptions'
          AND policyname = 'users read own subscription'
    ) THEN
        CREATE POLICY "users read own subscription" ON subscriptions
            FOR SELECT TO authenticated
            USING (auth.uid() = user_id);
    END IF;
END $$;

COMMIT;
