-- ECPay 綠界金流 — billing schema (DRAFT, review before apply). NOT executed.
--
-- Forward-only, non-destructive, idempotent. Shapes storage only; does NOT
-- change entitlement logic (is_user_pro / profiles.is_pro) and does NOT touch
-- any application code.
--
-- Scope (product decision, Brief B):
--   Phase 1  一次性 NT$199 -> 30 天 Pro                    (billing_mode='one_time')
--   Phase 2  信用卡定期定額 NT$199/月, ExecTimes=60         (billing_mode='recurring')
--   Both phases share ONE subscriptions row shape; the only fork is billing_mode.
--   Phase 2 is not implemented yet but the schema must not need rewriting for it.
--
-- Design contract:
--   1. Pro entitlement is time-based: now() < current_period_end. No is_pro
--      boolean on the billing path. (current_period_end == subscriptions.expires_at,
--      reused rather than renamed — existing code writes expires_at; see report.)
--   2. Entitlement is written ONLY from the server-to-server ReturnURL /
--      PeriodReturnURL / reconciliation job, never the browser OrderResultURL.
--   3. Every callback's raw payload is stored in payment_events, success or
--      failure, signed or not. Append-only audit source of truth.
--   4. Idempotent. See payment_events unique index and its NULLS NOT DISTINCT note.
--
-- ECPay facts this schema depends on (given as fact, Brief B — not re-derived):
--   * 首扣 -> ReturnURL; 第二期起 -> PeriodReturnURL. Two paths.
--   * 續扣 REUSES the same MerchantTradeNo; TotalSuccessTimes increments per
--     period. Therefore the idempotency key is (merchant_trade_no,
--     total_success_times, source), NOT merchant_trade_no alone.
--   * PeriodReturnURL notifies once (+4 retries) then is lost forever -> a daily
--     reconciliation job (QueryCreditCardPeriodInfo) re-observes; its landings go
--     to payment_events with source='reconciliation'.
--   * 6 consecutive auth failures -> ECPay auto-terminates the mandate (irreversible).
--   * Cancel/terminate is irreversible, so "期末失效" = set cancel_at_period_end now,
--     and a job calls Action=Cancel the day BEFORE period_end. Two timestamps
--     (canceled_at = user intent, ecpay_canceled_at = gateway confirmed) because
--     they can be out of sync.
--   * One-time first-charge failure voids that MerchantTradeNo (new order needed);
--     failed pending orders accumulate in orders and are normal. Pro is judged
--     ONLY from subscriptions.current_period_end, never from orders.

BEGIN;

-- ─────────────────────────────────────────────────────────────────────────
-- 1. subscriptions — extend existing table (all additions nullable/defaulted;
--    non-destructive to the 0 existing rows and to current code that writes
--    order_id / status / amount / expires_at). Existing order_id and expires_at
--    are KEPT (renaming would break the live skeleton at backend/main.py).
-- ─────────────────────────────────────────────────────────────────────────

ALTER TABLE subscriptions
    ADD COLUMN IF NOT EXISTS billing_mode          text,          -- 'one_time' | 'recurring'
    ADD COLUMN IF NOT EXISTS current_period_start  timestamptz,
    ADD COLUMN IF NOT EXISTS merchant_trade_no     text,          -- recurring mandate's MerchantTradeNo (=order_id of first order)
    ADD COLUMN IF NOT EXISTS merchant_member_id    text,          -- BindingCard token id (MerchantID + user_id)
    ADD COLUMN IF NOT EXISTS cancel_at_period_end  boolean NOT NULL DEFAULT FALSE,
    ADD COLUMN IF NOT EXISTS canceled_at           timestamptz,   -- user pressed cancel (intent)
    ADD COLUMN IF NOT EXISTS ecpay_canceled_at     timestamptz,   -- ECPay Action=Cancel confirmed (actual)
    ADD COLUMN IF NOT EXISTS total_success_times   integer NOT NULL DEFAULT 0,   -- periods charged so far
    ADD COLUMN IF NOT EXISTS consecutive_failures  integer NOT NULL DEFAULT 0,   -- 6th => ECPay auto-terminates
    ADD COLUMN IF NOT EXISTS exec_times_total      integer,       -- config ExecTimes (60 for recurring)
    ADD COLUMN IF NOT EXISTS period_type           text,          -- config PeriodType ('M')
    ADD COLUMN IF NOT EXISTS frequency             integer;       -- config Frequency (1)

COMMENT ON COLUMN subscriptions.expires_at IS
    'current_period_end. Pro entitlement = (status=''active'' AND now() < expires_at). '
    'Reused as current_period_end; NOT renamed because existing code writes expires_at.';
COMMENT ON COLUMN subscriptions.order_id IS
    'Legacy single-order id (skeleton). New model tracks orders in the orders table '
    'and the recurring mandate in merchant_trade_no. Retire order_id once code migrates.';
COMMENT ON COLUMN subscriptions.merchant_trade_no IS
    'ECPay recurring mandate MerchantTradeNo. Lookup key when a PeriodReturnURL / '
    'reconciliation callback arrives. Passed to CreditCardPeriodAction Action=Cancel.';

-- status enum guard
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'subscriptions_status_check') THEN
        ALTER TABLE subscriptions ADD CONSTRAINT subscriptions_status_check
            CHECK (status IN ('pending', 'active', 'past_due', 'canceled', 'expired'));
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'subscriptions_billing_mode_check') THEN
        ALTER TABLE subscriptions ADD CONSTRAINT subscriptions_billing_mode_check
            CHECK (billing_mode IS NULL OR billing_mode IN ('one_time', 'recurring'));
    END IF;
END $$;

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_constraint WHERE conname = 'subscriptions_amount_positive') THEN
        ALTER TABLE subscriptions ADD CONSTRAINT subscriptions_amount_positive
            CHECK (amount IS NULL OR amount > 0);
    END IF;
END $$;

-- One live recurring mandate per merchant_trade_no.
CREATE UNIQUE INDEX IF NOT EXISTS subscriptions_merchant_trade_no_uniq
    ON subscriptions (merchant_trade_no) WHERE merchant_trade_no IS NOT NULL;
CREATE INDEX IF NOT EXISTS subscriptions_member_id_idx
    ON subscriptions (merchant_member_id) WHERE merchant_member_id IS NOT NULL;
-- Reconciliation job: which recurring subs are live and need a daily query.
CREATE INDEX IF NOT EXISTS subscriptions_recurring_active_idx
    ON subscriptions (billing_mode, status) WHERE billing_mode = 'recurring' AND status = 'active';
-- Delayed-cancel job: user asked to cancel but ECPay not yet told.
CREATE INDEX IF NOT EXISTS subscriptions_cancel_pending_idx
    ON subscriptions (cancel_at_period_end) WHERE cancel_at_period_end = TRUE AND ecpay_canceled_at IS NULL;
-- (subscriptions_expires_at_idx on expires_at == current_period_end already exists.)

-- ─────────────────────────────────────────────────────────────────────────
-- 2. orders — NEW. Our-generated order per checkout attempt. Separate from
--    subscriptions because one subscription spans many orders over time
--    (one_time renewals each mint a new order; recurring's first order opens
--    the mandate) and failed pending orders accumulate here. Pro is NEVER
--    judged from orders — only from subscriptions.current_period_end.
-- ─────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS orders (
    id                uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    merchant_trade_no text NOT NULL UNIQUE,                 -- 'BLB'+yyyyMMddHHmmss+3 random, 20 chars
    user_id           uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    subscription_id   uuid REFERENCES subscriptions(id) ON DELETE SET NULL,
    billing_mode      text NOT NULL,                        -- 'one_time' | 'recurring'
    amount            integer NOT NULL,
    status            text NOT NULL DEFAULT 'pending',      -- 'pending' | 'paid' | 'failed'
    created_at        timestamptz NOT NULL DEFAULT now(),
    updated_at        timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT orders_billing_mode_check CHECK (billing_mode IN ('one_time', 'recurring')),
    CONSTRAINT orders_status_check       CHECK (status IN ('pending', 'paid', 'failed')),
    CONSTRAINT orders_amount_positive    CHECK (amount > 0)
);

CREATE INDEX IF NOT EXISTS orders_user_id_idx         ON orders (user_id);
CREATE INDEX IF NOT EXISTS orders_subscription_id_idx ON orders (subscription_id);
CREATE INDEX IF NOT EXISTS orders_status_idx          ON orders (status);

-- ─────────────────────────────────────────────────────────────────────────
-- 3. payment_events — NEW. Append-only raw ledger of every ECPay callback.
--    Never rewrite raw_payload, never DELETE (enforced by trigger below).
-- ─────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS payment_events (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    received_at         timestamptz NOT NULL DEFAULT now(),
    source              text NOT NULL,          -- 'return_url' | 'period_return_url' | 'reconciliation'
    merchant_trade_no   text,
    total_success_times integer,                -- may be NULL on 首扣 / one_time
    rtn_code            text,
    rtn_msg             text,
    checkmac_valid      boolean NOT NULL,        -- store even when false
    user_id             uuid REFERENCES auth.users(id) ON DELETE SET NULL,   -- NULL when unresolved / unsigned
    subscription_id     uuid REFERENCES subscriptions(id) ON DELETE SET NULL,
    order_id            uuid REFERENCES orders(id) ON DELETE SET NULL,
    raw_payload         jsonb NOT NULL,          -- complete, untruncated
    processed_at        timestamptz,             -- idempotency / processing marker
    CONSTRAINT payment_events_source_check
        CHECK (source IN ('return_url', 'period_return_url', 'reconciliation'))
);

-- Idempotency key: (merchant_trade_no, total_success_times, source).
--   * total_success_times is IN the key because recurring reuses one
--     merchant_trade_no across periods — month 2's charge (tst=2) must NOT be
--     swallowed as a duplicate of month 1 (tst=1).
--   * source is IN the key because the SAME charge is observed by two channels:
--     the live PeriodReturnURL push AND the daily reconciliation query. Each
--     channel must be able to land its own audit row; without source they would
--     collide and reconciliation could not independently confirm a period.
--   * NULLS NOT DISTINCT (Postgres 15+, DB is 17): 首扣/one_time carry NULL
--     total_success_times; with default NULL-distinct semantics, ECPay's up-to-4
--     retries of the same return_url would each insert (NULL treated distinct)
--     and defeat dedup. NULLS NOT DISTINCT makes those NULL-keyed retries collide
--     and dedup correctly.
CREATE UNIQUE INDEX IF NOT EXISTS payment_events_idem_uniq
    ON payment_events (merchant_trade_no, total_success_times, source) NULLS NOT DISTINCT;
CREATE INDEX IF NOT EXISTS payment_events_merchant_trade_no_idx ON payment_events (merchant_trade_no);
CREATE INDEX IF NOT EXISTS payment_events_unprocessed_idx       ON payment_events (processed_at) WHERE processed_at IS NULL;

-- Append-only enforcement: forbid DELETE, freeze the evidentiary fields; only
-- processed_at / late user_id / subscription_id / order_id resolution allowed.
CREATE OR REPLACE FUNCTION payment_events_immutable()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'payment_events is append-only: DELETE forbidden';
    END IF;
    IF NEW.raw_payload         IS DISTINCT FROM OLD.raw_payload
       OR NEW.merchant_trade_no   IS DISTINCT FROM OLD.merchant_trade_no
       OR NEW.total_success_times IS DISTINCT FROM OLD.total_success_times
       OR NEW.rtn_code            IS DISTINCT FROM OLD.rtn_code
       OR NEW.checkmac_valid      IS DISTINCT FROM OLD.checkmac_valid
       OR NEW.source              IS DISTINCT FROM OLD.source
       OR NEW.received_at         IS DISTINCT FROM OLD.received_at THEN
        RAISE EXCEPTION 'payment_events core fields are immutable';
    END IF;
    RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS payment_events_immutable_trg ON payment_events;
CREATE TRIGGER payment_events_immutable_trg
    BEFORE UPDATE OR DELETE ON payment_events
    FOR EACH ROW EXECUTE FUNCTION payment_events_immutable();

-- ─────────────────────────────────────────────────────────────────────────
-- 4. RLS
-- ─────────────────────────────────────────────────────────────────────────

-- payment_events: users must NOT read it at all. RLS on + zero policies =
-- deny-all to anon/authenticated; the callback handler uses the service role.
ALTER TABLE payment_events ENABLE ROW LEVEL SECURITY;

-- orders: read-own only; writes via service role.
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname='public' AND tablename='orders'
                     AND policyname='users read own orders') THEN
        CREATE POLICY "users read own orders" ON orders
            FOR SELECT TO authenticated USING (auth.uid() = user_id);
    END IF;
END $$;

-- subscriptions: RLS already enabled, no policy today (reads go via service
-- role). Add read-own; no write policy -> writes stay service-role only.
DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_policies WHERE schemaname='public' AND tablename='subscriptions'
                     AND policyname='users read own subscription') THEN
        CREATE POLICY "users read own subscription" ON subscriptions
            FOR SELECT TO authenticated USING (auth.uid() = user_id);
    END IF;
END $$;

COMMIT;
