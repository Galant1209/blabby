-- Per-user paid-subscription record. Coexists with profiles.is_pro:
--   profiles.is_pro = "currently entitled to Pro features" (boolean)
--   subscriptions   = audit/timeline ("when did they pay, how long good for")
-- Active subscription rows drive the entitlement; this table is the
-- source-of-truth for billing operations (extend, cancel, refund window).

CREATE TABLE IF NOT EXISTS subscriptions (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id    uuid NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    order_id   text UNIQUE,
    plan       text NOT NULL DEFAULT 'monthly',
    status     text NOT NULL DEFAULT 'pending',
    amount     integer,
    started_at timestamptz,
    expires_at timestamptz,
    created_at timestamptz DEFAULT now(),
    updated_at timestamptz DEFAULT now()
);

CREATE INDEX IF NOT EXISTS subscriptions_user_id_idx    ON subscriptions(user_id);
CREATE INDEX IF NOT EXISTS subscriptions_status_idx     ON subscriptions(status);
CREATE INDEX IF NOT EXISTS subscriptions_expires_at_idx ON subscriptions(expires_at);
