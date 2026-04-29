-- Sprint 2B: drill_usage tracks each completed drill attempt.
--
-- Inserted by /process (service-role bypasses RLS) after a valid
-- drill_score response. NEVER inserted by client.
--
-- Read by:
--   - GET /api/drill/check_quota  (counts rolling 7-day window)
--   - users (RLS SELECT policy below — own rows only)

CREATE TABLE IF NOT EXISTS drill_usage (
    id              UUID         PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id         UUID         NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    drill_tag       TEXT         NOT NULL,
    drill_score     INTEGER,                              -- nullable: keep raw 0-100 for analytics
    is_pro_at_time  BOOLEAN      NOT NULL DEFAULT FALSE,  -- snapshot for future churn analysis
    created_at      TIMESTAMPTZ  NOT NULL DEFAULT NOW()
);

-- Quota query reads (user_id, created_at) with a date filter, sorted DESC.
-- Composite index covers both filter and sort cheaply.
CREATE INDEX IF NOT EXISTS idx_drill_usage_user_time
    ON drill_usage (user_id, created_at DESC);

-- RLS: users can only SELECT their own rows. INSERTs via service_role
-- bypass RLS; we deliberately don't define an INSERT policy so the
-- anon/authenticated role cannot write directly.
ALTER TABLE drill_usage ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS drill_usage_select_own ON drill_usage;
CREATE POLICY drill_usage_select_own
    ON drill_usage
    FOR SELECT
    TO authenticated
    USING (auth.uid() = user_id);
