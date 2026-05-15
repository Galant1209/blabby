-- Add expiry to manual Pro grants. Grants with NULL pro_grant_expires_at
-- are permanent (current default). Once a grant has a non-null expiry
-- and NOW() > expires_at, the user is treated as not Pro.
--
-- Paid Pro (profiles.is_pro from webhook) is unaffected — only grants
-- have expiry semantics. The OR logic stays: paid never expires here,
-- granted expires per pro_grant_expires_at.

-- Step 1: Add the column. Nullable; null = permanent grant.
ALTER TABLE profiles
ADD COLUMN IF NOT EXISTS pro_grant_expires_at TIMESTAMPTZ;

-- Step 2: Update is_user_pro() — the single source of truth.
CREATE OR REPLACE FUNCTION is_user_pro(user_id UUID)
RETURNS BOOLEAN
LANGUAGE SQL
SECURITY DEFINER
STABLE
SET search_path = public
AS $$
  SELECT
    COALESCE(is_pro, FALSE)
    OR (
      COALESCE(is_pro_grant, FALSE)
      AND (pro_grant_expires_at IS NULL OR pro_grant_expires_at > NOW())
    )
  FROM profiles
  WHERE id = user_id;
$$;

GRANT EXECUTE ON FUNCTION is_user_pro(UUID) TO authenticated, anon, service_role;

-- Step 3: Update the analytics view so is_pro_effective honours expiry.
CREATE OR REPLACE VIEW user_pro_status AS
SELECT
  p.id,
  u.email,
  p.is_pro                          AS is_pro_paid,
  p.is_pro_grant,
  (
    p.is_pro
    OR (p.is_pro_grant AND (p.pro_grant_expires_at IS NULL OR p.pro_grant_expires_at > NOW()))
  )                                 AS is_pro_effective,
  p.pro_grant_reason,
  p.pro_grant_at,
  p.pro_grant_by,
  p.pro_grant_expires_at
FROM profiles p
JOIN auth.users u ON u.id = p.id;

-- Step 4: Update admin breakdown so expired grants don't inflate counts.
CREATE OR REPLACE FUNCTION get_admin_pro_breakdown()
RETURNS TABLE (
  total_pro_effective BIGINT,
  paying_users        BIGINT,
  granted_users       BIGINT,
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

REVOKE EXECUTE ON FUNCTION get_admin_pro_breakdown() FROM public, anon, authenticated;
GRANT  EXECUTE ON FUNCTION get_admin_pro_breakdown() TO service_role;

-- Step 5: Update admin users RPC — expose pro_grant_expires_at + update is_pro_effective.
CREATE OR REPLACE FUNCTION public.get_admin_users_full()
RETURNS TABLE (
  user_id uuid,
  email text,
  is_pro_paid boolean,
  is_pro_grant boolean,
  is_pro_effective boolean,
  pro_grant_reason text,
  pro_grant_at timestamptz,
  pro_grant_by text,
  pro_grant_expires_at timestamptz,
  created_at timestamptz,
  practice_count bigint,
  practice_count_7d bigint,
  drill_count bigint,
  last_practice timestamptz,
  main_weakness_tag text,
  is_waitlisted boolean,
  conversion_score int,
  churn_risk text
)
LANGUAGE SQL
SECURITY DEFINER
SET search_path = public
AS $$
  WITH base AS (
    SELECT
      p.id AS user_id,
      u.email,
      p.is_pro                          AS is_pro_paid,
      p.is_pro_grant,
      (
        p.is_pro
        OR (p.is_pro_grant AND (p.pro_grant_expires_at IS NULL OR p.pro_grant_expires_at > NOW()))
      )                                 AS is_pro_effective,
      p.pro_grant_reason,
      p.pro_grant_at,
      p.pro_grant_by,
      p.pro_grant_expires_at,
      p.created_at,
      COUNT(DISTINCT pr.id) AS practice_count,
      COUNT(DISTINCT pr.id) FILTER (WHERE pr.created_at >= NOW() - INTERVAL '7 days') AS practice_count_7d,
      COUNT(DISTINCT du.id) AS drill_count,
      MAX(pr.created_at) AS last_practice,
      MODE() WITHIN GROUP (ORDER BY pr.weakness_tag) FILTER (WHERE pr.weakness_tag IS NOT NULL) AS main_weakness_tag
    FROM profiles p
    JOIN auth.users u ON u.id = p.id
    LEFT JOIN practice_records pr ON pr.user_id = p.id
    LEFT JOIN drill_usage du ON du.user_id = p.id
    GROUP BY
      p.id, u.email, p.is_pro, p.is_pro_grant,
      p.pro_grant_reason, p.pro_grant_at, p.pro_grant_by, p.pro_grant_expires_at,
      p.created_at
  ),
  waitlist AS (
    SELECT DISTINCT lower(email) AS email FROM upgrade_intent
  )
  SELECT
    b.user_id,
    b.email,
    b.is_pro_paid,
    b.is_pro_grant,
    b.is_pro_effective,
    b.pro_grant_reason,
    b.pro_grant_at,
    b.pro_grant_by,
    b.pro_grant_expires_at,
    b.created_at,
    b.practice_count,
    b.practice_count_7d,
    b.drill_count,
    b.last_practice,
    b.main_weakness_tag,
    (lower(b.email) IN (SELECT email FROM waitlist)) AS is_waitlisted,
    (
      CASE WHEN b.practice_count_7d >= 3 THEN 20 ELSE 0 END +
      CASE WHEN b.practice_count >= 10 THEN 15 ELSE 0 END +
      CASE WHEN b.drill_count > 0 THEN 20 ELSE 0 END +
      CASE WHEN b.main_weakness_tag IS NOT NULL THEN 15 ELSE 0 END +
      CASE WHEN lower(b.email) IN (SELECT email FROM waitlist) THEN 30 ELSE 0 END
    )::int AS conversion_score,
    CASE
      WHEN b.last_practice >= NOW() - INTERVAL '1 day' THEN 'low'
      WHEN b.last_practice >= NOW() - INTERVAL '3 days' THEN 'medium'
      WHEN b.practice_count >= 3 THEN 'high'
      WHEN b.practice_count = 0 THEN 'new'
      ELSE 'medium'
    END AS churn_risk
  FROM base b
  ORDER BY conversion_score DESC, last_practice DESC NULLS LAST;
$$;

REVOKE EXECUTE ON FUNCTION public.get_admin_users_full() FROM public, anon, authenticated;
GRANT  EXECUTE ON FUNCTION public.get_admin_users_full() TO service_role;
