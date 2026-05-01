-- Separate paid Pro (webhook-controlled) from granted Pro (admin-controlled).
-- Before this migration: profiles.is_pro was used for both — admin grants
-- and LemonSqueezy webhook writes both targeted the same column, so a
-- subscription cancellation could silently revoke a manual grant.
--
-- After this migration:
--   profiles.is_pro       — webhook-only (paid status)
--   profiles.is_pro_grant — admin-only (manual grant) + audit trail
--   user_pro_status view  — single source of truth for is_pro_effective
--   is_user_pro()         — helper for app reads
--   get_admin_pro_breakdown() — admin analytics
--
-- IMPORTANT: Step 3 assumes any current is_pro=true was a manual grant.
-- Verify `select count(*) from profiles where is_pro=true` BEFORE running
-- this migration. If the LemonSqueezy webhook has already fired for real
-- payments, exclude those rows from Step 3 manually.

-- Step 1: Add is_pro_grant column (manual grants)
ALTER TABLE profiles
ADD COLUMN IF NOT EXISTS is_pro_grant BOOLEAN NOT NULL DEFAULT FALSE;

-- Step 2: Add grant metadata for audit trail
ALTER TABLE profiles
ADD COLUMN IF NOT EXISTS pro_grant_reason TEXT,
ADD COLUMN IF NOT EXISTS pro_grant_at     TIMESTAMPTZ,
ADD COLUMN IF NOT EXISTS pro_grant_by     TEXT;

-- Step 3: Migrate existing manually-toggled users to is_pro_grant
UPDATE profiles
SET
  is_pro_grant     = TRUE,
  pro_grant_reason = 'Migrated from is_pro before LemonSqueezy launch',
  pro_grant_at     = NOW(),
  pro_grant_by     = 'system_migration'
WHERE is_pro = TRUE;

-- Step 4: Reset is_pro — from now on this column is webhook-only
UPDATE profiles SET is_pro = FALSE WHERE is_pro = TRUE;

-- Step 5: Effective Pro status view
CREATE OR REPLACE VIEW user_pro_status AS
SELECT
  p.id,
  u.email,
  p.is_pro                          AS is_pro_paid,
  p.is_pro_grant,
  (p.is_pro OR p.is_pro_grant)      AS is_pro_effective,
  p.pro_grant_reason,
  p.pro_grant_at,
  p.pro_grant_by
FROM profiles p
JOIN auth.users u ON u.id = p.id;

-- Step 6: Helper function — single source of truth for is_pro checks
CREATE OR REPLACE FUNCTION is_user_pro(user_id UUID)
RETURNS BOOLEAN
LANGUAGE SQL
SECURITY DEFINER
STABLE
SET search_path = public
AS $$
  SELECT COALESCE(is_pro, FALSE) OR COALESCE(is_pro_grant, FALSE)
  FROM profiles
  WHERE id = user_id;
$$;

GRANT EXECUTE ON FUNCTION is_user_pro(UUID) TO authenticated, anon, service_role;

-- Step 7: Admin analytics — separate paying from granted
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
  SELECT
    COUNT(*) FILTER (WHERE is_pro OR is_pro_grant)        AS total_pro_effective,
    COUNT(*) FILTER (WHERE is_pro AND NOT is_pro_grant)   AS paying_users,
    COUNT(*) FILTER (WHERE is_pro_grant AND NOT is_pro)   AS granted_users,
    COUNT(*) FILTER (WHERE is_pro AND is_pro_grant)       AS both_paid_and_granted
  FROM profiles;
$$;

REVOKE EXECUTE ON FUNCTION get_admin_pro_breakdown() FROM public, anon, authenticated;
GRANT  EXECUTE ON FUNCTION get_admin_pro_breakdown() TO service_role;
