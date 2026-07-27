-- ============================================================================
-- 00000001_baseline_user_lookup_view.sql
--
-- BASELINE SNAPSHOT — DO NOT RUN AGAINST PRODUCTION.
--
-- public.user_lookup exists in production but has no definition anywhere in
-- this repo — origin unknown, never committed. Captured read-only via
-- pg_get_viewdef() on 2026-07-25 and recorded verbatim so the repo finally
-- describes what production actually contains.
--
-- ⚠️ RECORDED AS-IS, DELIBERATELY NOT FIXED.
--
-- This view exposes auth.users.email, and production grants ALL on it to
-- `anon`. The anon key is published in frontend/app/config.js:3 and this repo
-- is public, so in practice this view hands every user's email address to
-- anyone who asks. Supabase's advisors flag it as an ERROR.
--
-- Fixing it is TASK 2's job, not this snapshot's:
--   supabase/migrations/20260726_billing_identity_containment.sql §3
-- drops this view (it has zero consumers anywhere in backend/, frontend/,
-- supabase/ or scripts/ — verified by grep) and rebuilds user_pro_status
-- without email and without SECURITY DEFINER. That migration's code has
-- shipped to main but the SQL has NOT been executed yet.
--
-- Recording the hole accurately is the point. A baseline that quietly
-- sanitises production is worse than no baseline: the next person would
-- replay it, see something safe, and never learn the live system differs.
-- ============================================================================

CREATE OR REPLACE VIEW public.user_lookup AS
SELECT u.id,
       u.email,
       p.covenant_name,
       p.covenant_signed_at,
       p.is_pro,
       u.created_at
FROM auth.users u
LEFT JOIN public.profiles p ON p.id = u.id;

GRANT ALL ON public.user_lookup TO anon, authenticated, service_role;
