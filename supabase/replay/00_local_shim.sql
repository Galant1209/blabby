-- ============================================================================
-- supabase/replay/00_local_shim.sql
--
-- NOT A MIGRATION. Never runs against production and is not in
-- supabase/migrations/ for exactly that reason.
--
-- A stock Postgres has none of the objects Supabase provides for free, so a
-- clean replay of supabase/migrations/ fails immediately on `auth.users`.
-- This file supplies the minimum Supabase-shaped surface those migrations
-- reference, so the replay exercises OUR schema rather than dying on the
-- platform's. It is intentionally minimal — enough to satisfy references and
-- type-check policy expressions, nothing more.
-- ============================================================================

-- Roles Supabase creates on every project. Grants across the migration set
-- target these by name.
DO $$
DECLARE r text;
BEGIN
    FOREACH r IN ARRAY ARRAY['anon', 'authenticated', 'service_role'] LOOP
        IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = r) THEN
            EXECUTE format('CREATE ROLE %I NOLOGIN', r);
        END IF;
    END LOOP;
END $$;

-- service_role bypasses RLS on real Supabase, and Supabase's default privileges
-- give it table access. Both are needed here, because without them a replay
-- cannot distinguish a correct deny-all lockdown from one that also locked out
-- the backend: service_role would be refused either way, for the wrong reason.
-- FastAPI holds exactly this role (main.py:263), so "service_role can still
-- read" is the assertion that proves a lockdown did not break the product.
--
-- Deliberately NOT extended to anon/authenticated: those two must keep only the
-- privileges the migrations themselves grant, or the replay would stop being
-- able to show what a migration actually opened or closed.
ALTER ROLE service_role BYPASSRLS;
ALTER DEFAULT PRIVILEGES IN SCHEMA public GRANT ALL ON TABLES TO service_role;

CREATE SCHEMA IF NOT EXISTS auth;

-- Only the columns our migrations and RPCs actually touch: id (FK target),
-- email (get_user_id_by_email, get_admin_users_full, user_lookup) and
-- created_at (user_lookup).
CREATE TABLE IF NOT EXISTS auth.users (
    id         uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    email      text,
    created_at timestamptz NOT NULL DEFAULT now()
);

-- auth.uid() backs every RLS policy in the set. Returning NULL is correct for
-- a replay harness: we are proving the policies COMPILE and attach, not
-- simulating a signed-in user.
CREATE OR REPLACE FUNCTION auth.uid() RETURNS uuid
    LANGUAGE sql STABLE AS $$ SELECT NULL::uuid $$;

CREATE OR REPLACE FUNCTION auth.role() RETURNS text
    LANGUAGE sql STABLE AS $$ SELECT NULL::text $$;

GRANT USAGE ON SCHEMA auth   TO anon, authenticated, service_role;
GRANT USAGE ON SCHEMA public TO anon, authenticated, service_role;
