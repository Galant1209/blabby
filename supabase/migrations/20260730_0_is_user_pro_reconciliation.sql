-- ============================================================================
-- ⚠️  ENTITLEMENT-CRITICAL MIGRATION — CHANGES WHO IS PRO  ⚠️
--
-- 20260730_0_is_user_pro_reconciliation.sql
--
-- Closes the one object 20260726_billing_identity_containment.sql §2 planned
-- but never landed in production: public.is_user_pro(uuid) is still reading
-- the bare profiles.is_pro boolean instead of the time-windowed definition.
-- This file replaces the function body with that definition and nothing else.
--
-- Filename sorts before 20260730_ecpay_backend.sql (dictionary order under
-- both C and en_US.UTF-8 collation: '_' = 0x5F < '0' = 0x30 is false, so '0'
-- sorts first — verified by direct comparison, not assumed). ECPay's own
-- preflight requires this exact new body before it will apply; see
-- 20260730_ecpay_backend.sql:265-289.
--
-- ── WHAT CHANGES, IN ONE SENTENCE ────────────────────────────────────────────
--   before: is_pro OR (is_pro_grant AND not expired)      — reads a bare flag
--   after:  (is_pro_grant AND not expired)                  — unchanged
--           OR (an active, unexpired subscription exists)   — NEW
--
-- profiles.is_pro is no longer read for entitlement. The column itself is NOT
-- dropped — get_admin_pro_breakdown() and reconciliation tooling still read it
-- as a "has this user ever paid via LemonSqueezy" signal.
--
-- ── PRODUCTION GATE — VERIFIED READ-ONLY, 2026-07-30 ─────────────────────────
--   profiles              = 17 rows
--   bare is_pro = true    = 0    ← nobody is Pro through the flag this removes
--   is_pro_grant = true   = 7
--   grant active today    = 6    (all 6 have pro_grant_expires_at IS NULL —
--                                 permanent grants, unaffected by this change)
--   the 7th grant row     = pro_grant_expires_at = 2026-05-22 10:55:33.223+00
--                            i.e. genuinely expired 68 days ago, NOT NULL —
--                            confirmed by direct query, not inferred. Both the
--                            old body and the new body return false for this
--                            row today (is_pro_grant is true but the window
--                            has closed, and bare is_pro is false for it too).
--   subscriptions         = 0 rows
--   payment_events        = 0 rows
--
-- Because subscriptions is empty, the new subscription clause is a no-op
-- against real data: applying this file changes ZERO users' entitlement in
-- production today. It only changes what happens the moment the first ECPay
-- payment lands — which is the entire point.
--
-- ── EQUIVALENCE MATRIX — the only two permitted new-vs-old differences ───────
-- Every other combination of (grant state × subscription state × bare is_pro)
-- must return IDENTICAL results under both bodies. Verified in replay.sh; see
-- that file for the full 19-case enumeration this header summarises.
--
--   (a) an active, unexpired subscription exists
--       old = false, new = true    — the entire purpose of this migration.
--
--   (b) the user_id has no row in profiles at all
--       old = NULL,  new = false   — not literally the same value, but both
--       are "not Pro" at every call site. The old body is a bare
--       `SELECT ... FROM profiles WHERE id = user_id`: zero matching rows
--       makes a SQL function with a scalar return type yield NULL, confirmed
--       live against the production function pre-migration. The new body is
--       `EXISTS(...) OR EXISTS(...)`, which is boolean-valued and can never be
--       NULL. Every call site treats NULL and false identically:
--         * the one SQL consumer, public.user_pro_status, is
--           `is_user_pro(id) AS is_pro_effective` selected FROM profiles —
--           structurally unreachable with a nonexistent id, since the id comes
--           from the row itself
--         * both Python entry points, get_user_pro_status() (main.py:4245)
--           and async is_user_pro() (main.py:4263), wrap the RPC result in
--           bool(resp.data); bool(None) == bool(False) == False
--       There is also no registration window that could produce this case in
--       practice: handle_new_user() is an AFTER INSERT FOR EACH ROW trigger on
--       auth.users, so the profiles row commits in the same transaction as the
--       auth.users row. Verified 2026-07-30: auth.users = 17, profiles = 17,
--       zero orphans in either direction.
--
--   A THIRD divergence exists only when the preflight's bare-is_pro gate below
--   is violated (is_pro=true with no covering grant/subscription: old=true,
--   new=false). That is not a permitted difference — it is the exact
--   "silently revoke someone's Pro" failure this file must never cause, which
--   is why the gate aborts rather than letting that cell be reached. Today the
--   count is 0, so the gate passes; it exists for whenever this runs again
--   against a database that has drifted.
--
-- ── DEPENDENCIES CHECKED, NOT §4's SUPERSEDED DESIGN ─────────────────────────
-- The preflight verifies every other object 20260726 created (§1 payment_events
-- and its constraint/indexes/trigger/RLS, §3's two views, §5's anon-execute
-- revoke) — is_user_pro shares a migration file with them and a broken sibling
-- object is a signal something is wrong with the baseline this runs against.
--
-- §4 is checked ONLY for "RLS enabled on questions/reading_passages/
-- writing_questions" — NOT for the three named policies
-- (questions_read_authenticated / writing_questions_read_authenticated /
-- reading_passages_read_own) that 20260726 §4 originally planned to create.
--
-- Those three policies were superseded by
-- 20260731_content_access_lockdown.sql, applied to production 2026-07-30,
-- which DROPs them by name and replaces permissive-policy RLS with
-- zero-policy fail-closed RLS. is_user_pro's body has never read
-- questions/reading_passages/writing_questions or any policy on them — §4 was
-- never a real dependency, only part of a broad "is 20260726 coherent"
-- sanity check. Requiring those three specific policies to exist would check
-- for a state 20260731 deliberately and correctly replaced, permanently
-- aborting this migration over a divergence that is not a defect.
--
-- ── PROPERTIES ────────────────────────────────────────────────────────────
-- Forward-only, idempotent (CREATE OR REPLACE; safe rerun once already
-- applied), non-destructive (drops no table, column or row; does not touch
-- profiles.is_pro). Rollback in
-- 20260730_0_is_user_pro_reconciliation_rollback.sql — it REFUSES to run
-- while any active, unexpired subscription exists, because reverting the body
-- would silently revoke Pro from a paying customer. See that file's own
-- warning banner.
--
-- Independent of 20260731_content_access_lockdown.sql: zero object overlap,
-- neither preflight inspects anything the other touches, either may run
-- before or after the other.
--
-- NOT YET EXECUTED against production.
-- ============================================================================

BEGIN;

-- ── EXECUTABLE PRE-MUTATION GATE ────────────────────────────────────────────
DO $preflight$
DECLARE
    fn_oid            oid;
    fn_count          integer;
    fn_body           text;
    expected_body     text;
    orphaned_is_pro   integer;
    missing           text[] := '{}';
BEGIN
    ------------------------------------------------------------ §1 payment_events
    IF NOT EXISTS (
        SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = 'public' AND c.relname = 'payment_events' AND c.relkind = 'r'
    ) THEN
        missing := missing || 'table public.payment_events';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
         WHERE conrelid = 'public.payment_events'::regclass
           AND conname  = 'payment_events_source_check'
    ) THEN
        missing := missing || 'constraint payment_events_source_check';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
         WHERE schemaname = 'public' AND indexname = 'payment_events_idem_uniq'
    ) THEN
        missing := missing || 'index payment_events_idem_uniq';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
         WHERE schemaname = 'public' AND indexname = 'payment_events_merchant_trade_no_idx'
    ) THEN
        missing := missing || 'index payment_events_merchant_trade_no_idx';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
         WHERE schemaname = 'public' AND indexname = 'payment_events_unprocessed_idx'
    ) THEN
        missing := missing || 'index payment_events_unprocessed_idx';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
         WHERE n.nspname = 'public' AND p.proname = 'payment_events_immutable'
    ) THEN
        missing := missing || 'function payment_events_immutable()';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger t JOIN pg_class c ON c.oid = t.tgrelid
         WHERE c.relname = 'payment_events' AND t.tgname = 'payment_events_immutable_trg'
           AND NOT t.tgisinternal
    ) THEN
        missing := missing || 'trigger payment_events_immutable_trg';
    END IF;
    IF NOT COALESCE((
        SELECT c.relrowsecurity FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = 'public' AND c.relname = 'payment_events'
    ), false) THEN
        missing := missing || 'RLS enabled on payment_events';
    END IF;

    ------------------------------------------------------------------- §3 views
    IF EXISTS (
        SELECT 1 FROM pg_views WHERE schemaname = 'public' AND viewname = 'user_lookup'
    ) THEN
        missing := missing || 'user_lookup should be dropped (§3) but still exists';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = 'public' AND c.relname = 'user_pro_status' AND c.relkind = 'v'
           AND 'security_invoker=on' = ANY(c.reloptions)
    ) THEN
        missing := missing || 'view user_pro_status (security_invoker=on)';
    END IF;

    ----------------------------------------------- §4, RLS only — see header
    -- Deliberately NOT checking for questions_read_authenticated /
    -- writing_questions_read_authenticated / reading_passages_read_own: those
    -- were superseded by 20260731_content_access_lockdown.sql and checking for
    -- them would permanently abort this migration over a correct divergence.
    IF NOT COALESCE((
        SELECT bool_and(c.relrowsecurity) FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
         WHERE n.nspname = 'public'
           AND c.relname IN ('questions', 'reading_passages', 'writing_questions')
    ), false) THEN
        missing := missing || 'RLS enabled on questions/reading_passages/writing_questions';
    END IF;

    ------------------------------------------------------- §5 anon revoked
    IF has_function_privilege('anon', 'public.is_user_pro(uuid)', 'EXECUTE') THEN
        missing := missing || 'anon must not hold EXECUTE on is_user_pro(uuid)';
    END IF;

    IF array_length(missing, 1) > 0 THEN
        RAISE EXCEPTION
            'is_user_pro reconciliation: 20260726 baseline is incomplete or has '
            'drifted — missing/unexpected: %. Resolve before reconciling '
            'is_user_pro.', array_to_string(missing, '; ');
    END IF;

    ------------------------------------------- direct dependencies of the body
    IF NOT EXISTS (
        SELECT 1 FROM pg_attribute a WHERE a.attrelid = 'public.profiles'::regclass
           AND a.attname = 'is_pro_grant' AND a.attnum > 0 AND NOT a.attisdropped
    ) OR NOT EXISTS (
        SELECT 1 FROM pg_attribute a WHERE a.attrelid = 'public.profiles'::regclass
           AND a.attname = 'pro_grant_expires_at' AND a.attnum > 0 AND NOT a.attisdropped
    ) THEN
        RAISE EXCEPTION
            'is_user_pro reconciliation: profiles.is_pro_grant / '
            'pro_grant_expires_at are missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_attribute a WHERE a.attrelid = 'public.subscriptions'::regclass
           AND a.attname = 'status' AND a.attnum > 0 AND NOT a.attisdropped
    ) OR NOT EXISTS (
        SELECT 1 FROM pg_attribute a WHERE a.attrelid = 'public.subscriptions'::regclass
           AND a.attname = 'expires_at' AND a.attnum > 0 AND NOT a.attisdropped
    ) THEN
        RAISE EXCEPTION
            'is_user_pro reconciliation: subscriptions.status / expires_at are missing';
    END IF;

    -------------------------------------------- exactly one is_user_pro overload
    SELECT count(*) INTO fn_count
      FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
     WHERE n.nspname = 'public' AND p.proname = 'is_user_pro';
    IF fn_count <> 1 THEN
        RAISE EXCEPTION
            'is_user_pro reconciliation: expected exactly 1 overload of '
            'is_user_pro, found %. An unexpected overload could shadow this '
            'one.', fn_count;
    END IF;

    ------------------------------ mechanical entitlement gate — never revoke
    -- Anyone holding Pro ONLY through the bare is_pro flag (i.e. not covered
    -- by an active grant and not covered by an active subscription) would lose
    -- access the moment this file's new body goes live. Abort rather than
    -- silently downgrade anyone. Today's count is 0; this exists for reruns
    -- against a database that has drifted since.
    SELECT count(*) INTO orphaned_is_pro
      FROM public.profiles p
     WHERE COALESCE(p.is_pro, false)
       AND NOT (
             COALESCE(p.is_pro_grant, false)
             AND (p.pro_grant_expires_at IS NULL OR p.pro_grant_expires_at > now())
           )
       AND NOT EXISTS (
             SELECT 1 FROM public.subscriptions s
              WHERE s.user_id = p.id AND s.status = 'active' AND s.expires_at > now()
           );
    IF orphaned_is_pro > 0 THEN
        RAISE EXCEPTION
            'is_user_pro reconciliation: ABORT — % profile(s) are Pro only via '
            'the bare is_pro flag this migration stops reading, with no active '
            'grant or subscription to cover them. Give each an active grant or '
            'subscription before reconciling, or they lose Pro the moment this '
            'applies.', orphaned_is_pro;
    END IF;

    ------------------------------------ idempotency check — safe rerun signal
    -- Not an abort condition either way; recorded so the mutation below is
    -- provably a no-op semantically when already applied (CREATE OR REPLACE
    -- makes it syntactically a no-op regardless).
    SELECT p.oid INTO fn_oid
      FROM pg_proc p WHERE p.oid = to_regprocedure('public.is_user_pro(uuid)');
    SELECT regexp_replace(lower(p.prosrc), '\s+', ' ', 'g') INTO fn_body
      FROM pg_proc p WHERE p.oid = fn_oid;
    expected_body := regexp_replace(lower($expected$
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
    IF btrim(fn_body) = btrim(expected_body) THEN
        RAISE NOTICE 'is_user_pro reconciliation: already applied — proceeding as a safe no-op rerun';
    END IF;
END
$preflight$;


-- ── the reconciliation itself ────────────────────────────────────────────────
-- Signature notes (deliberate — do not "fix"):
--   * the parameter stays named user_id; CREATE OR REPLACE cannot rename it.
--   * referenced as is_user_pro.user_id because subscriptions also has a
--     user_id column — without this qualifier the paid clause could bind to
--     the column and match every row. This qualification is the correctness
--     pivot of this file.
CREATE OR REPLACE FUNCTION public.is_user_pro(user_id uuid)
 RETURNS boolean
 LANGUAGE sql
 STABLE SECURITY DEFINER
 SET search_path TO 'public'
AS $function$
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
$function$;

-- CREATE OR REPLACE preserves existing grants, but re-asserted explicitly so
-- this file is correct standalone even against a database where §5 somehow
-- had to be reapplied.
REVOKE EXECUTE ON FUNCTION public.is_user_pro(uuid) FROM PUBLIC, anon;

COMMIT;


-- ── verification (run after applying) ────────────────────────────────────────
-- 1. SELECT prosrc FROM pg_proc WHERE oid = to_regprocedure('public.is_user_pro(uuid)');
--    → contains "FROM subscriptions s"
--
-- 2. SELECT is_user_pro('<any existing user_id>'::uuid);
--    → matches the equivalence matrix in this file's header for that user's
--      actual grant/subscription state
--
-- 3. SELECT has_function_privilege('anon', 'public.is_user_pro(uuid)', 'EXECUTE');
--    → false
