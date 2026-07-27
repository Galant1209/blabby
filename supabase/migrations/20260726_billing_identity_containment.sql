-- ============================================================================
-- 20260726_billing_identity_containment.sql
--
-- Closes the four holes that must be shut BEFORE real money can flow:
--   §1  payment_events ledger  — makes every provider callback idempotent
--   §2  is_user_pro()          — time-windowed, so paid Pro can actually expire
--   §3  identity views         — stop exposing auth.users.email to anon
--   §4  RLS on shared content  — questions / reading_passages / writing_questions
--
-- Properties: forward-only, idempotent (safe to re-run), non-destructive
-- (drops no table, drops no column, deletes no row). Rollback lives in
-- 20260726_billing_identity_containment_rollback.sql, section-for-section.
--
-- NOT YET EXECUTED against production. Sections are written to be runnable
-- independently and in order — §1 → §2 → §3 → §4 — so each has its own
-- checkpoint. §2 carries a hard pre-flight gate that ABORTS the transaction
-- rather than silently revoking anyone's Pro.
--
-- Design of §1 is borrowed from the unmerged draft on origin/feat/billing-schema
-- (32cdafa) rather than reinvented; that branch is NOT merged by this file.
-- ============================================================================


-- ────────────────────────────────────────────────────────────────────────────
-- §1  payment_events — append-only ledger + idempotency key
-- ────────────────────────────────────────────────────────────────────────────

CREATE TABLE IF NOT EXISTS public.payment_events (
    id                  uuid PRIMARY KEY DEFAULT gen_random_uuid(),
    received_at         timestamptz NOT NULL DEFAULT now(),
    source              text        NOT NULL,
    merchant_trade_no   text,
    total_success_times integer,
    rtn_code            text,
    rtn_msg             text,
    checkmac_valid      boolean     NOT NULL,
    user_id             uuid REFERENCES auth.users(id)         ON DELETE SET NULL,
    subscription_id     uuid REFERENCES public.subscriptions(id) ON DELETE SET NULL,
    raw_payload         jsonb       NOT NULL,
    processed_at        timestamptz
);

-- 'lemonsqueezy' is additional to the draft's ECPay-only set: the same ledger
-- now carries LS replay protection, keyed by the sha256 of the raw signed body.
DO $$ BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'public.payment_events'::regclass
          AND conname  = 'payment_events_source_check'
    ) THEN
        ALTER TABLE public.payment_events
            ADD CONSTRAINT payment_events_source_check
            CHECK (source IN ('return_url', 'period_return_url',
                              'reconciliation', 'lemonsqueezy'));
    END IF;
END $$;

-- The idempotency key. NULLS NOT DISTINCT (Postgres 15+; this DB is 17) is
-- load-bearing: 首扣 / one_time callbacks carry a NULL total_success_times, and
-- under default NULL-distinct semantics ECPay's up-to-4 retries would each
-- insert a fresh row and defeat dedup entirely.
CREATE UNIQUE INDEX IF NOT EXISTS payment_events_idem_uniq
    ON public.payment_events (merchant_trade_no, total_success_times, source)
    NULLS NOT DISTINCT;

CREATE INDEX IF NOT EXISTS payment_events_merchant_trade_no_idx
    ON public.payment_events (merchant_trade_no);
CREATE INDEX IF NOT EXISTS payment_events_unprocessed_idx
    ON public.payment_events (processed_at) WHERE processed_at IS NULL;

-- Append-only: no DELETE, and the evidentiary columns are frozen. Only the
-- resolution fields (processed_at, user_id, subscription_id) may be updated.
CREATE OR REPLACE FUNCTION public.payment_events_immutable()
RETURNS trigger LANGUAGE plpgsql SET search_path = public AS $$
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

DROP TRIGGER IF EXISTS payment_events_immutable_trg ON public.payment_events;
CREATE TRIGGER payment_events_immutable_trg
    BEFORE UPDATE OR DELETE ON public.payment_events
    FOR EACH ROW EXECUTE FUNCTION public.payment_events_immutable();

-- Deny-all to anon/authenticated: RLS on with zero policies. The callback
-- handler reaches this table as service_role, which RLS does not constrain.
ALTER TABLE public.payment_events ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON TABLE public.payment_events FROM anon, authenticated;

COMMENT ON TABLE public.payment_events IS
    'Append-only ledger of every payment provider callback. The unique index '
    '(merchant_trade_no, total_success_times, source) NULLS NOT DISTINCT is the '
    'idempotency key: a duplicate insert means the callback is a replay/retry '
    'and must not move entitlement a second time.';


-- ────────────────────────────────────────────────────────────────────────────
-- §2  is_user_pro() — time-windowed entitlement
--
--   before: COALESCE(is_pro, false) OR (grant AND not expired)
--            → a single write to profiles.is_pro = Pro forever, and admin
--              cancel was a no-op because nothing ever revoked it.
--   after:  (grant AND not expired) OR (active subscription AND not expired)
--
-- profiles.is_pro is NOT dropped (neither did the draft): the column still
-- records "this user has paid at some point" and feeds get_admin_pro_breakdown's
-- paying_users. It simply stops being an entitlement oracle on its own.
-- ────────────────────────────────────────────────────────────────────────────

-- PRE-FLIGHT GATE. Aborts the whole transaction if any user is Pro *only*
-- through the bare is_pro flag, because for them this migration would be a
-- silent downgrade. Mechanical on purpose — do not replace with an eyeballed
-- row count. (Note: profiles is 17 rows, not the 3 that Supabase's list_tables
-- reports; that figure is a post-restart n_live_tup, not a real count.)
DO $$
DECLARE orphaned integer;
BEGIN
    SELECT count(*) INTO orphaned
    FROM public.profiles p
    WHERE COALESCE(p.is_pro, false)
      AND NOT EXISTS (
            SELECT 1 FROM public.subscriptions s
            WHERE s.user_id = p.id AND s.status = 'active' AND s.expires_at > now())
      AND NOT (
            COALESCE(p.is_pro_grant, false)
            AND (p.pro_grant_expires_at IS NULL OR p.pro_grant_expires_at > now()));

    IF orphaned > 0 THEN
        RAISE EXCEPTION
            'ABORT: % profile(s) are Pro only via bare profiles.is_pro and would '
            'lose access. Give each an active subscriptions row or an is_pro_grant '
            'before applying §2.', orphaned;
    END IF;
END $$;

-- Signature notes (deliberate — do not "fix"):
--   * the parameter stays named user_id; CREATE OR REPLACE cannot rename it.
--   * it is referenced as is_user_pro.user_id because subscriptions also has a
--     user_id column. Without that qualification the paid clause would bind to
--     the column and match every row — this is the correctness pivot here.
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


-- ────────────────────────────────────────────────────────────────────────────
-- §3  Identity containment — stop leaking auth.users.email to anon
--
-- Supabase advisors flag both public.user_pro_status and public.user_lookup as
-- SECURITY DEFINER views exposing auth.users to anon. The anon key is published
-- in frontend/app/config.js:3 and this repo is public, so "anon" means anyone.
--
-- Verified before writing this section (grep over backend/, frontend/,
-- supabase/, scripts/, venv excluded): ZERO consumers of either view. Every
-- backend Pro check goes through the is_user_pro() RPC, and get_admin_users_full
-- is its own SECURITY DEFINER function (20260515_pro_grant_expiry.sql:81-174)
-- that reads auth.users directly, never through a view. Dropping is therefore
-- safe; §3 of the rollback recreates them verbatim if that turns out wrong.
-- ────────────────────────────────────────────────────────────────────────────

DROP VIEW IF EXISTS public.user_pro_status;
DROP VIEW IF EXISTS public.user_lookup;

-- Re-created WITHOUT email and WITHOUT SECURITY DEFINER. security_invoker=on
-- means the view runs with the caller's rights, so RLS on profiles applies and
-- an anon caller sees nothing. Keeps the is_pro_effective concept that
-- 20260501_separate_paid_vs_granted_pro.sql introduced, minus the identity leak.
CREATE VIEW public.user_pro_status
WITH (security_invoker = on) AS
SELECT
    p.id,
    p.is_pro                                        AS is_pro_paid,
    p.is_pro_grant,
    p.pro_grant_expires_at,
    public.is_user_pro(p.id)                        AS is_pro_effective
FROM public.profiles p;

REVOKE ALL ON public.user_pro_status FROM anon;
GRANT SELECT ON public.user_pro_status TO authenticated, service_role;

COMMENT ON VIEW public.user_pro_status IS
    'Pro status without identity. Deliberately has no email column and is '
    'security_invoker — the previous SECURITY DEFINER version exposed '
    'auth.users.email to anon. Admin identity lookups use get_admin_users_full().';

-- public.user_lookup is intentionally NOT recreated: it has no definition
-- anywhere in this repo, no consumer anywhere in this repo, and existed solely
-- as an anon-readable auth.users projection. Rollback §3 restores it only if a
-- consumer is discovered.


-- ────────────────────────────────────────────────────────────────────────────
-- §4  RLS on the three shared-content tables
--
-- Verified before writing this section: the frontend makes exactly TWO
-- supabase .from() calls in total, both on pro_waitlist (index.html:4155,
-- admin.html:2426) — neither of these three tables is ever read with the anon
-- key. The only raw PostgREST fetches are upgrade.html:708/781 against
-- upgrade_intent. admin.html's `s.writing_questions` / `a.reading_passages`
-- are embedded objects inside BACKEND responses (main.py:5441, :5511), fetched
-- with a Bearer token and resolved server-side as service_role.
--
-- service_role bypasses RLS and this migration revokes nothing from it, so
-- Reading and Writing continue to function unchanged.
-- ────────────────────────────────────────────────────────────────────────────

ALTER TABLE public.questions          ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.reading_passages   ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.writing_questions  ENABLE ROW LEVEL SECURITY;

-- questions / writing_questions are shared drill+prompt content. Signed-in
-- users may read them; nobody but service_role may write.
DROP POLICY IF EXISTS questions_read_authenticated ON public.questions;
CREATE POLICY questions_read_authenticated ON public.questions
    FOR SELECT TO authenticated USING (true);

DROP POLICY IF EXISTS writing_questions_read_authenticated ON public.writing_questions;
CREATE POLICY writing_questions_read_authenticated ON public.writing_questions
    FOR SELECT TO authenticated USING (true);

-- reading_passages is NOT shared content: each row has an owner, and the
-- backend already enforces created_by ownership at main.py:7721. Anon could
-- previously read every passage body, vocab_targets and created_by directly,
-- bypassing that check. Own-rows-only mirrors the backend's own rule.
DROP POLICY IF EXISTS reading_passages_read_own ON public.reading_passages;
CREATE POLICY reading_passages_read_own ON public.reading_passages
    FOR SELECT TO authenticated USING (auth.uid() = created_by);

REVOKE ALL ON TABLE public.questions         FROM anon;
REVOKE ALL ON TABLE public.reading_passages  FROM anon;
REVOKE ALL ON TABLE public.writing_questions FROM anon;

GRANT SELECT ON public.questions         TO authenticated;
GRANT SELECT ON public.writing_questions TO authenticated;
GRANT SELECT ON public.reading_passages  TO authenticated;


-- ────────────────────────────────────────────────────────────────────────────
-- §5  anon EXECUTE grants (step 9)
--
-- is_user_pro is SECURITY DEFINER and was executable by anon, so anyone with
-- the published anon key could probe any user's Pro status by uuid. No caller
-- needs that: every invocation is server-side as service_role
-- (main.py get_user_pro_status / is_user_pro), and the frontend has none.
--
-- handle_new_user() is the auth.users signup trigger. It is invoked by the
-- trigger as its definer, never called directly by a client, so anon EXECUTE
-- is likewise unnecessary. Revoking it does NOT break signup.
--
-- FROM PUBLIC, not just FROM anon — corrected 2026-07-27 after executing this
-- against production and verifying the result rather than the exit status.
-- Postgres grants EXECUTE to PUBLIC by default on CREATE FUNCTION, and every
-- role is a member of PUBLIC. Revoking only from anon removed anon's explicit
-- entry from the ACL while leaving the inherited PUBLIC grant intact, so the
-- statement "succeeded" and changed nothing: anon could still call
-- is_user_pro() through the published anon key and read any user's Pro status.
--
-- Safe for the roles that must keep it: authenticated and service_role each
-- hold an EXPLICIT grant (verified in the live ACL), so a PUBLIC revoke does
-- not touch them. Post-fix production ACL is exactly:
--   postgres=X/postgres | authenticated=X/postgres | service_role=X/postgres
-- ────────────────────────────────────────────────────────────────────────────

REVOKE EXECUTE ON FUNCTION public.is_user_pro(uuid) FROM PUBLIC;
REVOKE EXECUTE ON FUNCTION public.is_user_pro(uuid) FROM anon;

DO $$ BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
        WHERE n.nspname = 'public' AND p.proname = 'handle_new_user'
    ) THEN
        EXECUTE 'REVOKE EXECUTE ON FUNCTION public.handle_new_user() FROM PUBLIC';
        EXECUTE 'REVOKE EXECUTE ON FUNCTION public.handle_new_user() FROM anon';
    END IF;
END $$;
