#!/usr/bin/env bash
# Replay supabase/migrations/ into an empty Postgres and assert the four admin
# RPCs get built. Used both locally and by the migration-replay CI job.
#
# Usage: PGURI=postgresql://... ./supabase/replay/replay.sh
set -euo pipefail

: "${PGURI:?PGURI must be set (e.g. postgresql://postgres:postgres@localhost:5432/postgres)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATIONS="$HERE/../migrations"
ECPAY_MIGRATION="$MIGRATIONS/20260730_ecpay_backend.sql"
RECON_MIGRATION="$MIGRATIONS/20260730_0_is_user_pro_reconciliation.sql"
RECON_ROLLBACK="$MIGRATIONS/20260730_0_is_user_pro_reconciliation_rollback.sql"

psql_run() { psql "$PGURI" -v ON_ERROR_STOP=1 -q -f "$1"; }

assert_ecpay_migration_left_no_partial_state() {
    local fixture_mode="${1:-clean}"
    psql "$PGURI" -v ON_ERROR_STOP=1 -q <<'SQL'
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
          FROM pg_constraint c
         WHERE c.conrelid = 'public.payment_events'::regclass
           AND c.conname = 'payment_events_source_check'
           AND pg_get_constraintdef(c.oid) LIKE '%ecpay_callback%'
    ) THEN
        RAISE EXCEPTION 'failed migration changed payment_events source constraint';
    END IF;
    IF to_regclass('public.subscriptions_merchant_trade_no_uniq') IS NOT NULL THEN
        RAISE EXCEPTION 'failed migration left merchant trade unique index';
    END IF;
    IF NOT EXISTS (
        SELECT 1
          FROM pg_proc p
         WHERE p.oid = to_regprocedure('public.is_user_pro(uuid)')
           AND p.prosrc ILIKE '%is_pro_grant%'
           AND p.prosrc ILIKE '%subscriptions%'
           AND p.prosrc NOT ILIKE '%COALESCE(p.is_pro, false)%'
    ) THEN
        RAISE EXCEPTION 'failed migration changed or lost is_user_pro dependency body';
    END IF;
END $$;
SQL

    case "$fixture_mode" in
        clean)
            psql "$PGURI" -v ON_ERROR_STOP=1 -q <<'SQL'
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
          FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'subscriptions'
           AND column_name IN ('merchant_trade_no', 'ecpay_trade_no')
    ) THEN
        RAISE EXCEPTION 'failed migration left an ECPay subscription column';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM pg_proc p
          JOIN pg_namespace n ON n.oid = p.pronamespace
         WHERE n.nspname = 'public'
           AND p.proname = 'accept_ecpay_payment'
    ) THEN
        RAISE EXCEPTION 'failed migration left accept_ecpay_payment';
    END IF;
END $$;
SQL
            ;;
        function_fixture)
            psql "$PGURI" -v ON_ERROR_STOP=1 -q <<'SQL'
DO $$
BEGIN
    IF to_regprocedure(
        'public.accept_ecpay_payment(text,integer,text,text,jsonb,text,uuid,integer,timestamp with time zone,timestamp with time zone)'
    ) IS NOT NULL THEN
        RAISE EXCEPTION 'failed migration created the expected acceptance RPC';
    END IF;
    IF to_regprocedure('public.accept_ecpay_payment(integer)') IS NULL THEN
        RAISE EXCEPTION 'failed migration removed the collision fixture';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'subscriptions'
           AND column_name IN ('merchant_trade_no', 'ecpay_trade_no')
    ) THEN
        RAISE EXCEPTION 'failed migration left an ECPay subscription column';
    END IF;
END $$;
SQL
            ;;
        merchant_fixture)
            psql "$PGURI" -v ON_ERROR_STOP=1 -q <<'SQL'
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
          FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'subscriptions'
           AND column_name = 'merchant_trade_no'
           AND data_type = 'text'
           AND is_nullable = 'YES'
    ) OR EXISTS (
        SELECT 1
          FROM information_schema.columns
         WHERE table_schema = 'public'
           AND table_name = 'subscriptions'
           AND column_name = 'ecpay_trade_no'
    ) THEN
        RAISE EXCEPTION 'failed migration changed the partial merchant fixture';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM pg_proc p
          JOIN pg_namespace n ON n.oid = p.pronamespace
         WHERE n.nspname = 'public'
           AND p.proname = 'accept_ecpay_payment'
    ) THEN
        RAISE EXCEPTION 'failed migration left accept_ecpay_payment';
    END IF;
END $$;
SQL
            ;;
        *)
            echo "unknown replay fixture mode: $fixture_mode" >&2
            return 1
            ;;
    esac
}

expect_ecpay_migration_failure() {
    local label="$1"
    local migration_file="${2:-$ECPAY_MIGRATION}"
    local fixture_mode="${3:-clean}"
    local output_file="$REPLAY_PROOF_DIR/${label}.log"

    if psql "$PGURI" -v ON_ERROR_STOP=1 -q -f "$migration_file" \
        >"$output_file" 2>&1; then
        echo "FAIL  $label: migration unexpectedly succeeded" >&2
        return 1
    fi
    assert_ecpay_migration_left_no_partial_state "$fixture_mode"
    echo "ok    $label (rejected, no partial migration state)"
}

run_ecpay_migration_contract_proofs() {
    REPLAY_PROOF_DIR="$(mktemp -d)"
    local fault_migration="$REPLAY_PROOF_DIR/20260730_forced_failure.sql"
    trap 'rm -rf "$REPLAY_PROOF_DIR"' RETURN

    echo
    echo "── prove full ECPay migration rollback and gates ─────"

    # Full-file atomicity: inject a failure after §2, without modifying the
    # repository migration, and prove §1/§2 rolled back with the transaction.
    awk '
        /-- §3  get_admin_pro_breakdown/ && !injected {
            print "SELECT 1 / 0;"
            injected = 1
        }
        { print }
    ' "$ECPAY_MIGRATION" >"$fault_migration"
    if ! grep -q 'SELECT 1 / 0;' "$fault_migration"; then
        echo "failed to inject migration fault" >&2
        return 1
    fi
    expect_ecpay_migration_failure "atomic_midfile_failure" "$fault_migration"

    # P1: a bare is_pro user must be rejected before any mutation.
    psql "$PGURI" -v ON_ERROR_STOP=1 -q <<'SQL'
INSERT INTO auth.users (id, email)
VALUES ('00000000-0000-0000-0000-000000000071',
        'bare-pro-preflight@example.invalid')
ON CONFLICT (id) DO NOTHING;
INSERT INTO public.profiles (id, is_pro, is_pro_grant)
VALUES ('00000000-0000-0000-0000-000000000071', true, false)
ON CONFLICT (id) DO UPDATE
SET is_pro = true, is_pro_grant = false, pro_grant_expires_at = NULL;
SQL
    expect_ecpay_migration_failure "preflight_bare_is_pro"
    psql "$PGURI" -v ON_ERROR_STOP=1 -q -c \
        "DELETE FROM auth.users WHERE id = '00000000-0000-0000-0000-000000000071'"

    # P2: the append-only dependency trigger may not be missing.
    psql "$PGURI" -v ON_ERROR_STOP=1 -q -c \
        "DROP TRIGGER payment_events_immutable_trg ON public.payment_events"
    expect_ecpay_migration_failure "preflight_missing_immutable_trigger"
    psql "$PGURI" -v ON_ERROR_STOP=1 -q <<'SQL'
CREATE TRIGGER payment_events_immutable_trg
    BEFORE UPDATE OR DELETE ON public.payment_events
    FOR EACH ROW EXECUTE FUNCTION public.payment_events_immutable();
SQL

    # P3: an unknown same-name overload must not be overwritten or ignored.
    psql "$PGURI" -v ON_ERROR_STOP=1 -q <<'SQL'
CREATE FUNCTION public.accept_ecpay_payment(integer)
RETURNS integer LANGUAGE sql AS 'SELECT $1';
SQL
    expect_ecpay_migration_failure \
        "preflight_unknown_function_overload" "$ECPAY_MIGRATION" "function_fixture"
    if [[ "$(psql "$PGURI" -v ON_ERROR_STOP=1 -qAt -c \
        "SELECT public.accept_ecpay_payment(17)")" != "17" ]]; then
        echo "unknown overload was changed by rejected migration" >&2
        return 1
    fi
    psql "$PGURI" -v ON_ERROR_STOP=1 -q -c \
        "DROP FUNCTION public.accept_ecpay_payment(integer)"

    # P4: a plausible partial/manual backfill with duplicate trade numbers and
    # no repository index is incompatible. The gate must reject it as-is rather
    # than guessing new identifiers or silently repairing unknown state.
    psql "$PGURI" -v ON_ERROR_STOP=1 -q <<'SQL'
ALTER TABLE public.subscriptions ADD COLUMN merchant_trade_no text;
INSERT INTO auth.users (id, email)
VALUES
    ('00000000-0000-0000-0000-000000000072',
     'partial-order-one@example.invalid'),
    ('00000000-0000-0000-0000-000000000073',
     'partial-order-two@example.invalid')
ON CONFLICT (id) DO NOTHING;
INSERT INTO public.subscriptions (
    user_id, order_id, status, amount, merchant_trade_no
) VALUES
    ('00000000-0000-0000-0000-000000000072',
     'PARTIAL-ORDER-ONE', 'pending', 199, 'DUPLICATE-PARTIAL'),
    ('00000000-0000-0000-0000-000000000073',
     'PARTIAL-ORDER-TWO', 'pending', 199, 'DUPLICATE-PARTIAL');
SQL
    expect_ecpay_migration_failure \
        "preflight_incompatible_subscription_mapping" "$ECPAY_MIGRATION" "merchant_fixture"
    psql "$PGURI" -v ON_ERROR_STOP=1 -q <<'SQL'
DELETE FROM auth.users
WHERE id IN (
    '00000000-0000-0000-0000-000000000072',
    '00000000-0000-0000-0000-000000000073'
);
ALTER TABLE public.subscriptions DROP COLUMN merchant_trade_no;
SQL

    rm -rf "$REPLAY_PROOF_DIR"
    trap - RETURN
}

#   log_file: unbound variable" once bash reached the OUTER function's own
# return — trap ... RETURN is not auto-scoped away when its function exits,
# it stays registered and refires on the next function return anywhere in the
# same call stack, by which point that other function's own `local log_file`
# no longer exists. Matches run_ecpay_migration_contract_proofs's proven
# pattern instead: one shared, non-local directory set up and torn down once
# by the caller (RECON_PROOF_DIR), no per-call trap here at all.
expect_reconciliation_migration_failure() {
    local label="$1"
    local log_file="$RECON_PROOF_DIR/${label}.log"
    if psql "$PGURI" -v ON_ERROR_STOP=1 -q -f "$RECON_MIGRATION" \
        >"$log_file" 2>&1; then
        echo "FAIL  $label: reconciliation migration unexpectedly succeeded" >&2
        cat "$log_file" >&2
        return 1
    fi
    echo "ok    $label (rejected)"
}

expect_reconciliation_rollback_failure() {
    local label="$1"
    local log_file="$RECON_PROOF_DIR/${label}.log"
    if psql "$PGURI" -v ON_ERROR_STOP=1 -q -f "$RECON_ROLLBACK" \
        >"$log_file" 2>&1; then
        echo "FAIL  $label: reconciliation rollback unexpectedly succeeded" >&2
        cat "$log_file" >&2
        return 1
    fi
    echo "ok    $label (rejected)"
}

# Verified by the reconciliation migration's own header:
#   - two ALLOWED differences: an active/unexpired subscription (old=false,
#     new=true — the entire point), and a nonexistent profiles row
#     (old=NULL, new=false — both non-Pro at every call site)
#   - every other combination of grant state x subscription state x bare
#     is_pro must return IDENTICAL results under both bodies
#
# By the point this hook fires (just before the main loop applies
# 20260730_0_is_user_pro_reconciliation.sql), 20260726_billing_identity_
# containment.sql has already installed the SAME canonical body earlier in
# the loop — that migration file's own §2 literally is this CREATE OR REPLACE.
# So "new" below is the live public.is_user_pro(), and "old" is the retired
# body's SQL expression evaluated inline (never installed as a live function
# during replay, since it would just be immediately overwritten again by
# 20260726). This is mathematically identical to calling the old body as a
# function: it is a pure, side-effect-free SQL expression.
run_is_user_pro_reconciliation_contract_proofs() {
    RECON_PROOF_DIR="$(mktemp -d)"
    trap 'rm -rf "$RECON_PROOF_DIR"' RETURN

    echo
    echo "── prove is_user_pro equivalence matrix and rollback gates ─"

    psql "$PGURI" -v ON_ERROR_STOP=1 -q <<'SQL'
DO $$
DECLARE
    rec        record;
    old_result boolean;
    new_result boolean;
    test_uid   uuid;
    failures   text[] := '{}';
BEGIN
    FOR rec IN
        SELECT * FROM (VALUES
        -- case, grant_mode, sub_mode, bare_is_pro, expected_old, expected_new, label
        (1,  'null',   'none',    false, true,  true,
             'permanent grant, no subscription'),
        (2,  'future', 'none',    false, true,  true,
             'future-dated grant, no subscription'),
        (3,  'past',   'none',    false, false, false,
             'expired grant, no subscription'),
        (4,  'none',   'none',    false, false, false,
             'no grant, no subscription'),
        (5,  'null',   'expired', false, true,  true,
             'permanent grant, expired subscription'),
        (6,  'past',   'expired', false, false, false,
             'expired grant, expired subscription'),
        (7,  'none',   'expired', false, false, false,
             'no grant, expired subscription'),
        (8,  'future', 'none',    true,  true,  true,
             'bare is_pro covered by a future-dated grant'),
        (9,  'null',   'active',  false, true,  true,
             'permanent grant, active subscription'),
        (10, 'none',   'active',  false, false, true,
             'ALLOWED DIFF (a): active subscription alone, no grant'),
        (11, 'past',   'active',  false, false, true,
             'ALLOWED DIFF (a): expired grant, active subscription'),
        (12, 'future', 'active',  false, true,  true,
             'future-dated grant, active subscription'),
        (13, 'null',   'active',  true,  true,  true,
             'permanent grant + active subscription + bare is_pro'),
        (14, 'none',   'active',  true,  true,  true,
             'bare is_pro covered by active subscription alone')
        ) AS t(case_num, grant_mode, sub_mode, bare_is_pro,
               expected_old, expected_new, label)
    LOOP
        test_uid := ('00000000-0000-0000-0000-0000000001'
                     || lpad(rec.case_num::text, 2, '0'))::uuid;

        INSERT INTO auth.users (id, email)
        VALUES (test_uid, 'recon-matrix-' || rec.case_num || '@example.invalid')
        ON CONFLICT (id) DO NOTHING;

        -- handle_new_user() already auto-created a profiles row (AFTER INSERT
        -- trigger on auth.users, is_pro/is_pro_grant default false) the instant
        -- the auth.users row above committed. A plain INSERT here would hit its
        -- primary key and fail with a duplicate-key error every single case.
        INSERT INTO public.profiles (id, is_pro, is_pro_grant, pro_grant_expires_at)
        VALUES (
            test_uid,
            rec.bare_is_pro,
            rec.grant_mode <> 'none',
            CASE rec.grant_mode
                WHEN 'null'   THEN NULL
                WHEN 'future' THEN now() + interval '30 days'
                WHEN 'past'   THEN now() - interval '1 day'
                ELSE NULL
            END
        )
        ON CONFLICT (id) DO UPDATE
        SET is_pro = EXCLUDED.is_pro,
            is_pro_grant = EXCLUDED.is_pro_grant,
            pro_grant_expires_at = EXCLUDED.pro_grant_expires_at;

        IF rec.sub_mode = 'active' THEN
            INSERT INTO public.subscriptions (user_id, order_id, status, amount, expires_at)
            VALUES (test_uid, 'RECON-MATRIX-' || rec.case_num, 'active', 199,
                    now() + interval '30 days');
        ELSIF rec.sub_mode = 'expired' THEN
            -- Genuinely representative of production: nothing demotes status
            -- when expires_at passes (verified against real data in 6A-R.7).
            INSERT INTO public.subscriptions (user_id, order_id, status, amount, expires_at)
            VALUES (test_uid, 'RECON-MATRIX-' || rec.case_num, 'active', 199,
                    now() - interval '1 day');
        END IF;

        SELECT COALESCE(p.is_pro, false)
               OR (COALESCE(p.is_pro_grant, false)
                   AND (p.pro_grant_expires_at IS NULL OR p.pro_grant_expires_at > now()))
          INTO old_result
          FROM public.profiles p WHERE p.id = test_uid;

        SELECT public.is_user_pro(test_uid) INTO new_result;

        IF old_result IS DISTINCT FROM rec.expected_old
           OR new_result IS DISTINCT FROM rec.expected_new THEN
            failures := failures || format(
                'case %s (%s): expected old=%s new=%s, got old=%s new=%s',
                rec.case_num, rec.label,
                rec.expected_old, rec.expected_new, old_result, new_result);
        END IF;

        -- Self-contained: gone before the next case, and before the rollback
        -- gate tests below run (which must see zero active subscriptions).
        DELETE FROM auth.users WHERE id = test_uid;
    END LOOP;

    -- Case 15: no profiles row at all. old is a genuine SQL NULL (a
    -- zero-row scalar SELECT), new is a boolean EXISTS() that can never be
    -- NULL. Both are "not Pro" at every call site — see the migration header.
    test_uid := '00000000-0000-0000-0000-000000000115'::uuid;
    SELECT COALESCE(p.is_pro, false)
           OR (COALESCE(p.is_pro_grant, false)
               AND (p.pro_grant_expires_at IS NULL OR p.pro_grant_expires_at > now()))
      INTO old_result
      FROM public.profiles p WHERE p.id = test_uid;
    SELECT public.is_user_pro(test_uid) INTO new_result;
    IF old_result IS NOT NULL OR new_result IS DISTINCT FROM false THEN
        failures := failures || format(
            'case 15 (no profiles row): expected old=NULL new=false, got old=%s new=%s',
            old_result, new_result);
    END IF;

    IF array_length(failures, 1) > 0 THEN
        RAISE EXCEPTION 'is_user_pro equivalence matrix FAILED: %',
            array_to_string(failures, ' | ');
    END IF;
    RAISE NOTICE 'is_user_pro equivalence matrix: 15/15 cases verified '
                 '(13 identical, 2 allowed differences)';
END $$;
SQL
    echo "ok    is_user_pro equivalence matrix (15/15, zero unexplained divergence)"

    # ── orphan gate: bare is_pro=true with no covering grant or subscription
    # must never be silently downgraded — the migration must refuse to apply.
    psql "$PGURI" -v ON_ERROR_STOP=1 -q <<'SQL'
INSERT INTO auth.users (id, email)
VALUES ('00000000-0000-0000-0000-000000000120', 'recon-orphan@example.invalid')
ON CONFLICT (id) DO NOTHING;
-- handle_new_user() already created the row (is_pro/is_pro_grant default
-- false); UPDATE it into an orphan rather than INSERT, which would collide.
INSERT INTO public.profiles (id, is_pro, is_pro_grant)
VALUES ('00000000-0000-0000-0000-000000000120', true, false)
ON CONFLICT (id) DO UPDATE
SET is_pro = true, is_pro_grant = false, pro_grant_expires_at = NULL;
SQL
    expect_reconciliation_migration_failure "preflight_orphan_bare_is_pro"
    if [[ "$(psql "$PGURI" -v ON_ERROR_STOP=1 -qAt -c \
        "SELECT prosrc FROM pg_proc WHERE oid = to_regprocedure('public.is_user_pro(uuid)')" \
        | tr -d '[:space:]')" != *"FROMsubscriptionss"* ]]; then
        echo "orphan-gate rejection left is_user_pro on the wrong body" >&2
        return 1
    fi
    psql "$PGURI" -v ON_ERROR_STOP=1 -q -c \
        "DELETE FROM auth.users WHERE id = '00000000-0000-0000-0000-000000000120'"
    echo "ok    orphan-gate rejection left the canonical body untouched"

    # ── rollback gate, abort path: an active, unexpired subscription must
    # block the rollback outright — reverting would silently revoke a paying
    # customer's Pro mid-period.
    psql "$PGURI" -v ON_ERROR_STOP=1 -q <<'SQL'
INSERT INTO auth.users (id, email)
VALUES ('00000000-0000-0000-0000-000000000130', 'recon-rollback-guard@example.invalid')
ON CONFLICT (id) DO NOTHING;
-- No explicit profiles insert needed: handle_new_user() already created one
-- with is_pro/is_pro_grant false, which is exactly what this test needs —
-- only the subscription matters for the rollback gate being tested here.
INSERT INTO public.subscriptions (user_id, order_id, status, amount, expires_at)
VALUES ('00000000-0000-0000-0000-000000000130', 'RECON-ROLLBACK-GUARD', 'active',
        199, now() + interval '30 days');
SQL
    expect_reconciliation_rollback_failure "rollback_blocked_by_active_subscription"
    if [[ "$(psql "$PGURI" -v ON_ERROR_STOP=1 -qAt -c \
        "SELECT prosrc FROM pg_proc WHERE oid = to_regprocedure('public.is_user_pro(uuid)')" \
        | tr -d '[:space:]')" != *"FROMsubscriptionss"* ]]; then
        echo "blocked rollback nonetheless changed is_user_pro's body" >&2
        return 1
    fi
    psql "$PGURI" -v ON_ERROR_STOP=1 -q -c \
        "DELETE FROM auth.users WHERE id = '00000000-0000-0000-0000-000000000130'"
    echo "ok    rollback correctly refused while a paying subscription is active"

    # ── rollback gate, success path: with zero active subscriptions, rollback
    # must actually revert the body — then restore it, so the main loop's own
    # (upcoming) application of the forward file, and 20260730_ecpay_backend.sql's
    # preflight later in the same loop, see the canonical body as expected.
    if [[ "$(psql "$PGURI" -v ON_ERROR_STOP=1 -qAt -c \
        "SELECT count(*) FROM public.subscriptions WHERE status='active' AND expires_at > now()")" \
        != "0" ]]; then
        echo "test data leaked an active subscription before the rollback-success check" >&2
        return 1
    fi
    psql_run "$RECON_ROLLBACK"
    if [[ "$(psql "$PGURI" -v ON_ERROR_STOP=1 -qAt -c \
        "SELECT prosrc FROM pg_proc WHERE oid = to_regprocedure('public.is_user_pro(uuid)')" \
        | tr -d '[:space:]')" == *"FROMsubscriptionss"* ]]; then
        echo "rollback ran but did not revert to the bare is_pro body" >&2
        return 1
    fi
    echo "ok    rollback reverted the body when no active subscription exists"

    rm -rf "$RECON_PROOF_DIR"
    trap - RETURN
}

echo "── shim ──────────────────────────────────────────────"
psql_run "$HERE/00_local_shim.sql"
echo "ok  00_local_shim.sql"

echo
echo "── migrations (filename order) ───────────────────────"
# *_rollback.sql is excluded on purpose: it is the documented undo for
# 20260726_billing_identity_containment.sql, not a forward step. Replaying it
# in sequence would revert the migration that ran immediately before it.
for f in $(ls "$MIGRATIONS"/*.sql | sort); do
    base="$(basename "$f")"
    case "$base" in
        *_rollback.sql) echo "skip  $base  (rollback script, not a forward migration)"; continue ;;
    esac
    if [[ "$f" == "$RECON_MIGRATION" ]]; then
        run_is_user_pro_reconciliation_contract_proofs
    fi
    if [[ "$f" == "$ECPAY_MIGRATION" ]]; then
        run_ecpay_migration_contract_proofs
    fi
    psql_run "$f"
    echo "ok    $base"
done

echo
echo "── prove is_user_pro reconciliation is safely rerunnable ──"
psql_run "$RECON_MIGRATION"
echo "ok    20260730_0_is_user_pro_reconciliation.sql (complete rerun)"
if [[ "$(psql "$PGURI" -v ON_ERROR_STOP=1 -qAt -c \
    "SELECT prosrc FROM pg_proc WHERE oid = to_regprocedure('public.is_user_pro(uuid)')" \
    | tr -d '[:space:]')" != *"FROMsubscriptionss"* ]]; then
    echo "rerun of the reconciliation migration left the wrong body in place" >&2
    exit 1
fi

echo
echo "── prove complete ECPay migration is safely rerunnable "
psql_run "$ECPAY_MIGRATION"
echo "ok    20260730_ecpay_backend.sql (complete rerun)"

echo
echo "── assert required RPCs exist ───────────────────────"
psql "$PGURI" -v ON_ERROR_STOP=1 -t -A <<'SQL'
DO $$
DECLARE
    expected text[] := ARRAY['get_admin_users_full','get_admin_user_activity',
                             'get_user_id_by_email','get_admin_pro_breakdown',
                             'accept_ecpay_payment'];
    fn text;
    missing text[] := '{}';
BEGIN
    FOREACH fn IN ARRAY expected LOOP
        IF NOT EXISTS (
            SELECT 1 FROM pg_proc p JOIN pg_namespace n ON n.oid = p.pronamespace
            WHERE n.nspname = 'public' AND p.proname = fn
        ) THEN
            missing := missing || fn;
        END IF;
    END LOOP;
    IF array_length(missing, 1) > 0 THEN
        RAISE EXCEPTION 'MISSING RPCs: %', array_to_string(missing, ', ');
    END IF;
    RAISE NOTICE 'all required RPCs present';
END $$;
SQL

psql "$PGURI" -v ON_ERROR_STOP=1 -c "
select p.proname as rpc, pg_get_function_identity_arguments(p.oid) as args
from pg_proc p join pg_namespace n on n.oid = p.pronamespace
where n.nspname='public'
  and p.proname in ('get_admin_users_full','get_admin_user_activity',
                    'get_user_id_by_email','get_admin_pro_breakdown',
                    'accept_ecpay_payment')
order by p.proname;"

echo
echo "── assert ECPay acceptance is atomic and idempotent ──"
psql "$PGURI" -v ON_ERROR_STOP=1 <<'SQL'
DO $$
DECLARE
    uid constant uuid := '00000000-0000-0000-0000-000000000061';
    other_uid constant uuid := '00000000-0000-0000-0000-000000000062';
    first_order constant text := '20990101ATOMIC000001';
    failure_order constant text := '20990101ATOMIC000002';
    amount_order constant text := '20990101ATOMIC000003';
    user_order constant text := '20990101ATOMIC000004';
    outcome text;
    original_expiry timestamptz := '2099-02-01 00:00:00+00';
BEGIN
    INSERT INTO auth.users (id, email)
    VALUES
        (uid, 'ecpay-replay@example.invalid'),
        (other_uid, 'ecpay-replay-other@example.invalid')
    ON CONFLICT (id) DO NOTHING;

    INSERT INTO public.subscriptions (
        user_id, order_id, merchant_trade_no, status, amount
    ) VALUES (
        uid, first_order, first_order, 'pending', 199
    );

    SELECT public.accept_ecpay_payment(
        first_order, NULL, '1', 'Succeeded', '{"fixture":"accepted"}'::jsonb,
        'ECPAY0001', uid, 199,
        '2099-01-02 00:00:00+00', original_expiry
    ) INTO outcome;
    IF outcome <> 'activated' THEN
        RAISE EXCEPTION 'first acceptance returned %, expected activated', outcome;
    END IF;
    IF (SELECT status FROM public.subscriptions
        WHERE merchant_trade_no = first_order) <> 'active' THEN
        RAISE EXCEPTION 'first acceptance did not activate subscription';
    END IF;
    IF (SELECT expires_at FROM public.subscriptions
        WHERE merchant_trade_no = first_order) IS DISTINCT FROM original_expiry THEN
        RAISE EXCEPTION 'first acceptance set the wrong expiry';
    END IF;
    IF (SELECT count(*) FROM public.payment_events
        WHERE merchant_trade_no = first_order AND source = 'ecpay_callback') <> 1 THEN
        RAISE EXCEPTION 'first acceptance did not create exactly one ledger row';
    END IF;

    -- Exact duplicate: every identity input is unchanged.
    SELECT public.accept_ecpay_payment(
        first_order, NULL, '1', 'Succeeded', '{"fixture":"accepted"}'::jsonb,
        'ECPAY0001', uid, 199,
        '2099-01-02 00:00:00+00', original_expiry
    ) INTO outcome;
    IF outcome <> 'duplicate' THEN
        RAISE EXCEPTION 'exact duplicate returned %, expected duplicate', outcome;
    END IF;
    IF (SELECT count(*) FROM public.payment_events
        WHERE merchant_trade_no = first_order AND source = 'ecpay_callback') <> 1
       OR (SELECT status FROM public.subscriptions
           WHERE merchant_trade_no = first_order) <> 'active'
       OR (SELECT expires_at FROM public.subscriptions
           WHERE merchant_trade_no = first_order) IS DISTINCT FROM original_expiry THEN
        RAISE EXCEPTION 'exact duplicate changed ledger or entitlement';
    END IF;

    -- Even a changed TotalSuccessTimes cannot extend a one-off order.
    SELECT public.accept_ecpay_payment(
        first_order, 2, '1', 'Succeeded', '{"fixture":"duplicate"}'::jsonb,
        'ECPAY0001', uid, 199,
        '2099-02-02 00:00:00+00', '2099-03-01 00:00:00+00'
    ) INTO outcome;
    IF outcome <> 'duplicate' THEN
        RAISE EXCEPTION 'duplicate returned %, expected duplicate', outcome;
    END IF;
    IF (SELECT count(*) FROM public.payment_events
        WHERE merchant_trade_no = first_order AND source = 'ecpay_callback') <> 1 THEN
        RAISE EXCEPTION 'duplicate created a second accepted ledger row';
    END IF;
    IF (SELECT expires_at FROM public.subscriptions
        WHERE merchant_trade_no = first_order) IS DISTINCT FROM original_expiry THEN
        RAISE EXCEPTION 'duplicate extended the subscription';
    END IF;

    INSERT INTO public.subscriptions (
        user_id, order_id, merchant_trade_no, status, amount
    ) VALUES
        (uid, failure_order, failure_order, 'pending', 199),
        (uid, amount_order, amount_order, 'pending', 199),
        (uid, user_order, user_order, 'pending', 199);

    SELECT public.accept_ecpay_payment(
        amount_order, NULL, '1', 'Succeeded', '{"fixture":"wrong-amount"}'::jsonb,
        'ECPAY0003', uid, 1,
        '2099-01-02 00:00:00+00', original_expiry
    ) INTO outcome;
    IF outcome <> 'rejected'
       OR EXISTS (SELECT 1 FROM public.payment_events
                  WHERE merchant_trade_no = amount_order)
       OR (SELECT status FROM public.subscriptions
           WHERE merchant_trade_no = amount_order) <> 'pending'
       OR (SELECT expires_at FROM public.subscriptions
           WHERE merchant_trade_no = amount_order) IS NOT NULL THEN
        RAISE EXCEPTION 'wrong amount was not rejected without side effects';
    END IF;

    SELECT public.accept_ecpay_payment(
        user_order, NULL, '1', 'Succeeded', '{"fixture":"wrong-user"}'::jsonb,
        'ECPAY0004', other_uid, 199,
        '2099-01-02 00:00:00+00', original_expiry
    ) INTO outcome;
    IF outcome <> 'rejected'
       OR EXISTS (SELECT 1 FROM public.payment_events
                  WHERE merchant_trade_no = user_order)
       OR (SELECT status FROM public.subscriptions
           WHERE merchant_trade_no = user_order) <> 'pending'
       OR (SELECT expires_at FROM public.subscriptions
           WHERE merchant_trade_no = user_order) IS NOT NULL THEN
        RAISE EXCEPTION 'wrong user was not rejected without side effects';
    END IF;
END $$;

CREATE OR REPLACE FUNCTION public.replay_force_activation_failure()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.merchant_trade_no = '20990101ATOMIC000002' THEN
        RAISE EXCEPTION 'forced replay activation failure';
    END IF;
    RETURN NEW;
END
$$;
CREATE TRIGGER replay_force_activation_failure_trg
BEFORE UPDATE ON public.subscriptions
FOR EACH ROW EXECUTE FUNCTION public.replay_force_activation_failure();

DO $$
DECLARE
    uid constant uuid := '00000000-0000-0000-0000-000000000061';
    failure_order constant text := '20990101ATOMIC000002';
BEGIN
    BEGIN
        PERFORM public.accept_ecpay_payment(
            failure_order, NULL, '1', 'Succeeded',
            '{"fixture":"forced-failure"}'::jsonb,
            'ECPAY0002', uid, 199,
            '2099-01-02 00:00:00+00', '2099-02-01 00:00:00+00'
        );
        RAISE EXCEPTION 'acceptance unexpectedly survived forced update failure';
    EXCEPTION
        WHEN OTHERS THEN
            IF SQLERRM = 'acceptance unexpectedly survived forced update failure' THEN
                RAISE;
            END IF;
    END;

    IF EXISTS (
        SELECT 1 FROM public.payment_events
        WHERE merchant_trade_no = failure_order AND source = 'ecpay_callback'
    ) THEN
        RAISE EXCEPTION 'ledger insert survived a rolled-back activation';
    END IF;
    IF (SELECT status FROM public.subscriptions
        WHERE merchant_trade_no = failure_order) <> 'pending' THEN
        RAISE EXCEPTION 'failed activation changed subscription state';
    END IF;
    IF (SELECT expires_at FROM public.subscriptions
        WHERE merchant_trade_no = failure_order) IS NOT NULL THEN
        RAISE EXCEPTION 'failed activation set an expiry';
    END IF;
END $$;

DROP TRIGGER replay_force_activation_failure_trg ON public.subscriptions;
DROP FUNCTION public.replay_force_activation_failure();
SQL

echo
echo "── assert RPC security and precise conflict handling ─"
psql "$PGURI" -v ON_ERROR_STOP=1 <<'SQL'
DO $$
DECLARE
    fn regprocedure := to_regprocedure(
        'public.accept_ecpay_payment(text,integer,text,text,jsonb,text,uuid,integer,timestamp with time zone,timestamp with time zone)'
    );
    is_definer boolean;
    settings text[];
BEGIN
    IF fn IS NULL THEN
        RAISE EXCEPTION 'accept_ecpay_payment signature not found';
    END IF;
    SELECT prosecdef, proconfig INTO is_definer, settings
      FROM pg_proc WHERE oid = fn;
    IF NOT is_definer THEN
        RAISE EXCEPTION 'accept_ecpay_payment is not SECURITY DEFINER';
    END IF;
    IF settings IS NULL OR NOT settings @> ARRAY['search_path=public'] THEN
        RAISE EXCEPTION 'accept_ecpay_payment search_path is not fixed to public';
    END IF;
    IF EXISTS (
        SELECT 1
          FROM pg_proc p
          CROSS JOIN LATERAL aclexplode(p.proacl) acl
         WHERE p.oid = fn
           AND acl.privilege_type = 'EXECUTE'
           AND acl.grantee IN (0, 'anon'::regrole::oid,
                               'authenticated'::regrole::oid)
    ) THEN
        RAISE EXCEPTION 'untrusted role can execute accept_ecpay_payment';
    END IF;
    IF NOT EXISTS (
        SELECT 1
          FROM pg_proc p
          CROSS JOIN LATERAL aclexplode(p.proacl) acl
         WHERE p.oid = fn
           AND acl.privilege_type = 'EXECUTE'
           AND acl.grantee = 'service_role'::regrole::oid
    ) THEN
        RAISE EXCEPTION 'service_role cannot execute accept_ecpay_payment';
    END IF;
END $$;

CREATE UNIQUE INDEX replay_unrelated_payment_event_unique
    ON public.payment_events (rtn_msg)
    WHERE rtn_msg = 'UNRELATED_UNIQUE';

DO $$
DECLARE
    uid constant uuid := '00000000-0000-0000-0000-000000000061';
    unrelated_order constant text := '20990101ATOMIC000005';
    violated text;
BEGIN
    INSERT INTO public.payment_events (
        source, merchant_trade_no, rtn_code, rtn_msg, checkmac_valid, raw_payload
    ) VALUES (
        'reconciliation', 'UNRELATEDSEED', '1', 'UNRELATED_UNIQUE', TRUE, '{}'
    );
    INSERT INTO public.subscriptions (
        user_id, order_id, merchant_trade_no, status, amount
    ) VALUES (
        uid, unrelated_order, unrelated_order, 'pending', 199
    );

    BEGIN
        PERFORM public.accept_ecpay_payment(
            unrelated_order, NULL, '1', 'UNRELATED_UNIQUE',
            '{"fixture":"unrelated-unique"}'::jsonb,
            'ECPAY0005', uid, 199,
            '2099-01-02 00:00:00+00', '2099-02-01 00:00:00+00'
        );
        RAISE EXCEPTION 'unrelated unique violation was swallowed as duplicate';
    EXCEPTION
        WHEN unique_violation THEN
            GET STACKED DIAGNOSTICS violated = CONSTRAINT_NAME;
            IF violated <> 'replay_unrelated_payment_event_unique' THEN
                RAISE EXCEPTION 'wrong unique violation surfaced: %', violated;
            END IF;
    END;

    IF EXISTS (SELECT 1 FROM public.payment_events
               WHERE merchant_trade_no = unrelated_order)
       OR (SELECT status FROM public.subscriptions
           WHERE merchant_trade_no = unrelated_order) <> 'pending' THEN
        RAISE EXCEPTION 'unrelated unique failure left partial acceptance state';
    END IF;
END $$;

DROP INDEX public.replay_unrelated_payment_event_unique;
SQL

echo
echo "── assert concurrent duplicate acceptance ────────────"
psql "$PGURI" -v ON_ERROR_STOP=1 <<'SQL'
INSERT INTO public.subscriptions (
    user_id, order_id, merchant_trade_no, status, amount
) VALUES (
    '00000000-0000-0000-0000-000000000061',
    '20990101ATOMIC000006', '20990101ATOMIC000006', 'pending', 199
);
SQL

CONCURRENCY_DIR="$(mktemp -d)"
trap 'rm -rf "$CONCURRENCY_DIR"' EXIT
RPC_SQL="select public.accept_ecpay_payment(
    '20990101ATOMIC000006', NULL, '1', 'Succeeded',
    '{\"fixture\":\"concurrent\"}'::jsonb, 'ECPAY0006',
    '00000000-0000-0000-0000-000000000061', 199,
    '2099-01-02 00:00:00+00', '2099-02-01 00:00:00+00'
);"
psql "$PGURI" -v ON_ERROR_STOP=1 -qAt -c "$RPC_SQL" >"$CONCURRENCY_DIR/one" &
pid1=$!
psql "$PGURI" -v ON_ERROR_STOP=1 -qAt -c "$RPC_SQL" >"$CONCURRENCY_DIR/two" &
pid2=$!
wait "$pid1"
wait "$pid2"
results="$(sort "$CONCURRENCY_DIR/one" "$CONCURRENCY_DIR/two" | tr '\n' ' ')"
if [[ "$results" != "activated duplicate " ]]; then
    echo "concurrent outcomes were: $results" >&2
    exit 1
fi
rm -rf "$CONCURRENCY_DIR"
trap - EXIT

psql "$PGURI" -v ON_ERROR_STOP=1 <<'SQL'
DO $$
BEGIN
    IF (SELECT count(*) FROM public.payment_events
        WHERE merchant_trade_no = '20990101ATOMIC000006'
          AND source = 'ecpay_callback') <> 1
       OR (SELECT status FROM public.subscriptions
           WHERE merchant_trade_no = '20990101ATOMIC000006') <> 'active'
       OR (SELECT expires_at FROM public.subscriptions
           WHERE merchant_trade_no = '20990101ATOMIC000006')
          IS DISTINCT FROM '2099-02-01 00:00:00+00'::timestamptz THEN
        RAISE EXCEPTION 'concurrent duplicate changed ledger or entitlement';
    END IF;
END $$;
SQL

echo
echo "── assert the eight previously-unbacked tables exist ──"
psql "$PGURI" -v ON_ERROR_STOP=1 -c "
select tablename,
       (select count(*) from pg_policies g where g.schemaname='public' and g.tablename=t.tablename) as policies,
       c.relrowsecurity as rls
from pg_tables t join pg_class c on c.relname=t.tablename
join pg_namespace n on n.oid=c.relnamespace and n.nspname='public'
where t.schemaname='public'
  and t.tablename in ('practice_records','profiles','questions','writing_questions',
                      'writing_submissions','upgrade_intent','rec_log','pro_waitlist')
order by tablename;"

echo
echo "── assert content lockdown by ROLE, not by catalog ────"
# Catalogs are an index, not evidence. Every verdict below is re-derived by
# actually attempting the read as anon / authenticated / service_role.
#
# The reading_questions cases are the load-bearing ones: this file must prove
# the lockdown did NOT disturb the column-level GRANT from 20260714. On
# 2026-07-30 an audit misread that grant as a leak and nearly deleted it.
psql "$PGURI" -v ON_ERROR_STOP=1 <<'SQL'
DO $$
DECLARE
    tbl      text;
    n        bigint;
    failures text[] := '{}';
BEGIN
    -- ── anon must reach nothing on the three content tables ──────────────
    FOREACH tbl IN ARRAY ARRAY['questions','reading_passages','writing_questions'] LOOP
        BEGIN
            SET LOCAL ROLE anon;
            EXECUTE format('SELECT count(*) FROM public.%I', tbl) INTO n;
            failures := failures || format('anon READ %s -> ALLOWED (%s rows)', tbl, n);
        EXCEPTION WHEN insufficient_privilege THEN
            RAISE NOTICE 'anon          %-20s -> DENIED 42501       [expected]', tbl;
        END;
        RESET ROLE;
    END LOOP;

    -- ── authenticated must reach nothing either ──────────────────────────
    FOREACH tbl IN ARRAY ARRAY['questions','reading_passages','writing_questions'] LOOP
        BEGIN
            SET LOCAL ROLE authenticated;
            EXECUTE format('SELECT count(*) FROM public.%I', tbl) INTO n;
            failures := failures || format('authenticated READ %s -> ALLOWED (%s rows)', tbl, n);
        EXCEPTION WHEN insufficient_privilege THEN
            RAISE NOTICE 'authenticated %-20s -> DENIED 42501       [expected]', tbl;
        END;
        RESET ROLE;
    END LOOP;

    -- ── zero policies: RLS on with USING(true) attached is not a lockdown ─
    SELECT count(*) INTO n FROM pg_policies
     WHERE schemaname='public'
       AND tablename IN ('questions','reading_passages','writing_questions');
    IF n <> 0 THEN
        failures := failures || format('%s policy/policies survived on the content tables', n);
    ELSE
        RAISE NOTICE 'policies on the three content tables            -> 0            [expected]';
    END IF;

    -- ── RLS actually enabled (the other half of the two layers) ──────────
    SELECT count(*) INTO n FROM pg_class c
      JOIN pg_namespace ns ON ns.oid=c.relnamespace AND ns.nspname='public'
     WHERE c.relname IN ('questions','reading_passages','writing_questions')
       AND c.relrowsecurity;
    IF n <> 3 THEN
        failures := failures || format('RLS enabled on only %s/3 content tables', n);
    ELSE
        RAISE NOTICE 'RLS enabled on the three content tables         -> 3/3          [expected]';
    END IF;

    -- ── service_role keeps working, or FastAPI is dead ───────────────────
    FOREACH tbl IN ARRAY ARRAY['questions','reading_passages','writing_questions',
                               'reading_questions'] LOOP
        BEGIN
            SET LOCAL ROLE service_role;
            EXECUTE format('SELECT count(*) FROM public.%I', tbl) INTO n;
            RAISE NOTICE 'service_role  %-20s -> ALLOWED            [expected]', tbl;
        EXCEPTION WHEN insufficient_privilege THEN
            failures := failures || format('service_role READ %s -> DENIED (FastAPI would break)', tbl);
        END;
        RESET ROLE;
    END LOOP;

    -- ── reading_questions: UNTOUCHED. Answers denied ─────────────────────
    BEGIN
        SET LOCAL ROLE authenticated;
        EXECUTE 'SELECT count(*) FROM public.reading_questions WHERE correct_answer IS NOT NULL' INTO n;
        failures := failures || 'authenticated READ reading_questions.correct_answer -> ALLOWED (ANSWER LEAK)';
    EXCEPTION WHEN insufficient_privilege THEN
        RAISE NOTICE 'authenticated correct_answer                    -> DENIED 42501 [expected]';
    END;
    RESET ROLE;

    -- ── reading_questions: UNTOUCHED. Answering path still works ─────────
    BEGIN
        SET LOCAL ROLE authenticated;
        EXECUTE 'SELECT count(*) FROM (SELECT id, passage_id, question_type, question_text,'
                ' options, order_idx, created_at FROM public.reading_questions) q' INTO n;
        RAISE NOTICE 'authenticated reading_questions safe columns    -> ALLOWED      [expected]';
    EXCEPTION WHEN insufficient_privilege THEN
        failures := failures || 'authenticated READ reading_questions safe columns -> DENIED '
                             || '(the lockdown broke the answering path)';
    END;
    RESET ROLE;

    IF array_length(failures, 1) > 0 THEN
        RAISE EXCEPTION 'content lockdown verification failed: %',
            array_to_string(failures, ' | ');
    END IF;
    RAISE NOTICE 'content lockdown verified by role';
END $$;
SQL

echo
echo "── prove content lockdown is safely rerunnable ────────"
psql_run "$MIGRATIONS/20260731_content_access_lockdown.sql"
echo "ok    20260731_content_access_lockdown.sql (complete rerun)"

echo
echo "── prove content lockdown rollback re-opens, then relock"
psql_run "$MIGRATIONS/20260731_content_access_lockdown_rollback.sql"
psql "$PGURI" -v ON_ERROR_STOP=1 <<'SQL'
DO $$
DECLARE n bigint;
BEGIN
    -- The rollback must genuinely re-open, or its warning banner is a lie.
    SET LOCAL ROLE anon;
    SELECT count(*) INTO n FROM public.questions;
    RESET ROLE;
    RAISE NOTICE 'after rollback: anon reads questions            -> ALLOWED (%s rows)', n;

    -- ...and must NOT have touched reading_questions.
    BEGIN
        SET LOCAL ROLE authenticated;
        PERFORM correct_answer FROM public.reading_questions LIMIT 1;
        RAISE EXCEPTION 'rollback exposed reading_questions.correct_answer';
    EXCEPTION WHEN insufficient_privilege THEN
        RAISE NOTICE 'after rollback: correct_answer                  -> still DENIED [expected]';
    END;
    RESET ROLE;
END $$;
SQL
psql_run "$MIGRATIONS/20260731_content_access_lockdown.sql"
echo "ok    rollback verified, re-locked"

echo
python3 "$HERE/test_public_vocabulary_access.py" --disposable
python3 "$HERE/test_vocabulary_publication.py" --disposable
python3 "$HERE/test_atomic_vocabulary_quota.py" --disposable
echo "REPLAY OK"
