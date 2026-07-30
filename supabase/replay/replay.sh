#!/usr/bin/env bash
# Replay supabase/migrations/ into an empty Postgres and assert the four admin
# RPCs get built. Used both locally and by the migration-replay CI job.
#
# Usage: PGURI=postgresql://... ./supabase/replay/replay.sh
set -euo pipefail

: "${PGURI:?PGURI must be set (e.g. postgresql://postgres:postgres@localhost:5432/postgres)}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MIGRATIONS="$HERE/../migrations"

psql_run() { psql "$PGURI" -v ON_ERROR_STOP=1 -q -f "$1"; }

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
    psql_run "$f"
    echo "ok    $base"
done

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
echo "REPLAY OK"
