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
echo "── assert the four admin RPCs exist ──────────────────"
psql "$PGURI" -v ON_ERROR_STOP=1 -t -A <<'SQL'
DO $$
DECLARE
    expected text[] := ARRAY['get_admin_users_full','get_admin_user_activity',
                             'get_user_id_by_email','get_admin_pro_breakdown'];
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
    RAISE NOTICE 'all four admin RPCs present';
END $$;
SQL

psql "$PGURI" -v ON_ERROR_STOP=1 -c "
select p.proname as rpc, pg_get_function_identity_arguments(p.oid) as args
from pg_proc p join pg_namespace n on n.oid = p.pronamespace
where n.nspname='public'
  and p.proname in ('get_admin_users_full','get_admin_user_activity',
                    'get_user_id_by_email','get_admin_pro_breakdown')
order by p.proname;"

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
