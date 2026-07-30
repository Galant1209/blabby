-- TASK 6A-R.4 production preflight (read-only)
--
-- Run this against the production PostgreSQL database before requesting
-- approval to apply 20260730_ecpay_backend.sql.
--
-- This script intentionally contains no DDL or DML. The transaction is
-- explicitly READ ONLY, so PostgreSQL will reject writes even if the script is
-- changed accidentally during an interactive session.

BEGIN TRANSACTION READ ONLY;
SET LOCAL statement_timeout = '15s';
SET LOCAL lock_timeout = '3s';

-- 1. Connection identity. Verify this is the intended production database.
SELECT
    current_database() AS database_name,
    current_user AS database_role,
    current_setting('server_version') AS postgres_version,
    now() AS checked_at;

-- All three dependency objects must resolve as ordinary/partitioned tables.
SELECT
    required.relation,
    c.relkind,
    CASE WHEN c.relkind IN ('r', 'p') THEN 'ok' ELSE 'FAIL' END AS result
FROM (VALUES
    ('public.profiles'),
    ('public.subscriptions'),
    ('public.payment_events')
) AS required(relation)
LEFT JOIN pg_class c ON c.oid = to_regclass(required.relation);

-- 2. Profile and entitlement counts.
SELECT
    count(*) AS profiles_total,
    count(*) FILTER (WHERE COALESCE(is_pro, false)) AS is_pro_true,
    count(*) FILTER (WHERE COALESCE(is_pro_grant, false)) AS is_pro_grant_true,
    count(*) FILTER (
        WHERE COALESCE(is_pro_grant, false)
          AND (pro_grant_expires_at IS NULL OR pro_grant_expires_at > now())
    ) AS active_grants
FROM public.profiles;

-- Must return zero. These are the users that the 20260726 is_user_pro()
-- change would remove if that dependency has not already been applied.
SELECT count(*) AS users_at_risk
FROM public.profiles p
WHERE COALESCE(p.is_pro, false)
  AND NOT (
        COALESCE(p.is_pro_grant, false)
        AND (p.pro_grant_expires_at IS NULL OR p.pro_grant_expires_at > now())
  )
  AND NOT EXISTS (
        SELECT 1
        FROM public.subscriptions s
        WHERE s.user_id = p.id
          AND s.status = 'active'
          AND s.expires_at > now()
  );

-- Every active grant must remain entitled under the current is_user_pro().
-- Must return zero if the 20260726 function body is installed correctly.
SELECT count(*) AS active_grants_not_entitled
FROM public.profiles p
WHERE COALESCE(p.is_pro_grant, false)
  AND (p.pro_grant_expires_at IS NULL OR p.pro_grant_expires_at > now())
  AND NOT public.is_user_pro(p.id);

-- 3. Subscription truth and current columns.
SELECT status, count(*) AS rows
FROM public.subscriptions
GROUP BY status
ORDER BY status;

SELECT
    count(*) AS subscriptions_total,
    count(*) FILTER (
        WHERE status = 'active' AND expires_at > now()
    ) AS active_unexpired,
    count(*) FILTER (WHERE amount IS NULL) AS amount_null,
    count(*) FILTER (WHERE amount IS NOT NULL AND amount <> 199) AS amount_not_199
FROM public.subscriptions;

SELECT
    ordinal_position,
    column_name,
    data_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'subscriptions'
ORDER BY ordinal_position;

-- Compare the exact columns consumed by the migration/RPC. Every row must say
-- ok. Extra unrelated subscription columns are allowed.
WITH expected(column_name, data_type, is_nullable) AS (
    VALUES
        ('id',         'uuid',                     'NO'),
        ('user_id',    'uuid',                     'NO'),
        ('order_id',   'text',                     'YES'),
        ('plan',       'text',                     'NO'),
        ('status',     'text',                     'NO'),
        ('amount',     'integer',                  'YES'),
        ('started_at', 'timestamp with time zone', 'YES'),
        ('expires_at', 'timestamp with time zone', 'YES'),
        ('created_at', 'timestamp with time zone', 'YES'),
        ('updated_at', 'timestamp with time zone', 'YES')
)
SELECT
    e.column_name,
    e.data_type AS expected_type,
    c.data_type AS actual_type,
    e.is_nullable AS expected_nullable,
    c.is_nullable AS actual_nullable,
    CASE
        WHEN c.column_name IS NULL THEN 'FAIL: missing'
        WHEN (c.data_type, c.is_nullable) =
             (e.data_type, e.is_nullable) THEN 'ok'
        ELSE 'FAIL: incompatible'
    END AS result
FROM expected e
LEFT JOIN information_schema.columns c
  ON c.table_schema = 'public'
 AND c.table_name = 'subscriptions'
 AND c.column_name = e.column_name
ORDER BY e.column_name;

-- Before 20260730, both ECPay columns should normally be absent. A complete
-- rerun may show nullable text columns, but merchant_trade_no must then also
-- have the exact unique partial index shown by the following query.
SELECT
    column_name,
    data_type,
    is_nullable
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'subscriptions'
  AND column_name IN ('merchant_trade_no', 'ecpay_trade_no')
ORDER BY column_name;

SELECT
    i.indexrelid::regclass AS index_name,
    i.indisunique,
    i.indisvalid,
    i.indisready,
    pg_get_indexdef(i.indexrelid) AS definition
FROM pg_index i
WHERE i.indexrelid =
      to_regclass('public.subscriptions_merchant_trade_no_uniq');

-- 4. payment_events dependency installed by 20260726.
SELECT
    to_regclass('public.payment_events') AS payment_events_relation,
    (SELECT count(*) FROM public.payment_events) AS payment_events_rows;

SELECT
    c.conname,
    pg_get_constraintdef(c.oid) AS definition
FROM pg_constraint c
WHERE c.conrelid = 'public.payment_events'::regclass
  AND c.conname = 'payment_events_source_check';

-- Must be exactly the four-value 20260726 set, or the five-value set after a
-- complete 20260730 application. Any other array is incompatible.
SELECT array_agg(match[1] ORDER BY match[1]) AS allowed_source_values
FROM pg_constraint c
CROSS JOIN LATERAL regexp_matches(
    pg_get_constraintdef(c.oid), $regex$'([^']+)'$regex$, 'g'
) AS match
WHERE c.conrelid = 'public.payment_events'::regclass
  AND c.conname = 'payment_events_source_check';

-- Must show payment_events_immutable_trg enabled and firing BEFORE UPDATE OR
-- DELETE through public.payment_events_immutable().
SELECT
    t.tgname,
    t.tgenabled,
    t.tgtype,
    pg_get_triggerdef(t.oid) AS definition,
    p.oid::regprocedure AS function_signature
FROM pg_trigger t
JOIN pg_proc p ON p.oid = t.tgfoid
WHERE t.tgrelid = 'public.payment_events'::regclass
  AND NOT t.tgisinternal
ORDER BY t.tgname;

-- Must be unique, valid, ready, and indnullsnotdistinct=true.
SELECT
    i.indexrelid::regclass AS index_name,
    i.indisunique,
    i.indisvalid,
    i.indisready,
    i.indnullsnotdistinct,
    pg_get_indexdef(i.indexrelid) AS definition
FROM pg_index i
WHERE i.indexrelid = to_regclass('public.payment_events_idem_uniq');

-- 5. Exact current dependency function. Review the returned body for:
-- active grant OR active, unexpired subscription; no bare profiles.is_pro.
SELECT
    p.oid::regprocedure AS signature,
    p.prosecdef AS security_definer,
    p.provolatile AS volatility,
    p.proconfig AS function_config,
    pg_get_functiondef(p.oid) AS definition
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = 'public'
  AND p.oid = to_regprocedure('public.is_user_pro(uuid)');

-- Supporting 20260726 identity-containment evidence.
SELECT
    c.relname AS view_name,
    c.reloptions
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
  AND c.relname IN ('user_pro_status', 'user_lookup')
ORDER BY c.relname;

-- 6. Detect absent, expected, or partially-applied acceptance RPC.
-- Before 20260730 this should return no rows. If it exists, its identity
-- arguments must exactly match the migration and it must be SECURITY DEFINER
-- with search_path=public.
SELECT
    p.oid::regprocedure AS signature,
    pg_get_function_identity_arguments(p.oid) AS identity_arguments,
    p.prosecdef AS security_definer,
    p.proconfig AS function_config,
    pg_get_userbyid(p.proowner) AS owner,
    pg_get_functiondef(p.oid) AS definition
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = 'public'
  AND p.proname = 'accept_ecpay_payment';

-- Expected before migration: total_overloads=0. Expected after a complete
-- migration: total_overloads=1 and expected_signatures=1. Any other combination
-- is a collision and the executable gate will abort.
SELECT
    count(*) AS total_overloads,
    count(*) FILTER (
        WHERE p.oid = to_regprocedure(
            'public.accept_ecpay_payment(text,integer,text,text,jsonb,text,uuid,integer,timestamp with time zone,timestamp with time zone)'
        )
    ) AS expected_signatures
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = 'public'
  AND p.proname = 'accept_ecpay_payment';

-- If the RPC already exists, public/anon/authenticated must be false and
-- service_role must be true. With no RPC this query returns no rows.
SELECT
    p.oid::regprocedure AS signature,
    has_function_privilege('public', p.oid, 'EXECUTE') AS public_execute,
    has_function_privilege('anon', p.oid, 'EXECUTE') AS anon_execute,
    has_function_privilege('authenticated', p.oid, 'EXECUTE') AS authenticated_execute,
    has_function_privilege('service_role', p.oid, 'EXECUTE') AS service_role_execute
FROM pg_proc p
JOIN pg_namespace n ON n.oid = p.pronamespace
WHERE n.nspname = 'public'
  AND p.proname = 'accept_ecpay_payment';

ROLLBACK;
