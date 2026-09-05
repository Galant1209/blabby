#!/usr/bin/env python3
"""PG17 role/ACL/owner behavior proof. Disposable database ONLY; run by replay.sh."""
import os
from pathlib import Path
import subprocess
import sys
from test_atomic_vocabulary_quota import sql, SIGNATURE

MIGRATION = Path(__file__).parents[1] / 'migrations/20260905120000_vocabulary_public_access_lockdown.sql'


def apply():
    result = subprocess.run(['psql', os.environ['PGURI'], '-Xq', '-v', 'ON_ERROR_STOP=1', '-f', str(MIGRATION)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def preserved():
    return sql(f"""SELECT json_build_object(
        'rpc', pg_get_functiondef('{SIGNATURE}'::regprocedure),
        'rpc_acl', (SELECT proacl FROM pg_proc WHERE oid='{SIGNATURE}'::regprocedure),
        'owned_acl', (SELECT json_agg(row_to_json(t)) FROM (
            SELECT relname, relacl, relrowsecurity FROM pg_class WHERE oid IN
            ('public.user_vocabulary'::regclass, 'public.vocabulary_review_logs'::regclass)) t),
        'owned_policy', (SELECT json_agg(row_to_json(t)) FROM (
            SELECT * FROM pg_policies WHERE schemaname='public' AND tablename IN
            ('user_vocabulary', 'vocabulary_review_logs') ORDER BY tablename, policyname) t));""")


def main():
    assert sys.argv[1:] == ['--disposable'], 'Disposable DB acknowledgement required'
    assert sql('SHOW server_version_num;').startswith('17')
    before = preserved()
    # Reproduce both raw broad access and independent PUBLIC column grants.
    sql('''GRANT SELECT ON public.vocabulary_items TO PUBLIC, anon, authenticated;
           GRANT SELECT(id, word, tags) ON public.vocabulary_items TO PUBLIC, anon, authenticated;
           CREATE POLICY "Anyone can read vocabulary_items" ON public.vocabulary_items FOR SELECT USING(true);''')
    for role in ('anon', 'authenticated'):
        sql(f'SET ROLE {role}; SELECT word FROM public.vocabulary_items LIMIT 1;')
    print('PASS baseline fixture: anon/authenticated raw reads allowed', flush=True)
    apply()
    for role in ('anon', 'authenticated'):
        for projection in ('*', 'id,word', 'tags'):
            result = sql(f'SET ROLE {role}; SELECT {projection} FROM public.vocabulary_items LIMIT 1;', check=False)
            assert result.returncode != 0 and 'permission denied for table vocabulary_items' in result.stderr
        assert sql(f"SELECT has_any_column_privilege('{role}', 'public.vocabulary_items', 'SELECT');") == 'f'
        print(f'PASS {role}: full/safe/internal column SELECT denied including PUBLIC grants', flush=True)
    sql('SET ROLE service_role; SELECT * FROM public.vocabulary_items LIMIT 1;')
    print('PASS service_role reads corpus', flush=True)
    assert preserved() == before
    print('PASS owned ACL/RLS and atomic RPC definition/EXECUTE unchanged', flush=True)
    # Real ownership and backend join tests; all fixtures and shim changes roll back.
    sql('''BEGIN;
        CREATE OR REPLACE FUNCTION auth.uid() RETURNS uuid LANGUAGE sql STABLE AS
          $$ SELECT nullif(current_setting('request.jwt.claim.sub', true), '')::uuid $$;
        GRANT SELECT, UPDATE ON public.user_vocabulary TO authenticated;
        GRANT SELECT ON public.vocabulary_review_logs TO authenticated;
        INSERT INTO auth.users(id) VALUES
          ('00000000-0000-0000-0000-000000000091'), ('00000000-0000-0000-0000-000000000092');
        INSERT INTO public.vocabulary_items(id,word,zh_meaning) VALUES
          ('00000000-0000-0000-0000-000000000093','round-n-fixture','synthetic');
        INSERT INTO public.user_vocabulary(user_id,vocabulary_item_id) VALUES
          ('00000000-0000-0000-0000-000000000091','00000000-0000-0000-0000-000000000093'),
          ('00000000-0000-0000-0000-000000000092','00000000-0000-0000-0000-000000000093');
        INSERT INTO public.vocabulary_review_logs(user_id,user_vocabulary_id,vocabulary_item_id,review_type,result)
          SELECT user_id,id,vocabulary_item_id,'flashcard','correct' FROM public.user_vocabulary
          WHERE vocabulary_item_id='00000000-0000-0000-0000-000000000093';
        SET LOCAL request.jwt.claim.sub = '00000000-0000-0000-0000-000000000091';
        SET LOCAL ROLE authenticated;
        DO $$ DECLARE n integer; BEGIN
          SELECT count(*) INTO n FROM public.user_vocabulary;
          IF n <> 1 THEN RAISE EXCEPTION 'owner isolation failed'; END IF;
          SELECT count(*) INTO n FROM public.vocabulary_review_logs;
          IF n <> 1 THEN RAISE EXCEPTION 'review isolation failed'; END IF;
          UPDATE public.user_vocabulary SET review_count=7;
          GET DIAGNOSTICS n = ROW_COUNT;
          IF n <> 1 THEN RAISE EXCEPTION 'SRS owner update failed'; END IF;
        END $$;
        RESET ROLE;
        SET LOCAL ROLE service_role;
        DO $$ DECLARE n integer; BEGIN
          SELECT count(*) INTO n FROM public.user_vocabulary u
            JOIN public.vocabulary_items v ON u.vocabulary_item_id=v.id
            WHERE u.user_id='00000000-0000-0000-0000-000000000091' AND u.review_count=7;
          IF n <> 1 THEN RAISE EXCEPTION 'backend joined owner read failed'; END IF;
          SELECT count(*) INTO n FROM public.user_vocabulary
            WHERE user_id='00000000-0000-0000-0000-000000000092' AND review_count=0;
          IF n <> 1 THEN RAISE EXCEPTION 'other owner was modified'; END IF;
        END $$;
        RESET ROLE;
        ROLLBACK;''')
    print('PASS owner SELECT, SRS UPDATE, review isolation and backend service-role join', flush=True)
    apply()
    assert preserved() == before
    print('PASS idempotent rerun preserves owned/RPC contract', flush=True)
    print('PUBLIC VOCABULARY DB CONTRACT: PASS (7 proof groups)', flush=True)


if __name__ == '__main__':
    main()
