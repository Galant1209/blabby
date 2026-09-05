#!/usr/bin/env python3
"""Publication/default/ACL behavior on disposable PG17, never production."""
import os
from pathlib import Path
import subprocess
import sys
from test_atomic_vocabulary_quota import sql, literal
from test_public_vocabulary_access import preserved

MIGRATION = Path(__file__).parents[1] / 'migrations/20260905140000_vocabulary_publication_eligibility.sql'


def apply():
    result = subprocess.run(['psql', os.environ['PGURI'], '-Xq', '-v', 'ON_ERROR_STOP=1',
                             '-f', str(MIGRATION)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def main():
    assert sys.argv[1:] == ['--disposable']
    assert sql('SHOW server_version_num;').startswith('17')
    before = preserved()
    # Reproduce an upgrade from the pre-P schema with unknown existing content.
    # Everything here is in the explicitly disposable replay database.
    sql('ALTER TABLE public.vocabulary_items DROP COLUMN is_public;')
    sql("INSERT INTO public.vocabulary_items(word,zh_meaning) VALUES ('round-p-unknown','synthetic');")
    apply()
    assert sql("SELECT is_public FROM public.vocabulary_items WHERE word='round-p-unknown';") == 'f'
    assert preserved() == before
    print('PASS existing unknown rows default unpublished; owned/RPC contracts unchanged', flush=True)

    # Independent column grants plus permissive RLS must not enable publication.
    sql('''GRANT INSERT, UPDATE ON public.vocabulary_items TO PUBLIC, anon, authenticated;
        GRANT INSERT(is_public), UPDATE(is_public) ON public.vocabulary_items TO PUBLIC, anon, authenticated;
        CREATE POLICY round_p_write_fixture ON public.vocabulary_items FOR ALL USING(true) WITH CHECK(true);''')
    apply()
    for role in ('anon', 'authenticated'):
        for command in (
            "SELECT * FROM public.vocabulary_items",
            "SELECT is_public FROM public.vocabulary_items",
            "UPDATE public.vocabulary_items SET is_public=true",
            "INSERT INTO public.vocabulary_items(word,zh_meaning,is_public) VALUES ('rogue','rogue',true)",
        ):
            result = sql(f'SET ROLE {role}; {command};', check=False)
            assert result.returncode != 0 and 'permission denied' in result.stderr
        print(f'PASS {role}: raw SELECT and publication INSERT/UPDATE denied', flush=True)
    sql('DROP POLICY round_p_write_fixture ON public.vocabulary_items;')

    sql("SET ROLE service_role; INSERT INTO public.vocabulary_items(word,zh_meaning) VALUES ('round-p-default','synthetic');")
    assert sql("SELECT is_public FROM public.vocabulary_items WHERE word='round-p-default';") == 'f'
    sql("SET ROLE service_role; UPDATE public.vocabulary_items SET is_public=true WHERE word='round-p-default';")
    assert sql("SET ROLE service_role; SELECT word FROM public.vocabulary_items WHERE is_public=true AND word LIKE 'round-p-%';") == 'round-p-default'
    apply()
    assert sql("SELECT is_public FROM public.vocabulary_items WHERE word='round-p-default';") == 't'
    assert preserved() == before
    print('PASS explicit trusted publication, default false, rerun preserves decisions and RPC', flush=True)

    # Prove incompatible manual schema fails, instead of silently weakening defaults.
    sql('ALTER TABLE public.vocabulary_items ALTER COLUMN is_public SET DEFAULT true;')
    result = subprocess.run(['psql', os.environ['PGURI'], '-Xq', '-v', 'ON_ERROR_STOP=1',
                             '-f', str(MIGRATION)], capture_output=True, text=True)
    assert result.returncode != 0 and 'incompatible vocabulary publication column' in result.stderr
    sql('ALTER TABLE public.vocabulary_items ALTER COLUMN is_public SET DEFAULT false;')
    apply()
    print('PASS incompatible default rejected without guessing publication intent', flush=True)
    sql("DELETE FROM public.vocabulary_items WHERE word IN ('round-p-unknown','round-p-default');")
    print('VOCABULARY PUBLICATION DB CONTRACT: PASS (5 proof groups)', flush=True)


if __name__ == '__main__':
    main()
