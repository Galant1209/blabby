#!/usr/bin/env python3
"""Real PG17 transactions, ACLs and lock-removal proof. Disposable DB ONLY.

Uses psql + stdlib so the migration-replay CI job needs no Python DB driver.
Called by replay.sh; never point PGURI at a shared or production database.
"""
import concurrent.futures
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time
import uuid


MIGRATION = Path(__file__).parents[1] / "migrations/20260905040134_atomic_vocabulary_save_quota.sql"
SIGNATURE = "public.save_vocabulary_atomic(uuid,uuid,text,text,text,uuid)"
LOCK = "PERFORM pg_advisory_xact_lock(hashtextextended('blabby:vocabulary-save:' || p_user_id::text, 0));"
BARRIER = "2147483000, 17"


def literal(value):
    return "NULL" if value is None else "'" + str(value).replace("'", "''") + "'"


def sql(statement, *, check=True):
    result = subprocess.run(
        ["psql", os.environ["PGURI"], "-XqAt", "-v", "ON_ERROR_STOP=1", "-c", statement],
        text=True, capture_output=True, timeout=20,
    )
    if check and result.returncode:
        raise AssertionError(result.stderr)
    return result.stdout.strip() if check else result


def rpc(owner, *, item=None, word=None):
    return (
        "public.save_vocabulary_atomic("
        f"p_user_id => {literal(owner)}::uuid, "
        f"p_vocabulary_item_id => {literal(item)}::uuid, p_word => {literal(word)})"
    )


def save(owner, *, item=None, word=None, role="service_role"):
    return json.loads(sql(f"SET ROLE {role}; SELECT {rpc(owner, item=item, word=word)};"))


class Fixture:
    def __init__(self, count, entitlement="free", profile=True):
        self.owner = str(uuid.uuid4())
        self.items = [str(uuid.uuid4()) for _ in range(max(count + 3, 40))]
        self.extra_words = []
        sql(f"INSERT INTO auth.users(id) VALUES ({literal(self.owner)});")
        if profile:
            grant = entitlement in ("grant", "expired_grant")
            expiry = "now() - interval '1 day'" if entitlement == "expired_grant" else "NULL"
            sql(f"""INSERT INTO public.profiles(id, is_pro, is_pro_grant, pro_grant_expires_at)
                    VALUES ({literal(self.owner)}, {str(entitlement == 'bare_flag').lower()},
                            {str(grant).lower()}, {expiry});""")
        if entitlement in ("subscription", "expired_subscription"):
            expiry = "now() + interval '1 day'" if entitlement == "subscription" else "now() - interval '1 day'"
            sql(f"""INSERT INTO public.subscriptions(user_id, order_id, status, expires_at)
                    VALUES ({literal(self.owner)}, {literal('round-k-' + self.owner)}, 'active', {expiry});""")
        values = ",".join(f"({literal(item)}, {literal('rk-' + item)}, 'synthetic')" for item in self.items)
        sql(f"INSERT INTO public.vocabulary_items(id, word, zh_meaning) VALUES {values};")
        if count:
            values = ",".join(f"({literal(self.owner)}, {literal(item)})" for item in self.items[:count])
            sql(f"INSERT INTO public.user_vocabulary(user_id, vocabulary_item_id) VALUES {values};")

    def count(self):
        return int(sql(f"SELECT count(DISTINCT vocabulary_item_id) FROM public.user_vocabulary WHERE user_id={literal(self.owner)};"))

    def cleanup(self):
        sql(f"DELETE FROM auth.users WHERE id={literal(self.owner)};")
        sql("DELETE FROM public.vocabulary_items WHERE id IN (" + ",".join(map(literal, self.items)) + ");")
        for word in self.extra_words:
            sql(f"DELETE FROM public.vocabulary_items WHERE word={literal(word)};")


def pair(fixture, calls, *, insert_barrier=True):
    """Hold both connections in actual advisory waits before releasing them.

    For insertion cases a test-only BEFORE INSERT trigger takes a shared gate
    lock. With the real owner lock, one caller waits there, one on owner lock.
    With the owner lock removed, both have counted 29 and wait at the gate.
    This makes the mutation proof deterministic without sleeps in production SQL.
    """
    tag = "round-k-" + uuid.uuid4().hex
    lock = (f"pg_advisory_xact_lock({BARRIER})" if insert_barrier else
            f"pg_advisory_xact_lock(hashtextextended('blabby:vocabulary-save:' || {literal(fixture.owner)}, 0))")
    if insert_barrier:
        sql(f"""
            CREATE FUNCTION public.round_k_insert_gate() RETURNS trigger LANGUAGE plpgsql AS $$
            BEGIN PERFORM pg_advisory_xact_lock_shared({BARRIER}); RETURN NEW; END $$;
            CREATE TRIGGER round_k_insert_gate BEFORE INSERT ON public.user_vocabulary
            FOR EACH ROW EXECUTE FUNCTION public.round_k_insert_gate();
        """)
    blocker = subprocess.Popen(
        ["psql", os.environ["PGURI"], "-XqAt", "-v", "ON_ERROR_STOP=1"],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, bufsize=1,
    )
    try:
        blocker.stdin.write(f"BEGIN; DO $$ BEGIN PERFORM {lock}; END $$;\n\\echo READY\n")
        blocker.stdin.flush()
        assert select.select([blocker.stdout], [], [], 8)[0], "barrier connection did not start"
        assert blocker.stdout.readline().strip() == "READY"

        def worker(index):
            return json.loads(sql(
                f"SET application_name={literal(tag + str(index))}; SET statement_timeout='10s'; "
                f"BEGIN; SET LOCAL ROLE service_role; SELECT {calls[index]}; COMMIT;"
            ))

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, index) for index in range(2)]
            try:
                deadline = time.monotonic() + 8
                while time.monotonic() < deadline:
                    waiting = int(sql(f"SELECT count(*) FROM pg_stat_activity WHERE application_name LIKE {literal(tag + '%')} AND wait_event='advisory';"))
                    if waiting == 2:
                        break
                    if any(future.done() for future in futures):
                        raise AssertionError("RPC escaped synchronization or failed: " + str([f.result() for f in futures if f.done()]))
                    time.sleep(0.02)
                else:
                    raise AssertionError("two real connections never reached advisory waits")
            finally:
                blocker.stdin.write("COMMIT;\n\\q\n")
                blocker.stdin.flush()
            return [future.result(timeout=15) for future in futures]
    finally:
        if blocker.poll() is None:
            blocker.terminate()
        blocker.wait(timeout=5)
        if insert_barrier:
            sql("DROP TRIGGER round_k_insert_gate ON public.user_vocabulary; DROP FUNCTION public.round_k_insert_gate();")


def free_pair():
    fixture = Fixture(29)
    try:
        outcomes = pair(fixture, [rpc(fixture.owner, item=item) for item in fixture.items[29:31]])
        statuses = sorted(row["status"] for row in outcomes)
        count = fixture.count()
        assert statuses == ["inserted", "quota_reached"] and count == 30, (statuses, count)
        print("PASS free concurrent distinct items: inserted/quota_reached, count=30", flush=True)
    finally:
        fixture.cleanup()


def main():
    if sys.argv[1:] != ["--disposable"] or not os.environ.get("PGURI"):
        raise SystemExit("Requires --disposable and PGURI for an isolated replay database")
    assert int(sql("SHOW server_version_num;")) // 10000 == 17, "PostgreSQL 17 required"
    # Catalog, effective privileges and real permission errors, not source grep.
    assert sql(f"SELECT prosecdef AND provolatile='v' AND proconfig @> ARRAY['search_path=pg_catalog, pg_temp'] FROM pg_proc WHERE oid='{SIGNATURE}'::regprocedure;") == "t"
    for role in ("anon", "authenticated"):
        assert sql(f"SELECT has_function_privilege('{role}', '{SIGNATURE}', 'EXECUTE');") == "f"
        result = sql(f"SET ROLE {role}; SELECT {rpc(str(uuid.uuid4()), item=str(uuid.uuid4()))};", check=False)
        assert result.returncode and "permission denied for function" in result.stderr
    assert sql(f"SELECT has_function_privilege('service_role', '{SIGNATURE}', 'EXECUTE');") == "t"
    for role in ("anon", "authenticated", "service_role"):
        assert sql(f"SELECT has_any_column_privilege('{role}', 'public.user_vocabulary', 'INSERT');") == "f"
    # Even BYPASSRLS service_role cannot skip the RPC with direct INSERT.
    result = sql("SET ROLE service_role; INSERT INTO public.user_vocabulary(user_id,vocabulary_item_id) VALUES(gen_random_uuid(),gen_random_uuid());", check=False)
    assert result.returncode and "permission denied" in result.stderr
    print("PASS RPC ACL/definer/search_path and direct INSERT denial", flush=True)

    free_pair()
    fixture = Fixture(29)
    try:
        word = "round-k-" + uuid.uuid4().hex.translate(str.maketrans("0123456789", "abcdefghij"))
        fixture.extra_words.append(word)
        outcomes = pair(fixture, [rpc(fixture.owner, item=fixture.items[29]), rpc(fixture.owner, word=word)])
        assert sorted(row["status"] for row in outcomes) == ["inserted", "quota_reached"]
        assert fixture.count() == 30
        assert int(sql(f"SELECT count(*) FROM public.vocabulary_items WHERE word={literal(word)};")) == int(outcomes[1]["status"] == "inserted")
        print("PASS mixed ID/Reading-word concurrency: count=30, no losing-request corpus row", flush=True)
    finally:
        fixture.cleanup()
    fixture = Fixture(30)
    try:
        outcomes = pair(fixture, [rpc(fixture.owner, item=fixture.items[i]) for i in (0, 30)], insert_barrier=False)
        assert [row["status"] for row in outcomes] == ["existing", "quota_reached"]
        assert fixture.count() == 30
        absent_word = "round-k-" + uuid.uuid4().hex.translate(str.maketrans("0123456789", "abcdefghij"))
        fixture.extra_words.append(absent_word)
        assert save(fixture.owner, word=absent_word)["status"] == "quota_reached"
        assert sql(f"SELECT count(*) FROM public.vocabulary_items WHERE word={literal(absent_word)};") == "0"
        print("PASS count=30 existing/new concurrency; no orphan corpus row", flush=True)
    finally:
        fixture.cleanup()

    fixture = Fixture(29)
    try:
        call = rpc(fixture.owner, item=fixture.items[29])
        outcomes = pair(fixture, [call, call])
        assert sorted(row["status"] for row in outcomes) == ["existing", "inserted"]
        assert outcomes[0]["user_vocabulary_id"] == outcomes[1]["user_vocabulary_id"]
        assert fixture.count() == 30
        print("PASS concurrent same item: one saved row, no unique-conflict error", flush=True)
    finally:
        fixture.cleanup()

    for entitlement in ("grant", "subscription"):
        fixture = Fixture(35, entitlement)
        try:
            outcomes = pair(fixture, [rpc(fixture.owner, item=item) for item in fixture.items[35:37]])
            assert [row["status"] for row in outcomes] == ["inserted", "inserted"]
            assert fixture.count() == 37
            print(f"PASS Pro {entitlement}: two concurrent inserts, count=37", flush=True)
        finally:
            fixture.cleanup()
    for entitlement in ("bare_flag", "expired_grant", "expired_subscription"):
        fixture = Fixture(30, entitlement)
        try:
            assert save(fixture.owner, item=fixture.items[30])["status"] == "quota_reached"
        finally:
            fixture.cleanup()
    print("PASS canonical entitlement: bare/expired flags do not bypass", flush=True)

    fixture = Fixture(29, profile=False)
    try:
        word = "round-k-" + uuid.uuid4().hex.translate(str.maketrans("0123456789", "abcdefghij"))
        fixture.extra_words.append(word)
        call = rpc(fixture.owner, word=word)
        outcomes = pair(fixture, [call, call])
        assert sorted(row["status"] for row in outcomes) == ["existing", "inserted"]
        assert fixture.count() == 30
        assert sql(f"SELECT count(*) FROM public.vocabulary_items WHERE word={literal(word)};") == "1"
        saved_id = outcomes[0]["user_vocabulary_id"]
        sql(f"SET ROLE service_role; UPDATE public.user_vocabulary SET review_count=1 WHERE id={literal(saved_id)};")
        assert sql(f"SELECT review_count FROM public.user_vocabulary WHERE id={literal(saved_id)};") == "1"
        # A different isolation level must fail before a write, never use stale counts.
        result = sql(f"BEGIN ISOLATION LEVEL REPEATABLE READ; SET LOCAL ROLE service_role; SELECT {call};", check=False)
        assert result.returncode and "requires read committed" in result.stderr
        print("PASS Reading same-word concurrency without profile; review UPDATE preserved; isolation fails closed", flush=True)
    finally:
        fixture.cleanup()

    # A corpus insert must roll back too if the owned insert fails.
    fixture = Fixture(29)
    try:
        word = "round-k-rollback"
        fixture.extra_words.append(word)
        sql("""CREATE FUNCTION public.round_k_fail_insert() RETURNS trigger LANGUAGE plpgsql AS $$
               BEGIN RAISE EXCEPTION 'round-k injected failure'; END $$;
               CREATE TRIGGER round_k_fail_insert BEFORE INSERT ON public.user_vocabulary
               FOR EACH ROW EXECUTE FUNCTION public.round_k_fail_insert();""")
        try:
            result = sql(f"SET ROLE service_role; SELECT {rpc(fixture.owner, word=word)};", check=False)
            assert result.returncode
            assert sql(f"SELECT count(*) FROM public.vocabulary_items WHERE word={literal(word)};") == "0"
            assert fixture.count() == 29
        finally:
            sql("DROP TRIGGER round_k_fail_insert ON public.user_vocabulary; DROP FUNCTION public.round_k_fail_insert();")
        print("PASS owned-insert failure rolls back Reading corpus insert", flush=True)
    finally:
        fixture.cleanup()

    definition = sql(f"SELECT pg_get_functiondef('{SIGNATURE}'::regprocedure);")
    assert definition.count(LOCK) == 1
    try:
        sql(definition.replace(LOCK, "NULL; -- deliberate local mutation: owner lock removed"))
        try:
            free_pair()
        except AssertionError as error:
            assert error.args == ((["inserted", "inserted"], 31),), error
            print("MUTATION PROBE: expected failure reproduced — lock removed, two inserted, count=31", flush=True)
        else:
            raise AssertionError("lock removal did not turn the concurrency assertion red")
    finally:
        sql(definition)
    free_pair()

    # Reapplication restores both table and column grants, not merely function text.
    sql("GRANT INSERT ON public.user_vocabulary TO anon, authenticated, service_role; GRANT INSERT(user_id,vocabulary_item_id) ON public.user_vocabulary TO PUBLIC;")
    result = subprocess.run(["psql", os.environ["PGURI"], "-Xq", "-v", "ON_ERROR_STOP=1", "-f", str(MIGRATION)], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    for role in ("anon", "authenticated", "service_role"):
        assert sql(f"SELECT has_any_column_privilege('{role}', 'public.user_vocabulary', 'INSERT');") == "f"
    assert sql(f"SELECT pg_get_functiondef('{SIGNATURE}'::regprocedure);") == definition
    print("PASS migration rerun restores exact function and INSERT restrictions", flush=True)
    print("ATOMIC VOCABULARY DB CONTRACT: PASS", flush=True)


if __name__ == "__main__":
    main()
