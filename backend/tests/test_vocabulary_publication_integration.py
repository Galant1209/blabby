"""Real FastAPI + disposable PG17 behavior; provider/auth are deterministic doubles.

The adapter translates the small PostgREST fluent surface used here into real SQL.
It never emulates publication filtering, persistence, quota RPC, or ownership in
Python. This verifies app + database behavior, not the PostgREST network service.
Run after replay with BLABBY_PUBLICATION_TEST_DB=disposable and local PGURI.
"""
import json
import os
import re
import subprocess
from types import SimpleNamespace
from urllib.parse import urlparse, parse_qs
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient
import main


def ident(value):
    assert re.fullmatch(r'[a-z_]+', value), value
    return '"' + value + '"'


def literal(value):
    if value is None:
        return 'NULL'
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return 'ARRAY[' + ','.join(literal(v) for v in value) + ']::text[]'
    return "'" + str(value).replace("'", "''") + "'"


def sql(command, *, service=True):
    prefix = 'SET ROLE service_role; ' if service else ''
    result = subprocess.run(['psql', os.environ['PGURI'], '-XqAt', '-v', 'ON_ERROR_STOP=1',
                             '-c', prefix + command], capture_output=True, text=True, timeout=20)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


class Database:
    def table(self, name):
        assert name in {'vocabulary_items', 'user_vocabulary', 'vocabulary_review_logs', 'practice_records'}
        return Query(name)

    def rpc(self, name, params):
        assert name == 'save_vocabulary_atomic'
        args = ','.join(ident(k) + ' => ' + literal(v) for k, v in params.items())
        return SimpleNamespace(execute=lambda: SimpleNamespace(data=json.loads(
            sql(f'SELECT public.save_vocabulary_atomic({args});'))))


class Query:
    def __init__(self, name):
        self.name, self.columns, self.filters, self.orders = name, '*', [], []
        self.cap, self.offset, self.one = None, 0, False
        self.payload, self.operation = None, 'select'

    def select(self, columns):
        self.columns = columns
        return self

    def eq(self, key, value):
        self.filters.append(f't.{ident(key)} = {literal(value)}')
        return self

    def lte(self, key, value):
        self.filters.append(f't.{ident(key)} <= {literal(value)}')
        return self

    def or_(self, expression):
        first, second = expression.split(',zh_meaning.ilike.')
        pattern = json.loads(first.removeprefix('word.ilike.'))
        assert json.loads(second) == pattern
        self.filters.append(f'(t.word ILIKE {literal(pattern)} OR t.zh_meaning ILIKE {literal(pattern)})')
        return self

    def order(self, column, desc=False):
        self.orders.append('t.' + ident(column) + (' DESC' if desc else ' ASC'))
        return self

    def limit(self, cap):
        self.cap = cap
        return self

    def range(self, start, end):
        self.offset, self.cap = start, end - start + 1
        return self

    def maybe_single(self):
        self.one = True
        return self

    single = maybe_single

    def insert(self, payload):
        self.operation, self.payload = 'insert', payload
        return self

    def update(self, payload):
        self.operation, self.payload = 'update', payload
        return self

    def execute(self):
        table = 'public.' + ident(self.name)
        where = ' WHERE ' + ' AND '.join(self.filters) if self.filters else ''
        if self.operation == 'insert':
            rows = self.payload if isinstance(self.payload, list) else [self.payload]
            columns = list(rows[0])
            assert all(list(row) == columns for row in rows)
            values = ','.join('(' + ','.join(literal(row[c]) for c in columns) + ')' for row in rows)
            inner = f'INSERT INTO {table} ({",".join(map(ident, columns))}) VALUES {values} RETURNING *'
            command = f'WITH changed AS ({inner}) SELECT coalesce(json_agg(changed),\'[]\'::json) FROM changed;'
        elif self.operation == 'update':
            assignments = ','.join(ident(k) + '=' + literal(v) for k, v in self.payload.items())
            inner = f'UPDATE {table} t SET {assignments}{where} RETURNING t.*'
            command = f'WITH changed AS ({inner}) SELECT coalesce(json_agg(changed),\'[]\'::json) FROM changed;'
        else:
            join = ''
            if 'vocabulary_items(' in self.columns:
                own, nested = self.columns.split('vocabulary_items(')
                nested = nested.rstrip(')')
                own_cols = own.strip().rstrip(',').strip()
                projection = 't.*' if own_cols == '*' else ','.join('t.' + ident(c.strip()) for c in own_cols.split(','))
                projection += ',json_build_object(' + ','.join(
                    literal(c.strip()) + ',v.' + ident(c.strip()) for c in nested.split(',')) + ') AS vocabulary_items'
                join = ' JOIN public.vocabulary_items v ON v.id=t.vocabulary_item_id'
            else:
                projection = 't.*' if self.columns == '*' else ','.join('t.' + ident(c.strip()) for c in self.columns.split(','))
            order = ' ORDER BY ' + ','.join(self.orders) if self.orders else ''
            limit = f' LIMIT {int(self.cap)} OFFSET {int(self.offset)}' if self.cap is not None else ''
            command = f"SELECT coalesce(json_agg(r),'[]'::json) FROM (SELECT {projection} FROM {table} t{join}{where}{order}{limit}) r;"
        rows = json.loads(sql(command))
        return SimpleNamespace(data=(rows[0] if rows else None) if self.one else rows)


@pytest.fixture
def integration(monkeypatch):
    if os.getenv('BLABBY_PUBLICATION_TEST_DB') != 'disposable':
        pytest.skip('Explicit disposable PostgreSQL acknowledgement required')
    parsed = urlparse(os.environ['PGURI'])
    host = parse_qs(parsed.query).get('host', [parsed.hostname or ''])[0]
    assert host in {'localhost', '127.0.0.1', '::1'} or host.startswith('/tmp/blabby-round-p-pg.')
    assert sql('SHOW server_version_num;', service=False).startswith('17')
    a, b = str(uuid4()), str(uuid4())
    topic = 'roundp' + uuid4().hex
    sql(f'INSERT INTO auth.users(id) VALUES ({literal(a)}),({literal(b)});', service=False)
    sql(f"INSERT INTO public.practice_records(user_id,weakness_tag,topic,question) VALUES ({literal(a)},'weak_vocab',{literal(topic)},'synthetic question'),({literal(b)},'weak_vocab',{literal(topic)},'synthetic question');", service=False)
    monkeypatch.setattr(main, 'supabase_admin', Database())
    monkeypatch.setattr(main, 'verify_token', lambda token: {'Bearer A': a, 'Bearer B': b}[token])
    monkeypatch.setattr(main.limiter, 'enabled', False)
    client = TestClient(main.app)
    yield client, a, b, topic
    client.close()
    sql(f'DELETE FROM public.practice_records WHERE user_id IN ({literal(a)},{literal(b)});', service=False)
    sql(f'DELETE FROM auth.users WHERE id IN ({literal(a)},{literal(b)});', service=False)
    sql(f'DELETE FROM public.vocabulary_items WHERE topic={literal(topic)};', service=False)


def generate(monkeypatch, word, topic):
    payload = [dict(word=word, zh_meaning='合成測試', common_chunk='synthetic test phrase',
                    speaking_sentence='A generic synthetic sentence.', topic=topic, is_public=True)]
    monkeypatch.setattr(main, 'groq_client', SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(
        create=lambda **kwargs: SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=json.dumps(payload)))])
    ))))


def test_generate_save_publication_and_owner_srs(integration, monkeypatch):
    client, a, b, topic = integration
    generate(monkeypatch, 'generated' + topic, topic)
    response = client.post('/api/vocabulary/generate', headers={'Authorization': 'Bearer A'}, json={'is_public': True})
    assert response.status_code == 200, response.text
    item = response.json()['items'][0]
    assert response.json()['generated_count'] == 1
    assert sql(f"SELECT is_public FROM public.vocabulary_items WHERE id={literal(item['id'])};") == 'f'
    # Baseline broad query does include the row: the old API would publish it.
    assert sql(f"SELECT count(*) FROM public.vocabulary_items WHERE topic={literal(topic)};") == '1'
    saved = client.post('/api/vocabulary/my', headers={'Authorization': 'Bearer A'}, json={'vocabulary_item_id': item['id'], 'is_public': True})
    assert saved.status_code == 200, saved.text
    owned_id = saved.json()['id']
    for headers in ({}, {'Authorization': 'Bearer B'}):
        for params in ({'topic': topic}, {'q': item['word']}, {'search': '合成測試', 'topic': topic},
                       {'topic': topic, 'include_unpublished': 'true'}):
            data = client.get('/api/vocabulary/items', headers=headers, params=params).json()
            assert data['items'] == [] and data['next_offset'] is None and not data['has_more']
    personal = client.get('/api/vocabulary/my', headers={'Authorization': 'Bearer A'}).json()['items']
    assert personal[0]['vocabulary_items']['id'] == item['id']
    assert client.get('/api/vocabulary/my', headers={'Authorization': 'Bearer B'}, params={'user_id': a}).json()['items'] == []
    due = client.get('/api/vocabulary/review/today', headers={'Authorization': 'Bearer A'}).json()['items']
    assert due[0]['id'] == owned_id
    assert client.post('/api/vocabulary/review', headers={'Authorization': 'Bearer B'}, json={
        'user_vocabulary_id': owned_id, 'user_id': a, 'result': 'good'}).status_code == 404
    reviewed = client.post('/api/vocabulary/review', headers={'Authorization': 'Bearer A'}, json={'user_vocabulary_id': owned_id, 'result': 'good'})
    assert reviewed.status_code == 200 and reviewed.json()['review_count'] == 1
    # Another user's generation DB cache must not redistribute A's unpublished row.
    generate(monkeypatch, 'second' + topic, topic)
    second = client.post('/api/vocabulary/generate', headers={'Authorization': 'Bearer B'}).json()
    assert item['id'] not in [r['id'] for r in second['items']]
    # Only an explicit trusted fixture publication makes it publicly discoverable.
    sql(f"UPDATE public.vocabulary_items SET is_public=true WHERE id={literal(item['id'])};")
    visible = client.get('/api/vocabulary/items', params={'topic': topic, 'limit': 1}).json()
    assert [r['id'] for r in visible['items']] == [item['id']]
    assert not visible['has_more'] and visible['next_offset'] is None


def test_save_word_default_and_http_quota_on_unpublished_rows(integration):
    client, a, b, topic = integration
    # Unique ASCII normalized word; save_word does not accept a topic field.
    word = 'roundp' + ''.join(chr(97 + int(c, 16)) for c in uuid4().hex)
    try:
        response = client.post('/api/vocabulary/save_word', headers={'Authorization': 'Bearer A'},
                               json={'word': word, 'zh_meaning': '使用者提供文字', 'is_public': True})
        assert response.status_code == 200, response.text
        item_id = sql(f'SELECT id FROM public.vocabulary_items WHERE word={literal(word)};')
        assert sql(f'SELECT is_public FROM public.vocabulary_items WHERE id={literal(item_id)};') == 'f'
        assert client.get('/api/vocabulary/items', params={'q': word}).json()['items'] == []
        # Actual RPC fills the owner's collection to 30, all unpublished.
        for n in range(29):
            fixture_id = str(uuid4())
            sql(f"INSERT INTO public.vocabulary_items(id,word,zh_meaning,topic) VALUES ({literal(fixture_id)},'quota{n}','synthetic',{literal(topic)});")
            result = client.post('/api/vocabulary/my', headers={'Authorization': 'Bearer A'}, json={'vocabulary_item_id': fixture_id})
            assert result.status_code == 200, result.text
        existing = client.post('/api/vocabulary/my', headers={'Authorization': 'Bearer A'}, json={'vocabulary_item_id': item_id})
        assert existing.status_code == 200
        extra = str(uuid4())
        sql(f"INSERT INTO public.vocabulary_items(id,word,zh_meaning,topic) VALUES ({literal(extra)},'extra','synthetic',{literal(topic)});")
        assert client.post('/api/vocabulary/my', headers={'Authorization': 'Bearer A'}, json={'vocabulary_item_id': extra}).status_code == 403
        # Replay auth shim does not auto-create a profile on auth.users insertion.
        sql(f'INSERT INTO public.profiles(id,is_pro_grant) VALUES ({literal(a)},true) ON CONFLICT(id) DO UPDATE SET is_pro_grant=true;', service=False)
        assert client.post('/api/vocabulary/my', headers={'Authorization': 'Bearer A'}, json={'vocabulary_item_id': extra}).status_code == 200
        assert sql(f'SELECT count(*) FROM public.user_vocabulary WHERE user_id={literal(a)};') == '31'
    finally:
        sql(f'DELETE FROM public.vocabulary_items WHERE word={literal(word)};', service=False)
