"""Public HTTP contract, with semantic query double; no live DB or auth calls."""
import json
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient
import main

FIELDS = {'id', 'word', 'zh_meaning', 'topic', 'ielts_band_level', 'part_of_speech',
          'common_chunk', 'speaking_sentence', 'better_than', 'usage_note_zh'}


class Catalog:
    def __init__(self):
        self.rows = [dict(id=f'{i:03}', word=f'word{i:03}', zh_meaning='學習' if i == 110 else '意思',
                          topic='study' if i % 2 else 'work', ielts_band_level='6.0' if i % 2 else '7.0',
                          tags=['private'], created_at='private', common_mistake='private',
                          difficulty_level='private', simple_definition_en='private') for i in range(125)]
        self.calls = []
        self.failure = False

    def table(self, name):
        assert name == 'vocabulary_items'
        store = self

        class Query:
            def __init__(self):
                self.rows = store.rows.copy()
                self.orders = []
            def select(self, columns):
                assert set(columns.split(', ')) == FIELDS
                return self
            def eq(self, column, value):
                self.rows = [r for r in self.rows if r[column] == value]
                store.calls.append(('eq', column, value))
                return self
            def or_(self, expression):
                first, second = expression.split(',zh_meaning.ilike.')
                assert first.startswith('word.ilike.')
                pattern = json.loads(first.removeprefix('word.ilike.'))
                assert json.loads(second) == pattern
                term = pattern[1:-1].replace(r'\_', '_').lower()
                self.rows = [r for r in self.rows if term in r['word'].lower() or term in r['zh_meaning'].lower()]
                store.calls.append(('search', expression))
                return self
            def order(self, column, desc=False):
                assert not desc
                self.orders.append(column)
                return self
            def range(self, start, end):
                assert self.orders == ['word', 'id']
                assert end - start <= 100
                self.rows = sorted(self.rows, key=lambda r: (r['word'], r['id']))[start:end+1]
                store.calls.append(('range', start, end))
                return self
            def execute(self):
                if store.failure:
                    raise RuntimeError('secret-db-url-token')
                # Deliberately return extra columns: HTTP allowlist must also hold.
                return SimpleNamespace(data=self.rows)
        return Query()


@pytest.fixture
def catalog(monkeypatch):
    store = Catalog()
    monkeypatch.setattr(main, 'supabase_admin', store)
    monkeypatch.setattr(main.limiter, 'enabled', False)
    client = TestClient(main.app)
    yield store, client
    client.close()


def get(client, **params):
    return client.get('/api/vocabulary/items', params=params)


def test_anonymous_list_requires_no_auth(catalog):
    store, client = catalog
    response = get(client)
    assert response.status_code == 200
    assert len(response.json()['items']) == 50
    assert store.calls == [('range', 0, 50)]


def test_exact_public_allowlist_even_if_provider_returns_more(catalog):
    _, client = catalog
    assert all(set(row) == FIELDS for row in get(client).json()['items'])
    assert 'private' not in get(client).text


@pytest.mark.parametrize('params,ids', [
    ({'q': 'WORD110'}, ['110']), ({'q': '學習'}, ['110']),
    ({'search': 'word124'}, ['124']), ({'q': 'missing'}, []),
    ({'q': 'word110', 'topic': 'study'}, []),
])
def test_server_search_entire_corpus(catalog, params, ids):
    _, client = catalog
    assert [r['id'] for r in get(client, **params).json()['items']] == ids


@pytest.mark.parametrize('params,column,value', [
    ({'topic': 'study'}, 'topic', 'study'),
    ({'band': '7.0'}, 'ielts_band_level', '7.0'),
    ({'level': '6.0'}, 'ielts_band_level', '6.0'),
])
def test_server_filters(catalog, params, column, value):
    store, client = catalog
    rows = get(client, **params).json()['items']
    assert rows and all(row[column] == value for row in rows)
    assert ('eq', column, value) in store.calls


def test_limit_max_and_stable_pagination(catalog):
    _, client = catalog
    first = get(client, limit=100).json()
    assert len(first['items']) == 100 and first['next_offset'] == 100
    second = get(client, limit=100, offset=first['next_offset']).json()
    assert len(second['items']) == 25 and second['next_offset'] is None and not second['has_more']
    assert not ({r['id'] for r in first['items']} & {r['id'] for r in second['items']})


@pytest.mark.parametrize('params', [
    {'limit': 101}, {'limit': 0}, {'limit': -1}, {'limit': 'bad'},
    {'offset': -1}, {'offset': 100001}, {'q': 'x'*101},
    {'q': 'foo,word.neq.secret'}, {'q': '*'}, {'topic': 'x'*81},
])
def test_invalid_inputs_rejected_before_db(catalog, params):
    store, client = catalog
    assert get(client, **params).status_code == 422
    assert store.calls == []


def test_empty_page(catalog):
    _, client = catalog
    assert get(client, offset=1000).json() == dict(items=[], limit=50, offset=1000, has_more=False, next_offset=None)


def test_query_failure_is_explicit_safe_error(catalog):
    store, client = catalog
    store.failure = True
    response = get(client)
    assert response.status_code == 503
    assert response.json() == {'detail': 'Failed to load vocabulary items'}
    assert 'secret' not in response.text


def test_database_unavailable(catalog, monkeypatch):
    _, client = catalog
    monkeypatch.setattr(main, 'supabase_admin', None)
    assert get(client).status_code == 503


def test_all_query_variants_share_per_ip_rate_limit(catalog, monkeypatch):
    _, client = catalog
    monkeypatch.setattr(main.limiter, 'enabled', True)
    main.limiter.reset()
    try:
        for i in range(30):
            assert get(client, q=f'word{i}', topic='work', band='7.0', offset=i).status_code == 200
        assert get(client).status_code == 429
        assert get(client, q='different').status_code == 429
    finally:
        main.limiter.reset()


def test_frontend_public_bank_behavior():
    harness = Path(__file__).with_name('frontend_public_vocabulary_behavior.mjs')
    result = subprocess.run(['node', str(harness)], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_no_frontend_raw_corpus_read_dependency():
    app = Path(__file__).parents[2] / 'frontend'
    direct = re.compile(r"(?:\.from|\.table)\s*\(\s*['\"]vocabulary_items['\"]|/rest/v1/vocabulary_items")
    for path in app.rglob('*'):
        if path.suffix in {'.html', '.js', '.mjs'}:
            assert not direct.search(path.read_text()), str(path)
