"""HTTP mapping/ownership tests with a fake RPC, never Python inserts.
Real SQL proofs: supabase/replay/test_atomic_vocabulary_quota.py.
"""

from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

import main


OWNER = str(UUID(int=70001))
OTHER = str(UUID(int=70002))
ITEM = str(UUID(int=999))
ROUTES = ("/api/vocabulary/my", "/api/vocabulary/save_word")
QUOTA_ERROR = {
    "detail": {
        "error": "vocab_limit_reached",
        "limit": 30,
        "message": "Free users may save up to 30 words. Upgrade to Pro for unlimited vocabulary.",
    }
}


class Query:
    def __init__(self, store, table):
        self.store, self.table = store, table
        self.filters = []
        self.count_requested = False
        self.cap = None
        self.single_row = False
        self.payload = None

    def select(self, columns, *, count=None):
        self.count_requested = count == "exact"
        return self

    def eq(self, key, value):
        self.filters.append((key, value))
        return self

    def limit(self, size):
        self.cap = size
        return self

    def maybe_single(self):
        self.single_row = True
        return self

    single = maybe_single

    def insert(self, payload):
        raise AssertionError("Backend must not bypass atomic RPC with INSERT")

    def execute(self):
        assert not self.count_requested, "Python count is not quota authority"
        rows = [row.copy() for row in self.store.rows[self.table]
                if all(row.get(key) == value for key, value in self.filters)]
        if self.cap is not None:
            rows = rows[:self.cap]
        return SimpleNamespace(data=(rows[0] if rows else None) if self.single_row else rows)


class Store:
    def __init__(self):
        self.rows = {
            "vocabulary_items": [{"id": ITEM, "word": "syntheticword", "zh_meaning": "meaning"}],
            "user_vocabulary": [],
        }
        self.writes = []
        self.count_queries = []
        self.pro = False
        self.pro_failure = False
        self.count_failure = False
        self.rpc_failure = False
        self.override_response = False
        self.rpc_response = None
        self.rpc_calls = []
        self.auth = SimpleNamespace(get_user=self.get_user)

    def seed_saved(self, count, owner=OWNER):
        self.rows["user_vocabulary"].extend(
            {"id": str(UUID(int=100000 + int(UUID(owner)) * 100 + i)), "user_id": owner,
             "vocabulary_item_id": str(UUID(int=i + 1))}
            for i in range(count)
        )

    def get_user(self, token):
        return SimpleNamespace(user=SimpleNamespace(id=OWNER) if token == "valid-token" else None)

    def table(self, name):
        return Query(self, name)

    def rpc(self, name, params):
        assert name == "save_vocabulary_atomic", "No Python entitlement/count fallback"
        assert params["p_user_id"] == OWNER
        assert "is_pro" not in params and "p_is_pro" not in params
        self.rpc_calls.append(params)

        def execute():
            if self.rpc_failure:
                raise RuntimeError("private database failure details")
            if self.override_response:
                return SimpleNamespace(data=self.rpc_response)
            item_id, word = params["p_vocabulary_item_id"], params["p_word"]
            item = next((row for row in self.rows["vocabulary_items"]
                         if row["id"] == item_id or (word and row["word"] == word)), None)
            owned = next((row for row in self.rows["user_vocabulary"]
                          if row["user_id"] == OWNER and item and row["vocabulary_item_id"] == item["id"]), None)
            def result(status):
                return SimpleNamespace(data={
                    "status": status, "user_vocabulary_id": owned["id"],
                    "vocabulary_item_id": item["id"], "word": item["word"],
                })
            if owned:
                return result("existing")
            if self.pro_failure:
                raise RuntimeError("private entitlement failure details")
            if not self.pro:
                self.count_queries.append(OWNER)
                if self.count_failure:
                    raise RuntimeError("private count failure details")
                if sum(row["user_id"] == OWNER for row in self.rows["user_vocabulary"]) >= 30:
                    return SimpleNamespace(data={"status": "quota_reached", "limit": 30})
            if not item and not word:
                return SimpleNamespace(data={"status": "not_found"})
            if not item:
                item = {"id": str(UUID(int=80000)), "word": word, "zh_meaning": params["p_zh_meaning"]}
                self.rows["vocabulary_items"].append(item)
                self.writes.append(("vocabulary_items", item.copy()))
            owned = {"id": str(UUID(int=90000)), "user_id": OWNER,
                     "vocabulary_item_id": item["id"], "source": params["p_source"],
                     "source_practice_record_id": params["p_source_practice_record_id"]}
            self.rows["user_vocabulary"].append(owned)
            self.writes.append(("user_vocabulary", owned.copy()))
            return result("inserted")

        return SimpleNamespace(execute=execute)


@pytest.fixture
def saves(monkeypatch):
    store = Store()
    monkeypatch.setattr(main, "supabase_admin", store)
    monkeypatch.setattr(main.limiter, "enabled", False)
    # Do not enter the client context: application startup is outside this unit contract.
    client = TestClient(main.app)

    def post(route, *, authorization="Bearer valid-token", **extra):
        payload = {"vocabulary_item_id": ITEM} if route == ROUTES[0] else {"word": "syntheticword"}
        payload.update(extra)
        headers = {} if authorization is None else {"Authorization": authorization}
        return client.post(route, json=payload, headers=headers)

    yield store, post
    client.close()


@pytest.mark.parametrize("route", ROUTES)
def test_free_29_can_save_30th_item(saves, route):
    store, post = saves
    store.seed_saved(29)
    store.seed_saved(40, OTHER)
    assert post(route).status_code == 200
    assert sum(row["user_id"] == OWNER for row in store.rows["user_vocabulary"]) == 30
    assert len(store.writes) == 1


def test_canonical_rejects_31st_new_item(saves):
    store, post = saves
    store.seed_saved(30)
    response = post(ROUTES[0])
    assert response.status_code == 403
    assert response.json() == QUOTA_ERROR
    assert store.writes == []


@pytest.mark.parametrize("catalog_exists", [True, False])
def test_save_word_cannot_bypass_free_vocabulary_limit(saves, catalog_exists):
    store, post = saves
    store.seed_saved(30)
    if not catalog_exists:
        store.rows["vocabulary_items"] = []
    response = post(ROUTES[1])
    assert response.status_code == 403
    assert response.json() == QUOTA_ERROR
    # Reject before even creating a sparse shared catalog row.
    assert store.writes == []


@pytest.mark.parametrize("route", ROUTES)
def test_existing_readd_at_limit_is_idempotent(saves, route):
    store, post = saves
    store.seed_saved(30)
    store.rows["user_vocabulary"][29]["vocabulary_item_id"] = ITEM
    store.count_failure = True  # No quota query is needed for an existing link.
    assert post(route).status_code == 200
    assert len(store.rows["user_vocabulary"]) == 30
    assert store.writes == []
    assert store.count_queries == []


@pytest.mark.parametrize("route", ROUTES)
def test_pro_can_save_above_free_limit(saves, route):
    store, post = saves
    store.seed_saved(35)
    store.pro = True
    assert post(route).status_code == 200
    assert len(store.rows["user_vocabulary"]) == 36
    assert store.count_queries == []


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("authorization", [None, "Bearer ", "Bearer expired", "Basic invalid"])
def test_save_requires_verified_bearer_identity(saves, route, authorization):
    store, post = saves
    assert post(route, authorization=authorization).status_code == 401
    assert store.writes == []
    assert store.count_queries == []


@pytest.mark.parametrize("route", ROUTES)
def test_client_cannot_choose_saved_item_owner(saves, route):
    store, post = saves
    store.seed_saved(1, OTHER)
    store.rows["user_vocabulary"][0]["vocabulary_item_id"] = ITEM
    assert post(route, user_id=OTHER).status_code == 200
    assert store.writes[-1][1]["user_id"] == OWNER
    assert len(store.rows["user_vocabulary"]) == 2


@pytest.mark.parametrize("route", ROUTES)
def test_other_users_existing_link_cannot_bypass_owners_limit(saves, route):
    store, post = saves
    store.seed_saved(30)
    store.seed_saved(1, OTHER)
    store.rows["user_vocabulary"][-1]["vocabulary_item_id"] = ITEM
    response = post(route, user_id=OTHER)
    assert response.status_code == 403
    assert response.json() == QUOTA_ERROR
    assert store.writes == []


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("failure", ["count_failure", "rpc_failure", "pro_failure"])
def test_atomic_rpc_failures_never_fall_back_to_python_insert(saves, route, failure):
    store, post = saves
    setattr(store, failure, True)
    response = post(route)
    assert response.status_code == 503
    assert "private" not in response.text
    assert store.writes == []


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("payload", [None, {}, {"status": "unexpected"},
                                    {"status": "inserted"}, {"status": "quota_reached", "limit": 99}])
def test_missing_or_invalid_rpc_contract_fails_closed(saves, route, payload):
    store, post = saves
    store.override_response = True
    store.rpc_response = payload
    assert post(route).status_code == 503
    assert store.writes == []


@pytest.mark.parametrize("route", ROUTES)
def test_client_cannot_supply_pro_entitlement(saves, route):
    store, post = saves
    store.seed_saved(30)
    response = post(route, is_pro=True, p_is_pro=True, user_id=OTHER)
    assert response.status_code == 403
    assert response.json() == QUOTA_ERROR
    assert store.rpc_calls[0]["p_user_id"] == OWNER
    assert store.writes == []


def test_missing_catalog_item_retains_404(saves):
    store, post = saves
    store.rows["vocabulary_items"] = []
    assert post(ROUTES[0]).status_code == 404
    assert store.writes == []


def test_new_word_under_limit_creates_catalog_then_owned_link(saves):
    store, post = saves
    store.seed_saved(29)
    store.rows["vocabulary_items"] = []
    response = post(ROUTES[1], word=" Synthetic-Word! ", zh_meaning="meaning")
    assert response.status_code == 200
    assert response.json()["status"] == "added"
    assert [table for table, _ in store.writes] == ["vocabulary_items", "user_vocabulary"]
    assert store.writes[0][1]["word"] == "synthetic-word"
    assert store.writes[1][1]["vocabulary_item_id"] == store.writes[0][1]["id"]
