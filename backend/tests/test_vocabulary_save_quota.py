"""Hermetic HTTP contracts for both vocabulary saves; no live auth/DB calls."""

from types import SimpleNamespace
from uuid import UUID

import pytest
from fastapi.testclient import TestClient

import main


OWNER = "verified-user"
OTHER = "another-user"
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
        self.payload = payload
        return self

    def execute(self):
        if self.payload is not None:
            row = {"id": str(UUID(int=10000 + len(self.store.writes))), **self.payload}
            if self.table == "user_vocabulary":
                assert not any(
                    (old["user_id"], old["vocabulary_item_id"])
                    == (row["user_id"], row["vocabulary_item_id"])
                    for old in self.store.rows[self.table]
                ), "fake enforces the owned-item unique constraint"
            self.store.rows[self.table].append(row)
            self.store.writes.append((self.table, row.copy()))
            return SimpleNamespace(data=[row])
        rows = [
            row.copy() for row in self.store.rows[self.table]
            if all(row.get(key) == value for key, value in self.filters)
        ]
        count = None
        if self.count_requested:
            self.store.count_queries.append((self.table, list(self.filters)))
            if self.store.count_failure:
                raise RuntimeError("count unavailable")
            count = None if self.store.missing_count else len(rows)
        if self.cap is not None:
            rows = rows[:self.cap]
        data = (rows[0] if rows else None) if self.single_row else rows
        return SimpleNamespace(data=data, count=count)


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
        self.missing_count = False
        self.auth = SimpleNamespace(get_user=self.get_user)

    def seed_saved(self, count, owner=OWNER):
        self.rows["user_vocabulary"].extend(
            {"id": f"saved-{owner}-{i}", "user_id": owner,
             "vocabulary_item_id": str(UUID(int=i + 1))}
            for i in range(count)
        )

    def get_user(self, token):
        return SimpleNamespace(user=SimpleNamespace(id=OWNER) if token == "valid-token" else None)

    def table(self, name):
        return Query(self, name)

    def rpc(self, name, params):
        assert name == "is_user_pro"
        assert params == {"user_id": OWNER}

        def execute():
            if self.pro_failure:
                raise RuntimeError("pro lookup unavailable")
            return SimpleNamespace(data=self.pro)

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
@pytest.mark.parametrize("failure", ["count_failure", "missing_count"])
def test_unavailable_count_fails_closed(saves, route, failure):
    store, post = saves
    setattr(store, failure, True)
    assert post(route).status_code == 503
    assert store.writes == []


@pytest.mark.parametrize("route", ROUTES)
def test_pro_lookup_failure_does_not_bypass_free_limit(saves, route):
    store, post = saves
    store.seed_saved(30)
    store.pro_failure = True
    assert post(route).json() == QUOTA_ERROR
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
