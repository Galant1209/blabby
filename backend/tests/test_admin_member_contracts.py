"""Admin member pagination and access contracts; no real services."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import subprocess
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
import main

UID = '10000000-0000-4000-8000-000000000001'
OTHER = '20000000-0000-4000-8000-000000000002'

class Query:
    def __init__(self, db, table):
        self.db, self.table = db, table
        self.filters, self.orders, self.bounds = [], [], None
        self.columns = ''
    def select(self, columns, **kw):
        self.columns = columns
        return self
    def eq(self, key, value):
        self.filters.append((key,value)); return self
    def order(self, key, desc=False):
        self.orders.append((key,desc)); return self
    def range(self, start, end):
        self.bounds=(start,end); return self
    def limit(self, limit):
        self.bounds=(0,limit-1); return self
    def execute(self):
        self.db.calls.append(self)
        if self.db.fail_tags and self.columns == 'weakness_tag':
            raise RuntimeError('metadata unavailable')
        rows=[dict(r) for r in self.db.rows[self.table] if all(r.get(k)==v for k,v in self.filters)]
        for key,desc in reversed(self.orders):
            rows.sort(key=lambda r:r.get(key,''),reverse=desc)
        count=len(rows)
        if self.bounds:
            rows=rows[self.bounds[0]:self.bounds[1]+1]
        return SimpleNamespace(data=rows,count=count)

class DB:
    def __init__(self):
        self.calls=[]; self.fail_tags=False
        self.rows={'practice_records':[{'id':f'{i:04}', 'user_id':UID, 'created_at':f'{i:04}', 'weakness_tag':'lack_detail'} for i in range(1005)] + [{'id':'other','user_id':OTHER,'created_at':'9999'}], 'subscriptions':[{'id':'s1','user_id':UID},{'id':'s2','user_id':OTHER}]}
        self.auth=SimpleNamespace(admin=SimpleNamespace(get_user_by_id=lambda uid:SimpleNamespace(user=SimpleNamespace(email='test@example.invalid'))))
    def table(self, table): return Query(self,table)

@pytest.fixture
def client_db():
    db=DB()
    with patch.object(main,'supabase_admin',db),patch.object(main,'verify_admin',return_value='admin'):
        yield TestClient(main.app),db

def test_speaking_newest_page_total_and_complete_lightweight_weakness_summary(client_db):
    client,db=client_db
    response=client.get(f'/admin/user/{UID}')
    assert response.status_code==200
    data=response.json()
    assert len(data['records'])==10
    assert data['records'][0]['id']=='1004'
    assert data['total_records']==1005
    assert data['weakness_counts']=={'lack_detail':1005}
    assert data['has_more'] is True
    assert all(('user_id',UID) in q.filters for q in db.calls)
    assert [q.bounds for q in db.calls if q.columns=='weakness_tag']==[(0,499),(500,999),(1000,1499)]

def test_speaking_next_page_does_not_repeat_summary(client_db):
    client,db=client_db
    data=client.get(f'/admin/user/{UID}?offset=1000&limit=10').json()
    assert len(data['records'])==5
    assert data['has_more'] is False
    assert len(db.calls)==1

@pytest.mark.parametrize('query',['limit=0','limit=101','offset=-1'])
def test_pagination_bounds(client_db,query):
    assert client_db[0].get(f'/admin/user/{UID}?{query}').status_code==422

def test_speaking_summary_failure_is_not_fake_zero(client_db):
    client,db=client_db; db.fail_tags=True
    data=client.get(f'/admin/user/{UID}').json()
    assert data['weakness_counts'] is None
    assert len(data['records'])==10

def test_scoped_subscription_lookup(client_db):
    client,db=client_db
    data=client.get(f'/api/admin/subscriptions?user_id={UID}').json()
    assert [r['id'] for r in data['subscriptions']]==['s1']
    assert ('user_id',UID) in db.calls[0].filters
    assert len(client.get('/api/admin/subscriptions').json()['subscriptions'])==2
    assert client.get('/api/admin/subscriptions?user_id=bad').status_code==400

@pytest.mark.parametrize('path',[f'/admin/user/{UID}',f'/api/admin/subscriptions?user_id={UID}',f'/admin/writing/submissions?user_id={UID}',f'/admin/reading/attempts?user_id={UID}'])
def test_backend_admin_guard_precedes_database(path):
    with patch.object(main,'verify_admin',side_effect=HTTPException(status_code=403)),patch.object(main,'supabase_admin') as db:
        assert TestClient(main.app).get(path).status_code==403
        db.table.assert_not_called()

def test_frontend_admin_member_behavior():
    harness=Path(__file__).with_name('frontend_admin_member_behavior.mjs')
    result=subprocess.run(['node',str(harness)],capture_output=True,text=True)
    assert result.returncode==0,result.stdout+result.stderr
