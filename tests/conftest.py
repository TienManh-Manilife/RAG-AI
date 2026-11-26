import pytest
from rag_engine import auth,documents

@pytest.fixture
def setup_db(tmp_path,monkeypatch):
    monkeypatch.setattr(auth,'DATA_DIR',tmp_path)
    monkeypatch.setattr(auth,'SQLITE_PATH',tmp_path/'test.sqlite3')
    monkeypatch.setattr(auth,'get_auth_config',lambda:{'staff_invite_code':'','bootstrap_admin_email':'',
        'bootstrap_admin_password':'','bootstrap_admin_name':'Admin'})
    auth.init_db();documents.init_docs()
    user=auth.create_account('Khách A','khach@example.com','CustomerPass1234')
    staff=auth.create_account('Nhân viên','staff@example.com','StaffPass123456',role='staff',_privileged=True)
    admin=auth.create_account('Quản trị','admin@example.com','AdminPass123456',role='admin',_privileged=True)
    return user,staff,admin
