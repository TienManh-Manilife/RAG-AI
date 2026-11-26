import pytest
from rag_engine import auth,documents,service
from rag_engine.retrieval import retrieve_context

def test_login_happy_path(setup_db):
    user,_,_=setup_db
    assert auth.login(user['email'],'CustomerPass1234')['id']==user['id']

def test_login_wrong_password(setup_db):
    assert auth.login('khach@example.com','bad') is None

def test_user_cannot_self_register_staff(setup_db):
    with pytest.raises(PermissionError):
        auth.create_account('Staff Fake','fake@example.com','FakePass123456',role='staff')

def test_user_cannot_create_admin(setup_db):
    with pytest.raises(PermissionError):
        auth.create_account('Admin Fake','fake@example.com','FakePass123456',role='admin')

def test_user_cannot_ingest(setup_db):
    user,_,_=setup_db
    with pytest.raises(PermissionError):documents.ingest(user,'test.md',b'## Muc 1\n' + b'a'*80)

def test_staff_can_ingest(setup_db):
    _,staff,_=setup_db
    result=documents.ingest(staff,'abc.md','## Quy định\nThời hạn là 24 tháng đối với bảo hành'.encode())
    assert result['chunks']==1

def test_malicious_extension_rejected(setup_db):
    _,staff,_=setup_db
    with pytest.raises(ValueError):documents.ingest(staff,'payload.exe',b'echo hello')

def test_visibility_filter_and_retrieval(setup_db):
    user,staff,_=setup_db
    documents.ingest(staff,'public.md','## Hỗ trợ\nBảo hành 12 tháng cho máy tính.'.encode(),audience='customer')
    documents.ingest(staff,'private.md','## Bí mật\nLương nhân viên 99 triệu một tháng.'.encode(),audience='staff')
    assert len(documents.list_docs(user))==1
    assert len(documents.list_docs(staff))==2
    hits=retrieve_context('lương nhân viên 99 triệu',actor=user,use_dense=False)
    assert not any('private.md'==h['source'] for h in hits)
    hits_staff=retrieve_context('lương nhân viên 99 triệu',actor=staff,use_dense=False)
    assert any('private.md'==h['source'] for h in hits_staff)

def test_user_cannot_delete(setup_db):
    user,staff,_=setup_db
    d=documents.ingest(staff,'abc.md',b'## Huong dan\n'+b'abcd '*30)
    with pytest.raises(PermissionError):documents.remove(user,d['id'])

def test_seeding_not_recreated_after_delete(setup_db):
    user,staff,_=setup_db
    assert documents.seed_documents()==2
    docs=documents.list_docs(staff)
    documents.remove(staff,docs[0]['id'])
    assert documents.seed_documents()==0
    assert len(documents.list_docs(staff))==len(docs)-1

def test_conversation_isolation(setup_db):
    user,staff,_=setup_db
    other=auth.create_account('Khách B','khachb@example.com','CustomerPass12345')
    c=service.new_conversation(user,'system','Test')
    service.add_message(user,c,'user','Xin chào')
    with pytest.raises(PermissionError):service.list_messages(other,c)

def test_handoff_auto_workflow(setup_db):
    user,staff,_=setup_db
    c=service.new_conversation(user,'system','Bảo hành')
    service.add_message(user,c,'user','Chưa biết chính sách')
    m=service.add_message(user,c,'assistant','Không đủ căn cứ',needs_handoff=True,reason='Thiếu nguồn')
    h=service.request_handoff(user,c,m,'','','','')
    assert service.request_handoff(user,c,m,'','','','')==h
    assert service.list_my_handoffs(user)[0]['status']=='pending'
    service.update_handoff(staff,h,'take')
    assert service.list_my_handoffs(user)[0]['status']=='in_progress'
    service.update_handoff(staff,h,'resolve','Đã xác minh qua chính sách.')
    assert service.list_my_handoffs(user)[0]['status']=='resolved'
    service.customer_reopen(user,h)
    assert service.list_my_handoffs(user)[0]['status']=='pending'

def test_customer_cannot_take_ticket(setup_db):
    user,_,_=setup_db
    with pytest.raises(PermissionError):service.update_handoff(user,'fake','take')

def test_cannot_reopen_other_users_ticket(setup_db):
    user,staff,_=setup_db
    other=auth.create_account('Khách B','khachb@example.com','CustomerPass12345')
    c=service.new_conversation(user,'system','Help')
    service.add_message(user,c,'user','Q')
    m=service.add_message(user,c,'assistant','A')
    h=service.request_handoff(user,c,m,'','','',[])
    service.update_handoff(staff,h,'take');service.update_handoff(staff,h,'resolve','Ok')
    with pytest.raises(PermissionError):service.customer_reopen(other,h)

def test_account_deactivation(setup_db):
    user,_,admin=setup_db
    auth.set_account_active(admin,user['id'],False)
    assert auth.login(user['email'],'CustomerPass1234') is None
    assert auth.get_user(user['id']) is None

def test_admin_cannot_self_disable(setup_db):
    _,_,admin=setup_db
    with pytest.raises(ValueError):auth.set_account_active(admin,admin['id'],False)

def test_user_cannot_enumerate_accounts(setup_db):
    user,_,_=setup_db
    with pytest.raises(PermissionError):auth.list_accounts(user)

def test_persisted_source_null_similarity(setup_db):
    user,_,_=setup_db
    c=service.new_conversation(user,'system','Sources')
    m=service.add_message(user,c,'assistant','Trả lời',hits=[{'source':'a.md','page':1,
                              'similarity':None,'evidence_score':0.5}])
    assert service.list_messages(user,c)[0]['sources'][0]['similarity']==0.0
