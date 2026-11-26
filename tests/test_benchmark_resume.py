import json
import pytest
from rag_engine import benchmark as b, benchmark_checkpoint as c, retrieval, ollama


def mock_answer(item, *, error=None, wrong=False):
    hits=[{'source':item['expected_source'],'page':item['expected_page'],'audience':'customer'}]
    return {'answer':'wrong' if wrong else item['reference_answer'],'hits':hits,'handoff':False,
            'evidence_score':.9,'reason':'','error':error,'meta':{'cited_ids':[1]}}


@pytest.fixture
def fake_identity(monkeypatch):
    monkeypatch.setattr(c,'_identity',lambda rows,ro,dense: {'rows':c._digest(rows),'ro':ro,'dense':dense})


def test_resume_skips_wrong_but_completed_and_retries_error(tmp_path,monkeypatch,fake_identity):
    rows=b.read_cases()[:3]
    lookup={x['question']:x for x in rows}; calls=[]
    def first(q,**kwargs):
        calls.append(lookup[q]['id'])
        return mock_answer(lookup[q],wrong=True,error='ConnectionError' if q==rows[1]['question'] else None)
    monkeypatch.setattr(b,'answer_question',first)
    r,dest=c.run_resumable_benchmark(rows,use_dense=False,report_dir=tmp_path)
    assert calls==[rows[0]['id'],rows[1]['id']]
    assert r['completed_cases']==1 and not r['complete'] and r['stop_reason']=='system_error'
    saved=json.loads((dest/'checkpoint.txt').read_text(encoding='utf-8'))
    assert list(saved['completed'])==[rows[0]['id']]
    assert 'keyphrase_mismatch' in saved['completed'][rows[0]['id']]['errors']
    assert saved['last_failure']['id']==rows[1]['id']
    calls.clear()
    def second(q,**kwargs):
        calls.append(lookup[q]['id']);return mock_answer(lookup[q])
    monkeypatch.setattr(b,'answer_question',second)
    r,_=c.run_resumable_benchmark(rows,use_dense=False,report_dir=tmp_path)
    assert calls==[rows[1]['id'],rows[2]['id']]
    assert r['complete'] and r['new_this_run']==2 and r['skipped_this_run']==1
    assert len(r['cases'])==3 and r['failed_attempts']==1
    assert r['summary']['answer_accuracy_checklist_proxy']==pytest.approx(2/3)
    calls.clear()
    r,_=c.run_resumable_benchmark(rows,use_dense=False,report_dir=tmp_path)
    assert calls==[] and r['skipped_this_run']==3


def test_batch_and_mode_isolation(tmp_path,monkeypatch,fake_identity):
    rows=b.read_cases()[:3]
    lookup={x['question']:x for x in rows}
    monkeypatch.setattr(b,'answer_question',lambda q,**kw:mock_answer(lookup[q]))
    r,a=c.run_resumable_benchmark(rows,use_dense=False,report_dir=tmp_path,max_new=1)
    assert r['completed_cases']==1 and r['stop_reason']=='batch_limit'
    r,_=c.run_resumable_benchmark(rows,use_dense=False,report_dir=tmp_path,max_new=1)
    assert r['completed_cases']==2 and r['new_this_run']==1
    r,z=c.run_resumable_benchmark(rows,use_dense=True,report_dir=tmp_path,max_new=1)
    assert a!=z and r['completed_cases']==1
    r,new=c.run_resumable_benchmark(rows,use_dense=False,report_dir=tmp_path,run_id='rep2',max_new=1)
    assert new!=a and r['skipped_this_run']==0


def test_changed_labels_refused(tmp_path,monkeypatch,fake_identity):
    rows=b.read_cases()[:1]
    monkeypatch.setattr(b,'answer_question',lambda *a,**kw:mock_answer(rows[0]))
    c.run_resumable_benchmark(rows,use_dense=False,report_dir=tmp_path)
    modified=[dict(rows[0],reference_answer='modified')]
    with pytest.raises(RuntimeError,match='changed'):
        c.run_resumable_benchmark(modified,use_dense=False,report_dir=tmp_path)


def test_checkpoint_survives_interrupt(tmp_path,monkeypatch,fake_identity):
    rows=b.read_cases()[:2]
    def first(q,**kwargs):
        if q==rows[1]['question']:raise KeyboardInterrupt()
        return mock_answer(rows[0])
    monkeypatch.setattr(b,'answer_question',first)
    with pytest.raises(KeyboardInterrupt):
        c.run_resumable_benchmark(rows,use_dense=False,report_dir=tmp_path)
    path=next(tmp_path.rglob('checkpoint.txt'))
    assert list(json.loads(path.read_text(encoding='utf-8'))['completed'])==[rows[0]['id']]
    monkeypatch.setattr(b,'answer_question',lambda q,**kw:mock_answer(rows[1]))
    r,_=c.run_resumable_benchmark(rows,use_dense=False,report_dir=tmp_path)
    assert r['complete'] and r['skipped_this_run']==1


def test_strict_dense_does_not_change_live_fallback(monkeypatch):
    record={'title':'bảo hành','section':'thời hạn','content':'bảo hành 12 tháng', 'embedding_json':'[1,0]',
            'embedding_model':retrieval.EMBED_MODEL,'filename':'policy.pdf','page':1,'id':'a','audience':'customer'}
    monkeypatch.setattr(retrieval,'authorized_chunks',lambda actor:[record])
    def fail(*a,**kw):raise ollama.OllamaUnavailable('offline')
    monkeypatch.setattr(ollama,'embed_batch',fail)
    assert retrieval.retrieve_context('bảo hành')[0]['similarity'] is None
    with retrieval.strict_dense_benchmark():
        with pytest.raises(ollama.OllamaUnavailable):retrieval.retrieve_context('bảo hành')
    assert retrieval.retrieve_context('bảo hành')[0]['similarity'] is None


def test_real_lexical_checkpoint(tmp_path,setup_db):
    from rag_engine import documents
    documents.seed_documents()
    rows=b.read_cases()
    b.validate_corpus(rows)
    r,d=c.run_resumable_benchmark(rows,use_dense=False,retrieval_only=True,report_dir=tmp_path,max_new=10)
    assert r['completed_cases']==10 and not r['complete']
    r,_=c.run_resumable_benchmark(rows,use_dense=False,retrieval_only=True,report_dir=tmp_path)
    assert r['complete'] and r['summary']['retrieval_recall_at_5']==1 and r['new_this_run']==40
    assert (d/'benchmark_latest.csv').is_file()
