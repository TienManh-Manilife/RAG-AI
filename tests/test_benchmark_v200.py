import json, subprocess, sys
from collections import Counter
from pathlib import Path
import pytest
from rag_engine import benchmark as b,benchmark_run as runner,benchmark_score as scorer,benchmark_inference as infer,documents,retrieval,ollama


def test_dataset_200(setup_db):
    documents.seed_documents()
    old=b.read_cases();rows=b.read_cases(b.FIXTURE_200)
    assert len(rows)==200 and len({r['question'].casefold().strip() for r in rows})==200
    assert Counter(r['should_handoff'] for r in rows)=={False:160,True:40}
    assert Counter(r['split'] for r in rows)=={'legacy50':50,'new150':150}
    for a,c in zip(old,rows):assert all(c[k]==v for k,v in a.items())
    b.validate_corpus(rows)
    assert all(scorer.term_coverage(r['reference_answer'],r['required_terms'])==1 for r in rows[50:] if not r['should_handoff'])
    parsed=[json.loads(l) for l in b.FIXTURE_200.with_suffix('.jsonl').read_text(encoding='utf-8').splitlines()]
    assert all(all(a[k]==v for k,v in c.items()) for a,c in zip(rows,parsed))


def test_real_lexical_200_offline(tmp_path,setup_db):
    documents.seed_documents();rows=b.read_cases(b.FIXTURE_200)
    report,dest=runner.run(rows,method='lexical',retrieval_only=True,report_dir=tmp_path,announce=lambda _:None)
    assert report['complete'] and report['completed_cases']==200
    assert report['summary']['unauthorized_retrieval_count']==0
    assert report['by_split']['new150']['cases']==150
    raw=dest/'predictions.json';before=raw.read_bytes();csv_before=(dest/'predictions.csv').read_bytes()
    # Standalone regrading does not write raw artifacts or need the database.
    subprocess.run([sys.executable,'-m','rag_engine.benchmark_score',str(raw)],check=True)
    assert raw.read_bytes()==before and (dest/'predictions.csv').read_bytes()==csv_before
    assert (dest/'scores.csv').exists()
    subprocess.run([sys.executable,'scripts/parse_metrics.py',str(tmp_path)],check=True,capture_output=True)
    assert (tmp_path/'metrics_summary.csv').exists()
    subprocess.run([sys.executable,'-m','rag_engine.benchmark_compare',str(tmp_path)],check=True)
    assert (tmp_path/'evidence_threshold_diagnostic.csv').exists()


def test_resume_preserves_wrong_and_retries_error(tmp_path,monkeypatch):
    rows=b.read_cases(b.FIXTURE_200)[:3];calls=[]
    monkeypatch.setattr(runner,'identity',lambda rows,method,ro:dict(dataset_sha256=scorer.digest(rows),method=method,retrieval_only=ro))
    def fake(row,method,ro):
        calls.append(row['id']);err='ConnectionError' if row==rows[1] and calls.count(row['id'])==1 else None
        return infer.raw_case(row,dict(answer='wrong',hits=[],handoff=False,evidence_score=0,reason='',error=err,meta={}),.1)
    monkeypatch.setattr(runner,'predict_case',fake)
    r,d=runner.run(rows,method='lexical',report_dir=tmp_path,announce=lambda _:None)
    assert not r['complete'] and r['completed_cases']==1
    assert r['failed_attempts']==1
    raw=scorer.load_raw(d/'predictions.json');assert set(raw['completed'])=={rows[0]['id']}
    assert 'errors' not in raw['completed'][rows[0]['id']]['prediction']
    r,_=runner.run(rows,method='lexical',report_dir=tmp_path,announce=lambda _:None)
    assert r['complete'] and calls==[rows[0]['id'],rows[1]['id'],rows[1]['id'],rows[2]['id']]
    assert r['summary']['answer_accuracy_checklist_proxy']==0


def test_variant_scoring_and_no_label_leak(monkeypatch):
    records=[dict(id='a',title='bảo hành',section='quy định',content='bảo hành 12 tháng',filename='a.pdf',page=1,audience='customer',embedding_json='[0.9,0.1]',embedding_model=retrieval.EMBED_MODEL),dict(id='b',title='bảo trì',section='quy định',content='bảo trì thiết bị',filename='b.pdf',page=2,audience='customer',embedding_json='[1,0]',embedding_model=retrieval.EMBED_MODEL)]
    monkeypatch.setattr(retrieval,'authorized_chunks',lambda actor:records)
    monkeypatch.setattr(ollama,'embed_batch',lambda texts:[[1,0]])
    for method in ('lexical','hybrid'):
        assert infer.retrieve_variant('bảo hành',{'role':'user'},method)==retrieval.retrieve_context('bảo hành',actor={'role':'user'},use_dense=method=='hybrid')
    assert infer.retrieve_variant('bảo hành',{'role':'user'},'dense')[0]['chunk_id']=='b'
    assert {r['chunk_id'] for r in infer.retrieve_variant('bảo hành',{'role':'user'},'standard_hybrid')}=={'a','b'}
    item=b.read_cases()[0];seen=[]
    def fake(q,role,method,ro):
        seen.append((q,role,method,ro));return dict(answer='raw',hits=[],handoff=False,evidence_score=0,reason='',error=None,meta={})
    monkeypatch.setattr(infer,'infer',fake)
    infer.predict_case(item,'hybrid',False)
    assert seen==[(item['question'],item['actor_role'],'hybrid',False)]


def test_no_gate_only_removes_thresholds(monkeypatch):
    h={'evidence_score':.1}
    monkeypatch.setattr(infer,'retrieve_variant',lambda *a:[h])
    monkeypatch.setattr(infer,'generate_answer',lambda *a:('test',{'cited_ids':[1]}))
    assert infer.infer('Một câu hỏi thông thường','user','no_gate',False)['handoff'] is False
    assert infer.infer('Một câu hỏi thông thường','user','standard_hybrid',False)['handoff'] is True
    monkeypatch.setattr(infer,'generate_answer',lambda *a:('test',{'missing_citation':True}))
    assert infer.infer('Một câu hỏi thông thường','user','no_gate',False)['handoff'] is True
