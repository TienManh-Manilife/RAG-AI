from rag_engine import benchmark as b, documents


def test_pdf_dataset_labels():
    rows=b.read_cases()
    assert len(rows)==50
    assert sum(r['actor_role']=='user' for r in rows)==38
    assert sum(r['actor_role']=='staff' for r in rows)==12
    assert sum(r['should_handoff'] for r in rows)==10
    assert all(r['expected_page'] and r['expected_clause'] for r in rows if not r['should_handoff'])
    assert all(not r['expected_source'] and r['expected_page'] is None for r in rows if r['should_handoff'])


def test_exact_page_not_filename_only():
    item={'expected_source':'policy.pdf','expected_page':4}
    assert not b.source_match({'source':'policy.pdf','page':3},item)
    assert b.source_match({'source':'policy.pdf','page':4},item)


def test_fact_group_matching():
    assert b.term_coverage('Trong 5 ngày làm việc.',[['05 ngày','5 ngày'],'làm việc'])==1
    assert b.term_coverage('Trong 15 ngày làm việc.',['5 ngày'])==0
    assert b.term_coverage('Không rõ.\n\n[Nguồn 1: 12 tháng]',['12 tháng'])==0


def test_cited_wrong_page(monkeypatch):
    item=b.read_cases()[0]
    hits=[{'source':item['expected_source'],'page':2,'audience':'customer'},
          {'source':item['expected_source'],'page':1,'audience':'customer'}]
    monkeypatch.setattr(b,'answer_question',lambda *a,**k:dict(answer=item['reference_answer'],hits=hits,handoff=False,evidence_score=.9,reason='',error=None,meta={'cited_ids':[1]}))
    report=b.run_benchmark([item]); row=report['cases'][0]
    assert row['first_relevant_rank']==2 and row['retrieval_hit_at_5'] and not row['citation_valid']
    assert report['summary']['answer_accuracy_checklist_proxy']==0


def test_service_error_not_correct_handoff(monkeypatch):
    item=next(r for r in b.read_cases() if r['should_handoff'])
    monkeypatch.setattr(b,'answer_question',lambda *a,**k:dict(answer='Error',hits=[],handoff=True,evidence_score=0,reason='Error',error='ConnectionError',meta={}))
    r=b.run_benchmark([item])['summary']
    assert r['system_error_rate']==1 and r['operational_handoff_rate']==1
    assert r['auto_test_pass_rate_proxy']==0 and r['unsupported_safe_handoff_rate']==0


def test_role_and_corpus(setup_db):
    documents.seed_documents()
    rows=b.read_cases();b.validate_corpus(rows)
    report=b.run_benchmark(rows,retrieval_only=True,use_dense=False)
    assert report['by_role']['user']['cases']==38
    assert report['by_role']['staff']['cases']==12
    assert report['summary']['unauthorized_retrieval_count']==0
    assert report['summary']['retrieval_recall_at_5']>=.9
