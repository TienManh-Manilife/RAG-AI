import json
from rag_engine.retrieval import tokenize,cosine,fold,retrieve_context
from rag_engine.handoff import should_handoff,personal_handoff
from rag_engine.pipeline import answer_question
from rag_engine.benchmark import read_cases,summarize,term_coverage,run_benchmark
from rag_engine import documents

def test_fifty_golden_questions():
    rows=read_cases()
    assert len(rows)==50
    assert sum(not x['should_handoff'] for x in rows)==40
    assert sum(x['should_handoff'] for x in rows)==10

def test_vietnamese_no_accent_matching():
    assert fold('Bảo hành')=='bao hanh'
    assert 'bao' in tokenize('Bảo hành thiết bị')

def test_cosine():
    assert abs(cosine([1,0],[1,0])-1)<1e-9
    assert cosine([1,0],[0,1])==0

def test_missing_evidence_escalates():
    decision=should_handoff([],{'insufficient_evidence':True})
    assert decision[0] is True

def test_missing_citation_escalates():
    h=[{'evidence_score':0.95}]
    assert should_handoff(h,{'missing_citation':True})[0]

def test_sufficient_answer_not_escalated():
    h=[{'evidence_score':0.95}]
    assert not should_handoff(h,{'missing_citation':False})[0]

def test_personal_verified_case_escalates():
    assert personal_handoff('Trạng thái đơn hàng của tôi hiện giờ là gì?')

def test_phrase_coverage():
    assert term_coverage('Bảo hành 12 tháng',['12','tháng'])==1
    assert term_coverage('Bảo hành 6 tháng',['12','tháng'])==0.5

def test_summarize_handoff_confusion():
    rows=[{'should_handoff':True,'pred_handoff':True,'retrieval_hit_at_5':None,
           'citation_valid':False,'term_coverage':None,'latency_s':1,'errors':[]},
          {'should_handoff':False,'pred_handoff':False,'retrieval_hit_at_5':True,
           'citation_valid':True,'term_coverage':1.,'latency_s':1,'errors':[]}]
    metrics=summarize(rows)
    assert metrics['handoff_f1']==1.
    assert metrics['handoff_rate']==0.5

def test_retrieval_50_offline(setup_db):
    documents.seed_documents()
    result=run_benchmark(retrieval_only=True,use_dense=False)
    assert result['summary']['cases']==50
    assert result['summary']['retrieval_recall_at_5']>=.90

def test_pipeline_personal_no_llm(setup_db):
    r=answer_question('Đơn hàng của tôi đang ở đâu?',actor={'role':'user'},use_dense=False)
    assert r['handoff'] is True
    assert r['evidence_score']==0


def test_policy_questions_are_not_automatically_personal():
    assert not personal_handoff('Đăng ký cần mật khẩu tối thiểu bao nhiêu ký tự?')
    assert not personal_handoff('Nhân viên có yêu cầu OTP để hoàn tiền không?')


def test_greeting_does_not_open_employee_ticket(setup_db):
    result=answer_question('Xin chào',actor={'role':'user'},use_dense=False)
    assert result['handoff'] is False
    assert result['meta']['smalltalk'] is True
