import json
from rag_engine import generation,pipeline

HITS=[{'title':'Chính sách bảo hành','source':'04_bao_hanh.md','section':'Mục 1',
       'page':1,'text':'Thiết bị tiêu chuẩn bảo hành 12 tháng.','evidence_score':0.9}]

def test_json_answer_citation(monkeypatch):
    monkeypatch.setattr(generation,'chat',lambda messages:'{"status":"answered","answer":"Bảo hành 12 tháng.","citations":[1]}')
    answer,meta=generation.generate_answer('Bảo hành bao lâu?',HITS)
    assert '12 tháng' in answer
    assert '[Nguồn 1:' in answer
    assert not meta['missing_citation']

def test_invalid_citation_escalates(monkeypatch):
    monkeypatch.setattr(generation,'chat',lambda messages:'{"status":"answered","answer":"Bảo hành 999 năm.","citations":[9]}')
    answer,meta=generation.generate_answer('Bảo hành bao lâu?',HITS)
    assert meta['insufficient_evidence']
    assert '999' not in answer

def test_insufficient_model_escalates(monkeypatch):
    monkeypatch.setattr(generation,'chat',lambda messages:'{"status":"insufficient","answer":"Không có đủ nguồn.","citations":[]}')
    answer,meta=generation.generate_answer('Giám đốc là ai?',HITS)
    assert meta['insufficient_evidence']

def test_invalid_model_json_escalates(monkeypatch):
    monkeypatch.setattr(generation,'chat',lambda messages:'The answer is yes')
    _,meta=generation.generate_answer('Bảo hành?',HITS)
    assert meta['answer_filtered']

def test_pipeline_generation_failure_produces_ticket_flag(monkeypatch):
    monkeypatch.setattr(pipeline,'retrieve_context',lambda *a,**kw:HITS)
    def fail(*a,**kw):raise ConnectionError('service offline')
    monkeypatch.setattr(pipeline,'generate_answer',fail)
    result=pipeline.answer_question('Bảo hành bao nhiêu?',actor={'role':'user'})
    assert result['handoff'] is True
    assert result['error']=='ConnectionError'
    assert result['evidence_score']==0

def test_no_source_does_not_generate(monkeypatch):
    monkeypatch.setattr(pipeline,'retrieve_context',lambda *a,**kw:[])
    def fail(*a,**kw):raise AssertionError('should never call LLM')
    monkeypatch.setattr(pipeline,'generate_answer',fail)
    result=pipeline.answer_question('Câu hoàn toàn xa lạ?',actor={'role':'user'})
    assert result['handoff'] is True
    assert 'căn cứ' in result['answer']
