"""Single source of truth for live chat and benchmark.
Any failure produces a transparent escalation rather than invented output.
"""
import logging
from .retrieval import retrieve_context
from .generation import generate_answer
from .handoff import should_handoff, personal_handoff
LOG=logging.getLogger(__name__)

def answer_question(question, actor=None, history=None, use_dense=True):
    question=(question or '').strip()
    if not question or len(question)>3000: raise ValueError('Câu hỏi phải có 1–3000 ký tự.')
    if question.casefold().strip(' .!?') in {'xin chào','chào','hello','hi','cảm ơn','thank you','tạm biệt'}:
        return {'answer':'Xin chào! Mình có thể hỗ trợ tra cứu chính sách bảo hành, đổi trả, giao hàng và các tài liệu đã được doanh nghiệp phê duyệt.',
                'hits':[],'meta':{'smalltalk':True},'handoff':False,'reason':'',
                'evidence_score':0.0,'error':None}
    if personal_handoff(question):
        ans='Cần xác minh thông tin cá nhân hoặc giao dịch. Vui lòng để nhân viên hỗ trợ trực tiếp.'
        metadata={'personal_verification':True}
        handoff, reason, score=should_handoff([],metadata)
        return {'answer':ans,'hits':[],'meta':metadata,'handoff':handoff,'reason':reason,'evidence_score':score,'error':None}
    try:
        hits=retrieve_context(question,actor=actor,top_k=5,use_dense=use_dense)
        score=max((x['evidence_score'] for x in hits),default=0)
        # Stop *before generation* for clearly out-of-scope questions.
        if score<0.24:
            ans='Không tìm được căn cứ tương ứng trong kho tài liệu được phép tra cứu.'
            meta={'insufficient_evidence':True}
        else:
            ans,meta=generate_answer(question,hits,history=history)
        escalate,reason,score=should_handoff(hits,meta)
        return {'answer':ans,'hits':hits,'meta':meta,'handoff':escalate,'reason':reason,'evidence_score':score,'error':None}
    except Exception as exc:
        LOG.exception('RAG pipeline failure')
        meta={'service_error':True}
        _,reason,_=should_handoff([],meta)
        return {'answer':'Hệ thống AI tạm thời chưa xử lý được. Yêu cầu được chuyển tới nhân viên.',
                'hits':[],'meta':meta,'handoff':True,'reason':reason,'evidence_score':0.0,
                'error':type(exc).__name__}
