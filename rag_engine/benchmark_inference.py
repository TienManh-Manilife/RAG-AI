"""Benchmark-only inference variants. Production retrieval and pipeline are unmodified.
No reference answer, page, required term, or gold decision enters inference.
"""
from __future__ import annotations
import json,time,math
from . import retrieval as r
from . import pipeline
from .generation import generate_answer
from .handoff import personal_handoff,should_handoff
from .config import EMBED_MODEL

METHODS=('lexical','dense','standard_hybrid','hybrid','no_gate')

def retrieve_variant(question,actor,method):
    if method in ('lexical','hybrid','no_gate'):
        with r.strict_dense_benchmark(method!='lexical'):
            return r.retrieve_context(question,actor=actor,top_k=5,use_dense=method!='lexical')
    # Same candidate universe and scoring primitives as production. Dense-only uses
    # cosine rank; standard hybrid keeps the same candidates but removes +0.03E.
    rows=r.authorized_chunks(actor)
    if not rows:return []
    from .ollama import embed_batch
    v=embed_batch([question])[0]
    if not v or not all(math.isfinite(x) for x in v):raise RuntimeError('Invalid query embedding')
    dense=[]
    for row in rows:
        if not row['embedding_json'] or row['embedding_model']!=EMBED_MODEL:raise RuntimeError('Missing current embedding')
        vec=json.loads(row['embedding_json'])
        if len(v)!=len(vec) or not all(math.isfinite(x) for x in vec):raise RuntimeError('Invalid document embedding')
        dense.append(max(0.,r.cosine(v,vec)))
    lexical=r._lexical(question,[x['title']+' '+x['section']+' '+x['content'] for x in rows]) if method=='standard_hybrid' else [(0.,0.)]*len(rows)
    rb={i:k for k,i in enumerate(sorted(range(len(rows)),key=lambda i:lexical[i][0],reverse=True),1)}
    rd={i:k for k,i in enumerate(sorted(range(len(rows)),key=lambda i:dense[i],reverse=True),1)}
    out=[]
    for i,row in enumerate(rows):
        bm,c=lexical[i];d=dense[i]
        if not bm and d<.45:continue
        s=max(0.,min(1.,(d-.30)/.43))
        e=.55*c+.45*s if method=='standard_hybrid' else s
        score=1/(50+rb[i])+1/(50+rd[i]) if method=='standard_hybrid' else d
        out.append((score,{'text':row['content'],'source':row['filename'],'title':row['title'],'section':row['section'],'page':row['page'],'chunk_id':row['id'],'audience':row['audience'],'similarity':round(d,4),'lexical_coverage':round(c,4),'evidence_score':round(e,4)}))
    return [h for _,h in sorted(out,key=lambda p:p[0],reverse=True)[:5]]


def infer(question,role,method,retrieval_only):
    actor={'role':role}
    if retrieval_only:
        if method=='no_gate':raise ValueError('no_gate changes only end-to-end routing')
        hits=retrieve_variant(question,actor,method)
        return dict(answer='',hits=hits,handoff=None,evidence_score=max((h['evidence_score'] for h in hits),default=0),reason='',error=None,meta={})
    if method in ('lexical','hybrid'):
        with r.strict_dense_benchmark(method=='hybrid'):
            return pipeline.answer_question(question,actor=actor,use_dense=method=='hybrid')
    # Mirror production guards; only the explicitly ablated feature changes.
    if question.casefold().strip(' .!?') in {'xin chào','chào','hello','hi','cảm ơn','thank you','tạm biệt'} or personal_handoff(question):
        return pipeline.answer_question(question,actor=actor,use_dense=True)
    hits=retrieve_variant(question,actor,method)
    score=max((h['evidence_score'] for h in hits),default=0)
    if not hits or (method!='no_gate' and score<.24):
        answer='Không tìm được căn cứ tương ứng trong kho tài liệu được phép tra cứu.';meta={'insufficient_evidence':True}
    else:answer,meta=generate_answer(question,hits)
    # no_gate keeps all other guards, including no-context and invalid citations.
    handoff,reason,score=should_handoff(hits,meta,min_similarity=0. if method=='no_gate' else None)
    return dict(answer=answer,hits=hits,handoff=handoff,evidence_score=score,reason=reason,error=None,meta=meta)


def raw_case(label,result,latency):
    # Freeze output and label side-by-side. Store evidence text and citation IDs
    # to support offline audit, not only the already-computed metrics.
    return {'label':dict(label),'prediction':{k:result.get(k) for k in ('answer','hits','handoff','evidence_score','reason','error','meta')},'latency_s':latency}


def predict_case(label,method,retrieval_only):
    start=time.monotonic()
    # Explicitly pass only the fields that the deployed model may observe.
    try:
        result=infer(label['question'],label['actor_role'],method,retrieval_only)
    except Exception as e:
        result=dict(answer='',hits=[],handoff=None,evidence_score=0,reason='',error=type(e).__name__,meta={})
    return raw_case(label,result,round(time.monotonic()-start,3))
