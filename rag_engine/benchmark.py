"""Benchmark dataset loader and compatibility API; CLI defaults to 200 cases.
Semantic correctness needs human adjudication; term coverage is only a proxy.
"""
from __future__ import annotations
import argparse,csv,io,json,re,time
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path
from .config import DATA_DIR,ROOT,CHAT_MODEL,EMBED_MODEL,RAG_MIN_RELEVANCE
from .pipeline import answer_question
from .retrieval import retrieve_context,fold

FIXTURE=ROOT/'benchmark'/'questions_50.csv'
FIXTURE_200=ROOT/'benchmark'/'questions_200.csv'

def read_cases(path=FIXTURE):
    with Path(path).open('r',encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    expected={FIXTURE.resolve():50,FIXTURE_200.resolve():200}.get(Path(path).resolve())
    if expected is not None and len(rows)!=expected:
        raise ValueError(f'Bộ chuẩn phải có đúng {expected} câu hỏi.')
    for row in rows:
        row['should_handoff']=str(row['should_handoff']).strip().lower() in {'1','true','yes','có'}
        row['required_terms']=json.loads(row.get('required_terms') or '[]')
        row['actor_role']=row.get('actor_role') or 'user'
        if 'alternate_refs' in row: row['alternate_refs']=json.loads(row['alternate_refs'] or '[]')
        if row['actor_role'] not in {'user','staff'}: raise ValueError('Vai trò không hợp lệ')
        row['expected_page']=int(row['expected_page']) if row.get('expected_page') else None
        if not row['should_handoff'] and (not row.get('expected_source') or not row['expected_page']):
            raise ValueError(f"{row['id']}: thiếu nguồn/trang chuẩn")
    if len({r['id'] for r in rows}) != len(rows): raise ValueError('Trùng ID câu hỏi')
    return rows

from .benchmark_score import term_coverage, source_match, summarize, score_case

def validate_corpus(rows):
    """Fail early rather than report fake retrieval misses on a wrong corpus."""
    from .documents import authorized_chunks
    from .auth import db
    from .config import KNOWLEDGE_DIR
    import hashlib
    expected_audiences={'Chinh_sach_khach_hang_AnPhat.pdf':'customer',
                         'Quy_trinh_nghiep_vu_noi_bo_AnPhat.pdf':'staff'}
    with db() as c:
        docs=[dict(r) for r in c.execute('SELECT filename,audience,sha256 FROM documents')]
    if len(docs)!=2 or {d['filename']:d['audience'] for d in docs}!=expected_audiences:
        raise ValueError('Benchmark PDF v2 yêu cầu đúng hai PDF với phạm vi customer/staff; kiểm tra Kho tài liệu.')
    for d in docs:
        local=KNOWLEDGE_DIR/d['filename']
        if local.is_file() and hashlib.sha256(local.read_bytes()).hexdigest()!=d['sha256']:
            raise ValueError(f"{d['filename']}: phiên bản PDF trong DB khác file chuẩn; cần upload đúng bản.")
    available={role:{(h['filename'],h['page']) for h in authorized_chunks({'role':role})} for role in ('user','staff')}
    for row in rows:
        if not row['should_handoff']:
            refs=[(row['expected_source'],row['expected_page'])]+[(x['source'],x['page']) for x in row.get('alternate_refs',[])]
            if any(ref not in available[row['actor_role']] for ref in refs):
                raise ValueError(f"{row['id']}: thiếu nguồn/trang hoặc sai phạm vi.")

def run_benchmark(rows=None, *, retrieval_only=False, use_dense=True, progress=None):
    rows=read_cases() if rows is None else rows
    if not rows:raise ValueError('Không có câu kiểm thử.')
    output=[]
    started=time.time()
    for i,item in enumerate(rows):
        tic=time.monotonic()
        role=item.get('actor_role','user')
        base={k:item.get(k) for k in ('id','category','question','reference_answer','expected_source','expected_page','expected_clause','should_handoff','required_terms','notes')}
        base['actor_role']=role
        for k in ('split','question_type','alternate_refs'):
            if k in item:base[k]=item[k]
        try:
            if retrieval_only:
                hits=retrieve_context(item['question'],actor={'role':role},top_k=5,use_dense=use_dense)
                result={'answer':'','hits':hits,'handoff':None,'evidence_score':max((r['evidence_score'] for r in hits),default=0),
                        'reason':'','error':None,'meta':{}}
            else:
                result=answer_question(item['question'],actor={'role':role},use_dense=use_dense)
            from .benchmark_inference import raw_case
            output.append(score_case(raw_case(base,result,round(time.monotonic()-tic,3)),retrieval_only))
        except Exception as exc:
            output.append({**base,'prediction':'','pred_handoff':None,'top_sources':[], 'top_refs':[],'cited_refs':[],
                           'retrieval_hit_at_5':False if item.get('expected_source') else None,'first_relevant_rank':None,
                           'citation_valid':None,'term_coverage':None,'reason':'',
                           'errors':['system_error'],'error':type(exc).__name__,
                           'evidence_score':0,'latency_s':round(time.monotonic()-tic,3)})
        if progress: progress(i+1,len(rows))
    return {'timestamp':datetime.now(timezone.utc).isoformat(), 'dataset':'anphat_pdf_v2',
            'chat_model':CHAT_MODEL,'embedding_model':EMBED_MODEL,'dense_requested':use_dense,'min_relevance':RAG_MIN_RELEVANCE,
            'mode':'retrieval_only' if retrieval_only else 'end_to_end',
            'duration_s':round(time.time()-started,2),
            'summary':summarize(output,retrieval_only=retrieval_only),
            'by_role':{role:summarize([r for r in output if r['actor_role']==role],retrieval_only=retrieval_only) for role in ('user','staff') if any(r['actor_role']==role for r in output)},'cases':output}

def write_report(report, report_dir=None):
    dest=Path(report_dir or DATA_DIR/'reports');dest.mkdir(parents=True,exist_ok=True)
    (dest/'benchmark_latest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (dest/'benchmark_latest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        cols=['id','actor_role','category','question','reference_answer','expected_source','expected_page','expected_clause','should_handoff','required_terms','prediction','pred_handoff','first_relevant_rank','retrieval_hit_at_5','top_refs','cited_refs','citation_valid','term_coverage','evidence_score','reason','errors','error','latency_s']
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        for r in report['cases']:
            writer.writerow({k:json.dumps(r[k],ensure_ascii=False) if isinstance(r.get(k),list) else r.get(k,'') for k in cols})
    return dest

def main():
    from .benchmark_run import main as run_main
    run_main()

if __name__=='__main__':main()
