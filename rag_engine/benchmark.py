"""Reproducible labeled benchmark (50 samples). Scores computed, never fabricated.
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

def read_cases(path=FIXTURE):
    with Path(path).open('r',encoding='utf-8-sig',newline='') as f:
        rows=list(csv.DictReader(f))
    if len(rows)!=50 and Path(path)==FIXTURE:
        raise ValueError('Bộ chuẩn phải có đúng 50 câu hỏi.')
    for row in rows:
        row['should_handoff']=str(row['should_handoff']).strip().lower() in {'1','true','yes','có'}
        row['required_terms']=json.loads(row.get('required_terms') or '[]')
        row['actor_role']=row.get('actor_role') or 'user'
        if row['actor_role'] not in {'user','staff'}: raise ValueError('Vai trò không hợp lệ')
        row['expected_page']=int(row['expected_page']) if row.get('expected_page') else None
        if not row['should_handoff'] and (not row.get('expected_source') or not row['expected_page']):
            raise ValueError(f"{row['id']}: thiếu nguồn/trang chuẩn")
    if len({r['id'] for r in rows}) != len(rows): raise ValueError('Trùng ID câu hỏi')
    return rows

def term_coverage(text,terms):
    """Each item is a mandatory group; a list denotes accepted alternatives."""
    if not terms: return None
    # Exclude generated citation suffix from fact matching.
    norm=fold(re.split(r'\[Nguồn \d+:',text or '',maxsplit=1,flags=re.I)[0])
    norm=re.sub(r'[–—−]', '-', norm)
    def match(t):
        t=re.sub(r'[–—−]', '-', fold(t))
        return bool(re.search(r'(?<!\w)'+re.escape(t)+r'(?!\w)',norm))
    return sum(any(match(t) for t in (group if isinstance(group,list) else [group])) for group in terms)/len(terms)

def source_match(hit, item):
    return (hit.get('source') == item.get('expected_source') and
            (item.get('expected_page') is None or hit.get('page') == item['expected_page']))

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
        if not row['should_handoff'] and (row['expected_source'],row['expected_page']) not in available[row['actor_role']]:
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
        try:
            if retrieval_only:
                hits=retrieve_context(item['question'],actor={'role':role},top_k=5,use_dense=use_dense)
                result={'answer':'','hits':hits,'handoff':None,'evidence_score':max((r['evidence_score'] for r in hits),default=0),
                        'reason':'','error':None,'meta':{}}
            else:
                result=answer_question(item['question'],actor={'role':role},use_dense=use_dense)
            hits=result['hits']
            ranks=[j for j,h in enumerate(hits,1) if source_match(h,item)] if item.get('expected_source') else []
            found=bool(ranks) if item.get('expected_source') else None
            cited_hits=[hits[j-1] for j in result['meta'].get('cited_ids',[]) if isinstance(j,int) and 1<=j<=len(hits)]
            cited=(any(source_match(h,item) for h in cited_hits) if item.get('expected_source') else bool(cited_hits)) if not retrieval_only else None
            leak=role=='user' and any(h.get('audience')=='staff' for h in hits)
            covered=term_coverage(result['answer'],item['required_terms']) if not retrieval_only and not result['handoff'] else None
            errors=[]
            if not item['should_handoff'] and found is False: errors.append('retrieval_miss')
            if leak:errors.append('unauthorized_retrieval')
            if not retrieval_only:
                if item['should_handoff'] and not result['handoff']:errors.append('missed_handoff')
                if not item['should_handoff'] and result['handoff']:errors.append('false_handoff')
                if not item['should_handoff'] and not result['handoff'] and not cited: errors.append('wrong_or_missing_citation')
                if covered is not None and covered<1: errors.append('keyphrase_mismatch')
                if result['error']:errors.append('system_error')
            def ref(h):return {'source':h.get('source'),'page':h.get('page'),'chunk_id':h.get('chunk_id'),'audience':h.get('audience')}
            output.append({**base,'prediction':result['answer'],'pred_handoff':result['handoff'],
                'evidence_score':result['evidence_score'],'top_sources':[h['source'] for h in hits],
                'top_refs':[ref(h) for h in hits],'cited_refs':[ref(h) for h in cited_hits],
                'retrieval_hit_at_5':found,'first_relevant_rank':min(ranks) if ranks else None,
                'citation_valid':cited,'term_coverage':covered,'reason':result['reason'],'errors':errors,
                'error':result['error'],'latency_s':round(time.monotonic()-tic,3)})
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

def summarize(rows,retrieval_only=False):
    n=len(rows)
    targets=[r for r in rows if not r['should_handoff']]
    hits=[r for r in targets if r['retrieval_hit_at_5'] is not None]
    metrics={'cases':n,'answerable_cases':len(targets),
             'insufficient_cases':n-len(targets),
             'retrieval_recall_at_5':sum(r['retrieval_hit_at_5'] is True for r in hits)/len(hits) if hits else None,
             'mean_latency_s':round(sum(r.get('latency_s',0) for r in rows)/n,3) if n else None,
             'error_counts':dict(Counter(err for r in rows for err in r['errors']))}
    if any('first_relevant_rank' in r for r in targets):
        for k in (1,3,5):
            metrics[f'retrieval_recall_at_{k}']=sum(0<(r.get('first_relevant_rank') or 0)<=k for r in targets)/len(targets) if targets else None
        metrics['mrr_at_5']=sum(1/r['first_relevant_rank'] if r.get('first_relevant_rank') else 0 for r in targets)/len(targets) if targets else None
    metrics['unauthorized_retrieval_count']=sum('unauthorized_retrieval' in r['errors'] for r in rows)
    if retrieval_only: return metrics
    completed=[r for r in rows if r['pred_handoff'] is not None and 'system_error' not in r['errors']]
    tp=sum(r['pred_handoff'] is True and r['should_handoff'] is True for r in completed)
    fp=sum(r['pred_handoff'] is True and r['should_handoff'] is False for r in completed)
    fn=sum(r['pred_handoff'] is False and r['should_handoff'] is True for r in completed)
    tn=sum(r['pred_handoff'] is False and r['should_handoff'] is False for r in completed)
    scoreable=[r for r in targets if r['pred_handoff'] is False and r['term_coverage'] is not None and 'system_error' not in r['errors']]
    metrics.update({'processed':len(completed), 'handoff_rate':sum(r['pred_handoff'] is True for r in completed)/len(completed) if completed else None,
       'operational_handoff_rate':sum(r['pred_handoff'] is True for r in rows)/n if n else None,
       'expected_handoff_rate':sum(r['should_handoff'] for r in rows)/n if n else None,
       'handoff_precision':tp/(tp+fp) if tp+fp else None,
       'handoff_recall':tp/(tp+fn) if tp+fn else None,
       'handoff_specificity':tn/(tn+fp) if tn+fp else None,
       'handoff_f1':2*tp/(2*tp+fp+fn) if (2*tp+fp+fn) else None,
       'confusion':{'tp':tp,'fp':fp,'fn':fn,'tn':tn},
       'answerable_completion_rate':len(scoreable)/len(targets) if targets else None,
       'citation_rate_on_answered':sum(r['citation_valid'] is True for r in scoreable)/len(scoreable) if scoreable else None,
       'keyphrase_coverage_proxy':sum(r['term_coverage'] for r in scoreable)/len(scoreable) if scoreable else None,
       'keyphrase_full_match_proxy':sum(r['term_coverage']==1 for r in scoreable)/len(scoreable) if scoreable else None,
       'system_error_rate':sum('system_error' in r['errors'] for r in rows)/n if n else None})
    # Strict automated checklist: matched document + grounded citation + all required facts;
    # not equivalent to verified semantic accuracy. Abstention is credited only when labeled.
    strict_pass=sum(r['pred_handoff'] is False and r['retrieval_hit_at_5'] is True
                    and r['citation_valid'] is True and r['term_coverage']==1.0 and not set(r['errors']) & {'system_error','unauthorized_retrieval'} for r in targets)
    correct_abstain=sum(r['pred_handoff'] is True and 'unauthorized_retrieval' not in r['errors'] for r in completed if r['should_handoff'])
    metrics['answer_accuracy_checklist_proxy']=strict_pass/len(targets) if targets else None
    metrics['auto_test_pass_rate_proxy']=(strict_pass+correct_abstain)/n if n else None
    metrics['unsupported_safe_handoff_rate']=correct_abstain/(n-len(targets)) if n>len(targets) else None
    return metrics

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
    p=argparse.ArgumentParser();p.add_argument('--retrieval-only',action='store_true')
    p.add_argument('--lexical-only',action='store_true')
    p.add_argument('--role',choices=('user','staff','all'),default='all')
    p.add_argument('--run-id',default='default',help='Separate checkpoint name for an independent experiment')
    p.add_argument('--max-new',type=int,help='Maximum new cases per invocation')
    args=p.parse_args()
    from .auth import init_db
    from .documents import init_docs,seed_documents
    init_db();init_docs();seed_documents()
    rows=read_cases();validate_corpus(rows)
    if args.role!='all': rows=[r for r in rows if r['actor_role']==args.role]
    from .benchmark_checkpoint import run_resumable_benchmark
    report,dest=run_resumable_benchmark(rows,retrieval_only=args.retrieval_only,use_dense=not args.lexical_only,
                          role=args.role,run_id=args.run_id,max_new=args.max_new,
                          announce=lambda message:print(message,flush=True))
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2))
    print('Saved:',dest/'benchmark_latest.json')
    print(f"Completed: {report['completed_cases']}/{report['total_cases']} | {report['stop_reason']}")
    if report['stop_reason']=='system_error':
        raise SystemExit(2)
if __name__=='__main__':main()
