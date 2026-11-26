"""Reproducible labeled benchmark (50 samples). Scores computed, never fabricated.
Semantic correctness needs human adjudication; term coverage is only a proxy.
"""
from __future__ import annotations
import argparse,csv,io,json,re,time
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
from .config import DATA_DIR,ROOT
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
    return rows

def term_coverage(text,terms):
    if not terms: return None
    norm=fold(text)
    return sum(fold(t) in norm for t in terms)/len(terms)

def run_benchmark(rows=None, *, retrieval_only=False, use_dense=True, progress=None):
    rows=rows or read_cases()
    if not rows:raise ValueError('Không có câu kiểm thử.')
    output=[]
    started=time.time()
    for i,item in enumerate(rows):
        tic=time.monotonic()
        try:
            if retrieval_only:
                hits=retrieve_context(item['question'],actor={'role':'user'},top_k=5,use_dense=use_dense)
                result={'answer':'','hits':hits,'handoff':None,'evidence_score':max((r['evidence_score'] for r in hits),default=0),
                        'reason':'','error':None,'meta':{}}
            else:
                result=answer_question(item['question'],actor={'role':'user'},use_dense=use_dense)
            hit_sources=[h['source'] for h in result['hits']]
            expected=item.get('expected_source') or ''
            found=(expected in hit_sources) if expected else None
            cited=bool(result['meta'].get('cited_ids'))
            covered=term_coverage(result['answer'],item['required_terms']) if not retrieval_only and not result['handoff'] else None
            errors=[]
            if item['should_handoff'] is False and found is False: errors.append('retrieval_miss')
            if not retrieval_only:
                if item['should_handoff'] and not result['handoff']:errors.append('missed_handoff')
                if not item['should_handoff'] and result['handoff']:errors.append('false_handoff')
                if not item['should_handoff'] and not result['handoff'] and not cited: errors.append('missing_citation')
                if covered is not None and covered<1: errors.append('keyphrase_mismatch')
                if result['error']:errors.append('system_error')
            output.append({**{k:item[k] for k in ('id','category','question','reference_answer','expected_source','should_handoff')},
                'prediction':result['answer'],'pred_handoff':result['handoff'],
                'evidence_score':result['evidence_score'],'top_sources':hit_sources,
                'retrieval_hit_at_5':found,'citation_valid':cited,'required_terms':item['required_terms'],
                'term_coverage':covered,'reason':result['reason'],'errors':errors,
                'error':result['error'],'latency_s':round(time.monotonic()-tic,3)})
        except Exception as exc:
            output.append({**item,'prediction':'','pred_handoff':None,'top_sources':[],
                           'retrieval_hit_at_5':False,'citation_valid':False,'term_coverage':None,
                           'errors':['system_error'],'error':type(exc).__name__,
                           'evidence_score':0,'latency_s':round(time.monotonic()-tic,3)})
        if progress: progress(i+1,len(rows))
    return {'timestamp':datetime.now(timezone.utc).isoformat(),
            'mode':'retrieval_only' if retrieval_only else 'end_to_end',
            'duration_s':round(time.time()-started,2),
            'summary':summarize(output,retrieval_only=retrieval_only),'cases':output}

def summarize(rows,retrieval_only=False):
    n=len(rows)
    targets=[r for r in rows if not r['should_handoff']]
    hits=[r for r in targets if r['retrieval_hit_at_5'] is not None]
    metrics={'cases':n,'answerable_cases':len(targets),
             'insufficient_cases':n-len(targets),
             'retrieval_recall_at_5':sum(r['retrieval_hit_at_5'] is True for r in hits)/len(hits) if hits else None,
             'mean_latency_s':round(sum(r.get('latency_s',0) for r in rows)/n,3) if n else None,
             'error_counts':dict(Counter(err for r in rows for err in r['errors']))}
    if retrieval_only: return metrics
    completed=[r for r in rows if r['pred_handoff'] is not None]
    tp=sum(r['pred_handoff'] is True and r['should_handoff'] is True for r in completed)
    fp=sum(r['pred_handoff'] is True and r['should_handoff'] is False for r in completed)
    fn=sum(r['pred_handoff'] is False and r['should_handoff'] is True for r in completed)
    tn=sum(r['pred_handoff'] is False and r['should_handoff'] is False for r in completed)
    scoreable=[r for r in targets if r['pred_handoff'] is False and r['term_coverage'] is not None]
    metrics.update({'processed':len(completed), 'handoff_rate':sum(r['pred_handoff'] is True for r in completed)/len(completed) if completed else None,
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
                    and r['citation_valid'] is True and r['term_coverage']==1.0 for r in targets)
    correct_abstain=sum(r['pred_handoff'] is True for r in completed if r['should_handoff'])
    metrics['answer_accuracy_checklist_proxy']=strict_pass/len(targets) if targets else None
    metrics['auto_test_pass_rate_proxy']=(strict_pass+correct_abstain)/n if n else None
    metrics['unsupported_safe_handoff_rate']=correct_abstain/(n-len(targets)) if n>len(targets) else None
    return metrics

def write_report(report, report_dir=None):
    dest=Path(report_dir or DATA_DIR/'reports');dest.mkdir(parents=True,exist_ok=True)
    (dest/'benchmark_latest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    with (dest/'benchmark_latest.csv').open('w',encoding='utf-8-sig',newline='') as f:
        cols=['id','category','question','expected_source','should_handoff','prediction','pred_handoff','retrieval_hit_at_5','citation_valid','term_coverage','evidence_score','reason','errors','latency_s']
        writer=csv.DictWriter(f,fieldnames=cols);writer.writeheader()
        for r in report['cases']:
            writer.writerow({k:';'.join(map(str,r.get(k,[]))) if isinstance(r.get(k),list) else r.get(k,'') for k in cols})
    return dest

def main():
    p=argparse.ArgumentParser();p.add_argument('--retrieval-only',action='store_true')
    p.add_argument('--lexical-only',action='store_true')
    args=p.parse_args()
    from .auth import init_db
    from .documents import init_docs,seed_documents
    init_db();init_docs();seed_documents()
    report=run_benchmark(retrieval_only=args.retrieval_only,use_dense=not args.lexical_only,
                         progress=lambda i,n:print(f'[{i}/{n}]',flush=True))
    dest=write_report(report)
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2))
    print('Saved:',dest/'benchmark_latest.json')
if __name__=='__main__':main()
