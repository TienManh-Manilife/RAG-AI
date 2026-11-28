"""CPU-only scoring; no imports of the RAG runtime, database or Ollama."""
from __future__ import annotations
import argparse,csv,hashlib,json,math,os,re,tempfile,unicodedata
from collections import Counter
from pathlib import Path
from statistics import median

def fold(text):
    text=(text or '').casefold().replace('đ','d')
    return ''.join(c for c in unicodedata.normalize('NFD',text) if not unicodedata.combining(c))

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
    return any(hit.get('source') == ref.get('source') and
               (ref.get('page') is None or hit.get('page') == ref['page'])
               for ref in ([{'source':item.get('expected_source'),'page':item.get('expected_page')}] + item.get('alternate_refs',[])) if ref.get('source'))

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


def score_case(raw,retrieval_only=False):
    """Compare frozen predictions to labels, never modify the raw result."""
    label=raw['label']; pred=raw['prediction']; hits=pred['hits']
    ranks=[i for i,h in enumerate(hits[:5],1) if source_match(h,label)] if label.get('expected_source') else []
    found=bool(ranks) if label.get('expected_source') else None
    citations=[hits[i-1] for i in pred.get('meta',{}).get('cited_ids',[]) if type(i) is int and 1<=i<=len(hits)]
    cited=(any(source_match(h,label) for h in citations) if label.get('expected_source') else bool(citations)) if not retrieval_only else None
    leak=label.get('actor_role')=='user' and any(h.get('audience')=='staff' for h in hits)
    handoff=pred.get('handoff'); err=pred.get('error'); answer=pred.get('answer','')
    covered=term_coverage(answer,label.get('required_terms',[])) if not retrieval_only and handoff is False and not err else None
    errors=[]
    if not label['should_handoff'] and found is False: errors.append('retrieval_miss')
    if leak: errors.append('unauthorized_retrieval')
    if err: errors.append('system_error')
    if not retrieval_only and not err:
        if label['should_handoff'] and handoff is False: errors.append('missed_handoff')
        if not label['should_handoff'] and handoff is True: errors.append('false_handoff')
        if not label['should_handoff'] and handoff is False and not cited: errors.append('wrong_or_missing_citation')
        if covered is not None and covered<1: errors.append('keyphrase_mismatch')
    def ref(h):return {k:h.get(k) for k in ('source','page','chunk_id','audience')}
    strict=(not retrieval_only and not label['should_handoff'] and handoff is False and found is True and cited is True and covered==1 and not err and not leak)
    safe=(not retrieval_only and label['should_handoff'] and handoff is True and not err and not leak)
    return {**label,'prediction':answer,'pred_handoff':handoff,'evidence_score':pred.get('evidence_score',0),
            'top_sources':[h.get('source') for h in hits],'top_refs':[ref(h) for h in hits],'cited_refs':[ref(h) for h in citations],
            'retrieval_hit_at_5':found,'first_relevant_rank':min(ranks) if ranks else None,'citation_valid':cited,
            'term_coverage':covered,'reason':pred.get('reason',''),'errors':errors,'error':err,'latency_s':raw['latency_s'],
            'checklist_pass':strict if not retrieval_only and not label['should_handoff'] else None,
            'overall_pass':bool(strict or safe) if not retrieval_only else None}


def digest(obj):
    return hashlib.sha256(json.dumps(obj,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def atomic_json(path,obj):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name+'.')
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as f:
            json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def write_csv(path,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r)) or ['id']
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader()
        for r in rows:w.writerow({k:json.dumps(v,ensure_ascii=False) if isinstance(v,(dict,list)) else v for k,v in r.items()})


def score_summary(rows,retrieval_only):
    s=summarize(rows,retrieval_only)
    times=sorted(r['latency_s'] for r in rows)
    s.update(median_latency_s=median(times) if times else None,p95_latency_s=times[math.ceil(.95*len(times))-1] if times else None,
             sum_case_latency_s=round(sum(times),3) if times else None)
    if not retrieval_only:
        c=s['confusion'];tp,fp,fn,tn=(c[k] for k in ('tp','fp','fn','tn'))
        s.update(handoff_accuracy=(tp+tn)/(tp+fp+fn+tn) if tp+fp+fn+tn else None,
                 false_answer_rate_on_insufficient=fn/(tp+fn) if tp+fn else None,
                 false_handoff_rate_on_answerable=fp/(fp+tn) if fp+tn else None)
    reviewed=[r for r in rows if r.get('expert_label') in ('Đúng','Sai')]
    if reviewed:
        s['expert_reviewed']=len(reviewed);s['expert_accuracy']=sum(r['expert_label']=='Đúng' for r in reviewed)/len(reviewed)
    return s


def load_raw(path):
    state=json.loads(Path(path).read_text(encoding='utf-8-sig'))
    if state.get('schema')!='rag-ai-raw-v2':raise ValueError('Not a raw benchmark v2 prediction file')
    ids=state['case_ids']; completed=state['completed'];labels=state['labels']
    if len(ids)!=len(set(ids)) or list(labels)!=ids or set(completed)-set(ids):raise ValueError('Invalid case IDs')
    if digest([labels[k] for k in ids])!=state['identity']['dataset_sha256']:raise ValueError('Label snapshot hash mismatch')
    for k,r in completed.items():
        if r['label']!=labels[k] or r['label']['id']!=k or r['prediction'].get('error') or not isinstance(r['prediction']['hits'],list):raise ValueError('Invalid raw prediction: '+k)
    return state


def grade_file(path):
    path=Path(path);state=load_raw(path);identity=state['identity'];ro=identity['retrieval_only']
    cases=[score_case(state['completed'][k],ro) for k in state['case_ids'] if k in state['completed']]
    review_path=path.with_name('expert_reviews.json')
    if not ro and review_path.exists():
        reviews=json.loads(review_path.read_text(encoding='utf-8'))
        for case in cases:
            entry=reviews.get(case['id'],{})
            if entry.get('prediction_sha256')==digest(state['completed'][case['id']]) and entry.get('label') in ('Đúng','Sai','Không áp dụng'):
                case['expert_label']=entry['label']
    n=len(cases);total=len(state['case_ids']);complete=n==total
    by_role={r:score_summary([x for x in cases if x['actor_role']==r],ro) for r in ('user','staff') if any(x['actor_role']==r for x in cases)}
    by_split={r:score_summary([x for x in cases if x.get('split')==r],ro) for r in sorted({x['split'] for x in cases if x.get('split')})}
    by_type={r:score_summary([x for x in cases if x.get('question_type')==r],ro) for r in sorted({x['question_type'] for x in cases if x.get('question_type')})}
    report={'schema':'rag-ai-score-v2','scorer_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'raw_sha256':digest(state),
            'dataset':state['dataset'],'method':identity['method'],'mode':'retrieval_only' if ro else 'end_to_end',
            'dense_requested':identity['method']!='lexical','chat_model':identity.get('chat_model'),'embedding_model':identity.get('embedding_model'),
            'min_relevance':identity.get('min_relevance'),'identity':identity,'complete':complete,'completed_cases':n,'total_cases':total,
            'pending_ids':[k for k in state['case_ids'] if k not in state['completed']], 'stop_reason':state.get('stop_reason'),
            'failed_attempts':state.get('failure_count',0),'summary_scope':'all_cases' if complete else 'completed_cases_only',
            'summary':score_summary(cases,ro),'by_role':by_role,'by_split':by_split,'by_type':by_type,'cases':cases}
    atomic_json(path.with_name('scores.json'),report);write_csv(path.with_name('scores.csv'),cases)
    return report


def main():
    p=argparse.ArgumentParser(description='Offline grading of saved raw predictions (no Ollama).');p.add_argument('path',type=Path,nargs='?',default=Path('results'));a=p.parse_args()
    paths=[a.path] if a.path.is_file() else sorted(a.path.rglob('predictions.json'))
    if not paths:raise SystemExit('No predictions.json found.')
    for path in paths:
        report=grade_file(path);print(f"{path.parent.name}: {report['completed_cases']}/{report['total_cases']} -> scores.json / scores.csv")

if __name__=='__main__':main()
