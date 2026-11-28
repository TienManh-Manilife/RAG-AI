"""Offline pairwise comparisons; threshold sweep is retrieval triage, not end-to-end."""
from __future__ import annotations
import argparse,itertools,math,json
from pathlib import Path
from .benchmark_score import load_raw,digest,write_csv

def compare_pair(left,right,scope):
    a,b=left['identity'],right['identity']
    for k in ('dataset_sha256','corpus_sha256','source_sha256','retrieval_only'):
        if a[k]!=b[k]:raise ValueError('Incompatible '+k)
    for k,v in a['model_digests'].items():
        if k in b['model_digests'] and b['model_digests'][k]!=v:raise ValueError('Different model digest '+k)
    if a['embedding_sha256'] and b['embedding_sha256'] and a['embedding_sha256']!=b['embedding_sha256']:raise ValueError('Different embeddings')
    if left['scorer_sha256']!=right['scorer_sha256']:raise ValueError('Different scorer versions; regrade first')
    x={r['id']:r for r in left['cases'] if scope=='all' or r.get('split')==scope};y={r['id']:r for r in right['cases'] if scope=='all' or r.get('split')==scope}
    if set(x)!=set(y):raise ValueError('Different case IDs')
    checks={'recall_at_1':lambda r:not r['should_handoff'] and r.get('first_relevant_rank')==1,
            'recall_at_3':lambda r:0<(r.get('first_relevant_rank') or 0)<=3,
            'recall_at_5':lambda r:0<(r.get('first_relevant_rank') or 0)<=5}
    if not a['retrieval_only']:checks.update(handoff_correct=lambda r:r['pred_handoff']==r['should_handoff'],checklist=lambda r:r['checklist_pass'] is True,overall_pass=lambda r:r['overall_pass'] is True)
    rows=[]
    for metric,fn in checks.items():
        ids=[k for k,r in x.items() if not (metric.startswith('recall') or metric=='checklist') or not r['should_handoff']]
        if not ids:continue
        pairs=[(fn(x[k]),fn(y[k])) for k in ids];left_only=sum(l and not r for l,r in pairs);right_only=sum(r and not l for l,r in pairs);n=len(ids);d=left_only+right_only
        p=min(1.,2*sum(math.comb(d,j)*(.5**d) for j in range(min(left_only,right_only)+1)))
        rows.append(dict(scope=scope,metric=metric,n=n,left_rate=sum(l for l,r in pairs)/n,right_rate=sum(r for l,r in pairs)/n,delta_right_minus_left=(right_only-left_only)/n,both_pass=sum(l and r for l,r in pairs),left_only=left_only,right_only=right_only,mcnemar_exact_p=p))
    return rows

def threshold_rows(state):
    # Evidence-only triage on retrieval outputs, ignoring LLM/personal guards.
    cases=list(state['completed'].values());out=[]
    for split in ('legacy50','new150','all'):
        rs=[r for r in cases if split=='all' or r['label'].get('split')==split]
        if not rs:continue
        for t in (.20,.24,.30,.38,.45,.50,.60,.70):
            flags=[(not r['prediction']['hits'] or r['prediction']['evidence_score']<t,r['label']['should_handoff']) for r in rs]
            tp=sum(p and g for p,g in flags);fp=sum(p and not g for p,g in flags);fn=sum(not p and g for p,g in flags);tn=sum(not p and not g for p,g in flags)
            out.append(dict(split=split,threshold=t,cases=len(rs),tp=tp,fp=fp,fn=fn,tn=tn,triage_f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else None,diagnostic='retrieval evidence only; not actual handoff'))
    return out

def main():
    p=argparse.ArgumentParser();p.add_argument('root',type=Path,nargs='?',default=Path('results/v200'));a=p.parse_args()
    reports=[];sweep=[]
    for path in sorted(a.root.glob('*/scores.json')):
        r=json.loads(path.read_text(encoding='utf-8'));state=load_raw(path.with_name('predictions.json'))
        if not r['complete'] or digest(state)!=r['raw_sha256']:print('SKIP incomplete/stale',path.parent.name);continue
        reports.append((path.parent.name,r))
        if r['mode']=='retrieval_only':sweep.extend(dict(run=path.parent.name,method=r['method'],**row) for row in threshold_rows(state))
    comparisons=[]
    for (ln,l),(rn,r) in itertools.combinations(reports,2):
        if l['mode']!=r['mode']:continue
        try:
            for scope in ('all','legacy50','new150'):
                comparisons.extend(dict(left=ln,right=rn,mode=l['mode'],**row) for row in compare_pair(l,r,scope))
        except ValueError as e:print('SKIP',ln,rn,str(e))
    write_csv(a.root/'paired_comparison.csv',comparisons);write_csv(a.root/'evidence_threshold_diagnostic.csv',sweep)
    print('Saved paired comparisons and evidence-only threshold diagnostics. No model calls.')
if __name__=='__main__':main()
