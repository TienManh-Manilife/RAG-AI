"""Plot completed V200 scores offline. No Ollama, Docker or database calls."""
from __future__ import annotations
import argparse, hashlib, json, math
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

METHODS = ['lexical','dense','standard_hybrid','hybrid','no_gate']
LABEL = dict(zip(METHODS,['Lexical','Dense-only','Standard Hybrid','EH-RAG','EH-RAG w/o Gate']))
COLORS = dict(zip(METHODS,['#64748b','#d97706','#7c3aed','#087f8c','#be6470']))
MODES = ['retrieval_only','end_to_end']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':120,'savefig.dpi':300})

def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def digest(x): return hashlib.sha256(json.dumps(x,ensure_ascii=False,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def load(root, run_id):
    reports={}
    for p in sorted(root.glob('*/scores.json')):
        r=read(p); m=r.get('method'); mode=r.get('mode')
        if m not in METHODS or mode not in MODES or p.parent.name!=f'{mode}_{m}_all_{run_id}': continue
        if not r.get('complete') or r.get('completed_cases')!=r.get('total_cases'):
            print('SKIP incomplete:',p.parent.name);continue
        raw=read(p.with_name('predictions.json'))
        if digest(raw)!=r['raw_sha256'] or raw['identity']!=r['identity']:
            raise SystemExit(f'Stale scores: {p}. Regrade first.')
        if len(r['cases'])!=r['total_cases'] or len({c['id'] for c in r['cases']})!=r['total_cases'] or set(raw['completed'])!={c['id'] for c in r['cases']}:
            raise SystemExit(f'Invalid case IDs: {p}')
        reports[(mode,m)]=(r,raw)
    if not reports: raise SystemExit('No completed all-role scores for this run ID. Run benchmark_score first.')
    first=next(iter(reports.values()))[0]
    for r,_ in reports.values():
        for key in ['dataset_sha256','corpus_sha256','source_sha256']:
            if r['identity'].get(key)!=first['identity'].get(key): raise SystemExit(f'Incompatible {key}; do not mix experiments.')
        if r['scorer_sha256']!=first['scorer_sha256']: raise SystemExit('Different scorer versions. Regrade all first.')
    model_ids={}; embeddings=set()
    for r,_ in reports.values():
        for k,v in r['identity'].get('model_digests',{}).items():
            if k in model_ids and model_ids[k]!=v: raise SystemExit('Different model digest: '+k)
            model_ids[k]=v
        if r['identity'].get('embedding_sha256'): embeddings.add(r['identity']['embedding_sha256'])
    if len(embeddings)>1: raise SystemExit('Different embeddings.')
    return reports

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path,nargs='?',default=Path('results/v200'));p.add_argument('--run-id',default='default');p.add_argument('--output',type=Path);a=p.parse_args()
    reports=load(a.root,a.run_id);out=a.output or a.root/'figures'/a.run_id;out.mkdir(parents=True,exist_ok=True)
    generated=[]
    def save(fig,name):
        fig.tight_layout();fig.savefig(out/(name+'.jpg'),bbox_inches='tight');fig.savefig(out/(name+'.svg'),bbox_inches='tight');plt.close(fig);generated.append(name)
    def rows(mode):return [(m,reports[(mode,m)][0]) for m in METHODS if (mode,m) in reports]
    def group(mode, metrics, name, title, scale=100, unit='%', scope=None):
        rs=rows(mode)
        if not rs:return
        x=np.arange(len(metrics));w=.78/len(rs);fig,ax=plt.subplots(figsize=(max(8,len(metrics)*1.9),5.1))
        vals=[]
        for i,(m,r) in enumerate(rs):
            s=r['summary'] if scope is None else r.get(scope[0],{}).get(scope[1],{})
            v=[float(s[k])*scale if s.get(k) is not None else np.nan for k,_ in metrics];vals.extend(v)
            bars=ax.bar(x+(i-(len(rs)-1)/2)*w,v,w*.94,label=LABEL[m],color=COLORS[m]);ax.bar_label(bars,labels=[f'{v:.1f}' if scale==100 and np.isfinite(v) else f'{v:.3f}' if np.isfinite(v) else '' for v in v],padding=2,fontsize=7,rotation=90 if len(rs)>3 else 0)
        if not any(np.isfinite(vals)):plt.close(fig);return
        ax.set_xticks(x,[t for _,t in metrics]);ax.set_ylabel(unit);ax.set_title(title);ax.grid(axis='y',alpha=.22);ax.set_axisbelow(True);ax.legend(loc='upper center',bbox_to_anchor=(.5,-.13),ncol=min(3,len(rs)),frameon=False);ax.set_ylim(0,120 if scale==100 else max(v for v in vals if np.isfinite(v))*1.25 or 1);save(fig,name)
    retrieval=[('retrieval_recall_at_1','Recall@1'),('retrieval_recall_at_3','Recall@3'),('retrieval_recall_at_5','Recall@5'),('mrr_at_5','MRR@5 × 100')]
    generation=[('answer_accuracy_checklist_proxy','Checklist\naccuracy'),('auto_test_pass_rate_proxy','Overall\npass'),('citation_rate_on_answered','Citation\nrate'),('answerable_completion_rate','Completion'),('keyphrase_coverage_proxy','Keyphrase\ncoverage')]
    handoff=[('handoff_precision','Precision'),('handoff_recall','Recall'),('handoff_f1','F1'),('handoff_accuracy','Accuracy'),('handoff_specificity','Specificity')]
    latency=[('mean_latency_s','Mean'),('median_latency_s','Median'),('p95_latency_s','P95')]
    for mode in MODES:
        group(mode,retrieval,mode+'_retrieval',mode.replace('_',' ').title()+': retrieval quality')
        group(mode,latency,mode+'_latency',mode.replace('_',' ').title()+': latency (lower is better)',1,'Seconds')
        rs=rows(mode)
        if not rs:continue
        fig,ax=plt.subplots(figsize=(8,5))
        for m,r in rs:
            v=np.sort([c['latency_s'] for c in r['cases'] if c.get('latency_s') is not None]);
            if len(v):ax.step(v,np.arange(1,len(v)+1)/len(v)*100,where='post',label=LABEL[m],color=COLORS[m])
        ax.set(xlabel='Latency (seconds)',ylabel='Cumulative cases (%)',title=mode.replace('_',' ').title()+': latency ECDF');ax.grid(alpha=.2);ax.legend();save(fig,mode+'_latency_ecdf')
        fig,ax=plt.subplots(figsize=(8,5))
        for m,r in rs:
            s=r['summary'];key='retrieval_recall_at_1' if mode=='retrieval_only' else 'handoff_f1'
            if s.get(key) is None or s.get('mean_latency_s') is None:continue
            x,y=s['mean_latency_s'],s[key]*100;ax.scatter(x,y,s=85,color=COLORS[m],label=LABEL[m]);ax.annotate(LABEL[m],(x,y),xytext=(6,6),textcoords='offset points',fontsize=8)
        ax.set(xlabel='Mean latency (seconds)',ylabel='Recall@1 (%)' if mode=='retrieval_only' else 'Handoff F1 (%)',title='Quality–latency trade-off');ax.margins(.18);ax.grid(alpha=.22);save(fig,mode+'_tradeoff')
    group('end_to_end',generation,'end_to_end_generation','End-to-end: automated answer quality (proxy)')
    group('end_to_end',handoff,'end_to_end_handoff','End-to-end: handoff quality')
    group('end_to_end',[('false_handoff_rate_on_answerable','False handoff'),('false_answer_rate_on_insufficient','Missed handoff')],'end_to_end_routing_errors','Routing error rates (lower is better)')
    rs=rows('end_to_end')
    if rs:
        fig,axes=plt.subplots(1,len(rs),figsize=(3.15*len(rs),3.5),squeeze=False)
        for ax,(m,r) in zip(axes[0],rs):
            c=r['summary']['confusion'];mat=np.array([[c['tp'],c['fn']],[c['fp'],c['tn']]]);ax.imshow(mat,cmap='Blues',vmin=0,vmax=max(1,max(max(x['summary']['confusion'].values()) for _,x in rs)))
            for i,j in np.ndindex(2,2):ax.text(j,i,str(mat[i,j]),ha='center',va='center',color='white' if mat[i,j]>max(mat.flatten())*.55 else 'black',fontsize=14)
            ax.set_xticks([0,1],['Handoff','AI']);ax.set_yticks([0,1],['Handoff','Answerable']);ax.set_title(LABEL[m]);ax.set_xlabel('Predicted');ax.set_ylabel('Ground truth')
        fig.suptitle('Handoff confusion matrices');save(fig,'end_to_end_confusion')
        errors=sorted({e for _,r in rs for e in r['summary']['error_counts']})
        if errors:
            fig,ax=plt.subplots(figsize=(max(9,len(errors)*1.55),5));x=np.arange(len(errors));w=.8/len(rs)
            for i,(m,r) in enumerate(rs):
                v=[r['summary']['error_counts'].get(e,0) for e in errors];b=ax.bar(x+(i-(len(rs)-1)/2)*w,v,w*.93,label=LABEL[m],color=COLORS[m]);ax.bar_label(b,fontsize=8,padding=2)
            ax.set_xticks(x,[e.replace('_','\n') for e in errors]);ax.set_ylabel('Cases');ax.set_title('Error counts (cases can have multiple errors)');ax.legend(ncol=2);ax.margins(y=.2);save(fig,'end_to_end_errors')
        for field,keys in [('by_role',['user','staff']),('by_split',['legacy50','new150'])]:
            for k in keys:group('end_to_end',[retrieval[0],handoff[2],generation[0],generation[1]],field+'_'+k,'End-to-end: '+k,scope=(field,k))
        # Paired binary outcomes, use the exact same labeled cases; no independent-sample assumption.
        if ('end_to_end','hybrid') in reports:
            h={c['id']:c for c in reports[('end_to_end','hybrid')][0]['cases']};other=[(m,r) for m,r in rs if m!='hybrid'];
            if other:
                fig,ax=plt.subplots(figsize=(9,4.8));x=np.arange(len(other));w=.24
                for j,(metric,fn) in enumerate([('Recall@1',lambda c:c.get('first_relevant_rank')==1),('Handoff accuracy',lambda c:c['pred_handoff']==c['should_handoff']),('Overall pass',lambda c:c['overall_pass'] is True)]):
                    v=[]
                    for m,r in other:
                        b={c['id']:c for c in r['cases']};ids=[k for k,c in h.items() if k in b and (metric!='Recall@1' or not c['should_handoff'])];v.append(100*sum(int(fn(h[k]))-int(fn(b[k])) for k in ids)/len(ids) if ids else np.nan)
                    bars=ax.bar(x+(j-1)*w,v,w,label=metric);ax.bar_label(bars,fmt='%.1f',fontsize=8,padding=3)
                ax.axhline(0,color='#64748b',lw=1);ax.set_xticks(x,[LABEL[m] for m,_ in other]);ax.set_ylabel('EH-RAG minus comparator (pp)');ax.set_title('Paired differences (descriptive, not significance)');ax.legend();ax.margins(y=.25);save(fig,'end_to_end_paired_delta')
    rs=rows('retrieval_only')
    if rs:
        fig,ax=plt.subplots(figsize=(8,5));ts=np.array([.20,.24,.30,.38,.45,.50,.60,.70])
        for m,r in rs:
            raw=reports[('retrieval_only',m)][1];cs=list(raw['completed'].values());vals=[]
            for t in ts:
                pairs=[(not c['prediction']['hits'] or c['prediction']['evidence_score']<t,c['label']['should_handoff']) for c in cs];tp=sum(p and g for p,g in pairs);fp=sum(p and not g for p,g in pairs);fn=sum(not p and g for p,g in pairs);vals.append(100*2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else np.nan)
            ax.plot(ts,vals,marker='o',label=LABEL[m],color=COLORS[m])
        ax.axvline(.38,ls='--',color='#777',label='Current threshold 0.38');ax.set(xlabel='Evidence threshold',ylabel='Evidence-only triage F1 (%)',title='Threshold sensitivity (diagnostic, not actual E2E handoff)');ax.legend();ax.grid(alpha=.2);save(fig,'evidence_threshold_diagnostic')
        fig,axes=plt.subplots(1,len(rs),figsize=(3.5*len(rs),4),squeeze=False)
        for ax,(m,r) in zip(axes[0],rs):
            cs=r['cases'];a1=[c['evidence_score'] for c in cs if not c['should_handoff']];a2=[c['evidence_score'] for c in cs if c['should_handoff']];ax.hist([a1,a2],bins=np.linspace(0,1,11),label=['Answerable','Needs handoff'],color=['#087f8c','#d97706'],alpha=.8);ax.axvline(.38,ls='--',color='#333');ax.set(title=LABEL[m],xlabel='Evidence score',ylabel='Cases');ax.legend(fontsize=8)
        fig.suptitle('Evidence score distribution by ground-truth label (retrieval only)');save(fig,'evidence_distribution')
    (out/'manifest.json').write_text(json.dumps({'run_id':a.run_id,'runs':[f'{mode}_{m}' for mode,m in reports],'figures':generated},indent=2),encoding='utf-8')
    print(f'Saved {len(generated)} figures (JPG + SVG) to {out.resolve()}');missing=[f'{mode}_{m}' for mode in MODES for m in METHODS if not(mode=='retrieval_only' and m=='no_gate') and (mode,m) not in reports];
    if missing:print('Missing configurations (not plotted):',', '.join(missing))
if __name__=='__main__':main()
