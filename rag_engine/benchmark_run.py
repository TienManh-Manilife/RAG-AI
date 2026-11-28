"""Resumable raw-inference benchmark; scoring runs only after inference.
The scorer can be invoked independently on a host without Docker or model access.
"""
from __future__ import annotations
import argparse,hashlib,json,re
from pathlib import Path
from datetime import datetime,timezone
from .config import DATA_DIR,ROOT,CHAT_MODEL,EMBED_MODEL,RAG_MIN_RELEVANCE
from .benchmark_checkpoint import _lock,_model_identity
from .benchmark_score import atomic_json,digest,load_raw,grade_file,write_csv
from .benchmark_inference import METHODS,predict_case

def now():return datetime.now(timezone.utc).isoformat()

def identity(rows,method,ro):
    from .documents import authorized_chunks
    corpus=authorized_chunks({'role':'staff'});dense=method!='lexical'
    if dense and any(not r['embedding_json'] or r['embedding_model']!=EMBED_MODEL for r in corpus):
        raise RuntimeError('Missing current document embeddings; reindex before benchmark.')
    # Same text fingerprint for ALL methods; vectors have a separate fingerprint.
    text=[{k:v for k,v in r.items() if k not in ('embedding_json','embedding_model')} for r in corpus]
    vector=[{k:r[k] for k in ('id','embedding_json','embedding_model')} for r in corpus] if dense else None
    files=('benchmark_inference.py','benchmark_run.py','retrieval.py','pipeline.py','generation.py','handoff.py','ollama.py','config.py','documents.py','auth.py')
    source={f:hashlib.sha256((ROOT/'rag_engine'/f).read_bytes()).hexdigest() for f in files}
    return dict(dataset_sha256=digest(rows),corpus_sha256=digest(text),embedding_sha256=digest(vector) if dense else None,
        source_sha256=digest(source),method=method,retrieval_only=ro,chat_model=CHAT_MODEL,embedding_model=EMBED_MODEL,
        model_digests=_model_identity(ro,dense),min_relevance=0. if method=='no_gate' else RAG_MIN_RELEVANCE,
        generation_gate=0. if method=='no_gate' else .24,top_k=5)


def export_predictions(path,state):
    # Export the model output with its labels. The grader never writes this file.
    write_csv(path.with_name('predictions.csv'),[{**x['label'], 'prediction':x['prediction']['answer'],
       'pred_handoff':x['prediction']['handoff'],'evidence_score':x['prediction']['evidence_score'],'hits':x['prediction']['hits'],
       'meta':x['prediction']['meta'],'reason':x['prediction']['reason'],'error':x['prediction']['error'],'latency_s':x['latency_s']}
       for k in state['case_ids'] if (x:=state['completed'].get(k)) is not None])


def run(rows,*,method='hybrid',retrieval_only=False,role='all',run_id='default',dataset='200',max_new=None,report_dir=None,announce=print):
    if not rows or len({r['id'] for r in rows})!=len(rows):raise ValueError('Empty or duplicate cases')
    if method not in METHODS or (retrieval_only and method=='no_gate'):raise ValueError('Invalid method/mode')
    if role not in ('all','user','staff') or not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',run_id):raise ValueError('Invalid run ID or role')
    if max_new is not None and max_new<1:raise ValueError('--max-new must be positive')
    mode='retrieval_only' if retrieval_only else 'end_to_end'
    root=Path(report_dir or DATA_DIR/'reports'/f'v{dataset}')
    dest=root/f'{mode}_{method}_{role}_{run_id}';dest.mkdir(parents=True,exist_ok=True)
    path=dest/'predictions.json'
    with _lock(dest/'predictions.lock'):
        ident=identity(rows,method,retrieval_only); ids=[r['id'] for r in rows]
        if path.exists():
            state=load_raw(path)
            if state['identity']!=ident or state['case_ids']!=ids:raise RuntimeError('Raw checkpoint identity changed; use a new --run-id. Existing predictions preserved.')
        else:
            state=dict(schema='rag-ai-raw-v2',dataset=f'anphat_pdf_v{dataset}',identity=ident,case_ids=ids,
                       labels={r['id']:r for r in rows},completed={},created_at=now(),failure_count=0,last_failure=None,stop_reason='partial')
            atomic_json(path,state)
        completed=state['completed'];skipped=len(completed);new=0
        announce(f'{dest} | completed {skipped}/{len(ids)}')
        try:
            for item in rows:
                key=item['id']
                if key in completed:continue
                if max_new is not None and new>=max_new:state['stop_reason']='batch_limit';break
                announce(f'RUN {key} [{len(completed)+1}/{len(ids)}]')
                raw=predict_case(item,method,retrieval_only);pred=raw['prediction']
                if pred.get('error'):
                    state['failure_count']+=1;state['last_failure']=dict(id=key,error=pred['error'],at=now(),latency_s=raw['latency_s']);state['stop_reason']='system_error';atomic_json(path,state)
                    announce(f'STOP {key}: {pred["error"]}; retry same command.');break
                completed[key]=raw;new+=1;state['stop_reason']='partial';atomic_json(path,state)
                announce(f'SAVED {key} [{len(completed)}/{len(ids)}]')
            if len(completed)==len(ids):state['stop_reason']='complete'
        except KeyboardInterrupt:
            state['stop_reason']='interrupted';atomic_json(path,state);raise
        finally:
            # Only raw data is checkpointed; a grading error cannot erase it.
            atomic_json(path,state)
        export_predictions(path,state)
        report=grade_file(path)
    return report,dest


def main():
    p=argparse.ArgumentParser(description='Benchmark inference + offline scoring; default 200 labeled cases')
    p.add_argument('--retrieval-only',action='store_true');p.add_argument('--method',choices=METHODS,default=None)
    p.add_argument('--lexical-only',action='store_true',help='Compatibility alias for --method lexical')
    p.add_argument('--dataset',choices=('50','200'),default='200');p.add_argument('--role',choices=('all','user','staff'),default='all')
    p.add_argument('--run-id',default='default');p.add_argument('--max-new',type=int)
    a=p.parse_args();method=a.method or ('lexical' if a.lexical_only else 'hybrid')
    if a.lexical_only and method!='lexical':p.error('--lexical-only conflicts with --method')
    if a.retrieval_only and method=='no_gate':p.error('no_gate is an end-to-end ablation only')
    from .auth import init_db
    from .documents import init_docs,seed_documents
    from .benchmark import read_cases,validate_corpus,FIXTURE,FIXTURE_200
    init_db();init_docs();seed_documents();rows=read_cases(FIXTURE_200 if a.dataset=='200' else FIXTURE);validate_corpus(rows)
    if a.role!='all':rows=[r for r in rows if r['actor_role']==a.role]
    report,dest=run(rows,method=method,retrieval_only=a.retrieval_only,role=a.role,run_id=a.run_id,dataset=a.dataset,max_new=a.max_new,announce=lambda msg:print(msg,flush=True))
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2));print(f"Completed {report['completed_cases']}/{report['total_cases']} | {report['stop_reason']} | {dest}")
    if report['stop_reason']=='system_error':raise SystemExit(2)

if __name__=='__main__':main()
