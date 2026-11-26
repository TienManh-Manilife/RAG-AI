"""Persistent, fail-fast CLI benchmark. Checkpoints contain JSON in UTF-8 TXT."""
from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from .config import DATA_DIR, ROOT, CHAT_MODEL, EMBED_MODEL, RAG_MIN_RELEVANCE, OLLAMA_URL


def _now():
    return datetime.now(timezone.utc).isoformat()


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode('utf-8')).hexdigest()


def _atomic_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix='.'+path.name+'.', suffix='.tmp', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(obj, f, ensure_ascii=False, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


@contextmanager
def _lock(path):
    # Runs inside the Linux app container. OS releases the lock after a crash.
    import fcntl
    with Path(path).open('a+b') as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another benchmark is using this checkpoint.') from exc
        try:
            yield
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def _model_identity(retrieval_only, use_dense):
    required = ([CHAT_MODEL] if not retrieval_only else []) + ([EMBED_MODEL] if use_dense else [])
    if not required:
        return {}
    import requests
    response = requests.get(OLLAMA_URL+'/api/tags', timeout=10)
    response.raise_for_status()
    models = {m.get('name'):m.get('digest') for m in response.json()['models']}
    out = {}
    for name in required:
        canonical = name if ':' in name else name+':latest'
        digest = models.get(name) or models.get(canonical)
        if not digest:
            raise RuntimeError(f'Model unavailable or missing digest: {name}. Check Ollama before resuming.')
        out[name] = digest
    return out


def _identity(rows, retrieval_only, use_dense):
    from .documents import authorized_chunks
    corpus = authorized_chunks({'role':'staff'})
    if use_dense:
        missing = [r['id'] for r in corpus if not r['embedding_json'] or r['embedding_model'] != EMBED_MODEL]
        if missing:
            raise RuntimeError(f'{len(missing)} chunks missing current embeddings. Refresh embeddings before hybrid benchmark.')
    # Include actual chunk content, access permissions and, for hybrid, stored vectors.
    snapshot = [{k:v for k,v in r.items() if use_dense or k not in ('embedding_json','embedding_model')} for r in corpus]
    source_files = ('benchmark.py','benchmark_checkpoint.py','retrieval.py','pipeline.py','generation.py','handoff.py','ollama.py','config.py','documents.py')
    source_hashes = {name:hashlib.sha256((ROOT/'rag_engine'/name).read_bytes()).hexdigest() for name in source_files}
    return {'dataset_sha256':_digest(rows), 'corpus_sha256':_digest(snapshot),
            'source_sha256':_digest(source_hashes), 'dense_requested':use_dense,
            'retrieval_only':retrieval_only, 'chat_model':CHAT_MODEL, 'embedding_model':EMBED_MODEL,
            'model_digests':_model_identity(retrieval_only,use_dense), 'min_relevance':RAG_MIN_RELEVANCE}


def run_resumable_benchmark(rows, *, retrieval_only=False, use_dense=True,
                            role='all', run_id='default', max_new=None,
                            report_dir=None, progress=None, announce=print):
    """Keep every valid execution (including wrong answers); retry only infrastructure failures."""
    from . import benchmark as b
    from .retrieval import strict_dense_benchmark
    if not rows or len({r['id'] for r in rows}) != len(rows):
        raise ValueError('Empty or duplicate benchmark IDs.')
    if not re.fullmatch(r'[a-zA-Z0-9_-]{1,64}',run_id) or role not in ('all','user','staff'):
        raise ValueError('Invalid run ID or role.')
    if max_new is not None and max_new < 1:
        raise ValueError('--max-new must be positive.')
    mode='retrieval_only' if retrieval_only else 'end_to_end'
    method='hybrid' if use_dense else 'lexical'
    root=Path(report_dir or DATA_DIR/'reports')
    dest=root/f'{mode}_{method}_{role}_{run_id}'
    dest.mkdir(parents=True,exist_ok=True)
    path=dest/'checkpoint.txt'
    with _lock(dest/'checkpoint.lock'):
        identity=_identity(rows,retrieval_only,use_dense)
        ids=[r['id'] for r in rows]
        if path.exists():
            state=json.loads(path.read_text(encoding='utf-8'))
            if state.get('schema')!=1 or state.get('identity')!=identity or state.get('case_ids')!=ids:
                raise RuntimeError('Checkpoint configuration/data/model changed. Use a new --run-id; old results are preserved.')
        else:
            state={'schema':1,'identity':identity,'case_ids':ids,'created_at':_now(),
                   'completed':{},'last_failure':None,'failure_count':0}
            _atomic_json(path,state)
        completed=state['completed']
        if not isinstance(completed,dict) or any(k not in ids or r.get('id')!=k or 'system_error' in r.get('errors',[]) or r.get('error') for k,r in completed.items()):
            raise RuntimeError('Invalid checkpoint contents. Do not edit checkpoint.txt manually.')
        new=0; skipped=len(completed); stopped=''; failed=None
        announce(f'Checkpoint: {path} | completed {skipped}/{len(rows)}')
        if progress:progress(skipped,len(rows))
        for item in rows:
            key=item['id']
            if key in completed:
                announce(f'SKIP {key} (already completed)')
                continue
            if max_new is not None and new>=max_new:
                stopped='batch_limit'; break
            announce(f'RUN {key} [{len(completed)+1}/{len(rows)}]')
            # Use exactly the existing per-case evaluation; only disallow silent dense fallback.
            with strict_dense_benchmark(use_dense):
                row=b.run_benchmark([item],retrieval_only=retrieval_only,use_dense=use_dense)['cases'][0]
            if row.get('error') or 'system_error' in row.get('errors',[]):
                failed=key; stopped='system_error'
                state['last_failure']={'id':key,'error':row.get('error'),'at':_now()}
                state['failure_count']+=1
                _atomic_json(path,state)
                announce(f'STOP {key}: {row.get("error") or "system_error"}. This case will be retried.')
                break
            completed[key]=row
            _atomic_json(path,state)
            new+=1
            announce(f'SAVED {key} [{len(completed)}/{len(rows)}]')
            if progress:progress(len(completed),len(rows))
        output=[completed[k] for k in ids if k in completed]
        complete=len(output)==len(rows)
        report={'timestamp':_now(),'dataset':'anphat_pdf_v2','chat_model':CHAT_MODEL,
                'embedding_model':EMBED_MODEL,'dense_requested':use_dense,'min_relevance':RAG_MIN_RELEVANCE,
                'mode':mode,'duration_s':round(sum(r.get('latency_s',0) for r in output),3),
                'duration_note':'Sum of completed case latencies across sessions, not wall clock time.',
                'complete':complete,'total_cases':len(rows),'completed_cases':len(output),
                'pending_ids':[k for k in ids if k not in completed], 'skipped_this_run':skipped,
                'new_this_run':new,'stop_reason':stopped or ('complete' if complete else 'partial'),
                'failed_case':failed,'failed_attempts':state['failure_count'],'last_failure':state['last_failure'],
                'identity':identity,'summary_scope':'all_cases' if complete else 'completed_cases_only',
                'summary':b.summarize(output,retrieval_only=retrieval_only),
                'by_role':{r:b.summarize([x for x in output if x['actor_role']==r],retrieval_only=retrieval_only)
                           for r in ('user','staff') if any(x['actor_role']==r for x in output)},'cases':output}
        b.write_report(report,dest)
        # Keep the legacy report location for the Streamlit viewer / existing workflows.
        b.write_report(report,root)
        return report,dest
