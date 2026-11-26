"""Hybrid BM25 (Vietnamese normalized) + Ollama bge-m3 dense retrieval.
All candidate filtering, including staff-only sources, happens BEFORE similarity scoring.
Relevance index is NOT a calibrated probability of answer correctness.
"""
from __future__ import annotations
import json
import math
import re
import unicodedata
from collections import Counter
from .documents import authorized_chunks
from .config import EMBED_MODEL

STOPWORDS = set('là có được của cho với các những một về tôi bạn công ty ở tại như nào nhiêu khi nếu từ để thì và hoặc hay theo trong không muốn cần việc hỏi thế nào có thể giúp làm'.split())
SYNONYMS = {'ship':'giao', 'shipper':'giao', 'order':'đơn', 'refund':'hoàn', 'hủy':'huỷ',
            'đổi':'đổi', 'lỗi':'lỗi', 'bill':'hóa', 'hoá':'hóa', 'invoice':'hóa',
            'bảo':'bảo', 'hành':'hành', 'ticket':'yêu', 'hotline':'điện', 'feedback':'khiếu'}

def fold(text):
    text = (text or '').casefold().replace('đ','d')
    return ''.join(c for c in unicodedata.normalize('NFD', text) if not unicodedata.combining(c))

def tokenize(text):
    words = re.findall(r'[a-z0-9]+', fold(text))
    return [SYNONYMS.get(t,t) for t in words if t not in STOPWORDS and len(t) > 1]

def cosine(a,b):
    if not a or not b or len(a)!=len(b): return 0.0
    dot=sum(x*y for x,y in zip(a,b))
    na=math.sqrt(sum(x*x for x in a))
    nb=math.sqrt(sum(y*y for y in b))
    return dot/(na*nb) if na*nb else 0.0

def _lexical(query, texts):
    docs = [Counter(tokenize(t)) for t in texts]
    q = tokenize(query)
    if not q: return [(0.0,0.0)]*len(texts)
    n=len(docs)
    df=Counter(term for d in docs for term in d)
    idf={t:math.log(1+(n-df.get(t,0)+0.5)/(df.get(t,0)+0.5)) for t in set(q)}
    weights=sum(idf.values()) or 1
    avglen=sum(sum(d.values()) for d in docs)/max(n,1)
    scores=[]
    for doc in docs:
        dl=sum(doc.values())
        bm25=0
        covered=0
        for term in set(q):
            f=doc[term]
            if f:
                covered+=idf[term]
                bm25+=idf[term]*f*2.2/(f+1.2*(0.25+0.75*dl/max(avglen,1)))
        # full query coverage is a reproducible interpretable signal
        scores.append((bm25, covered/weights))
    return scores

def retrieve_context(query, actor=None, top_k=5, use_dense=True):
    query = (query or '').strip()
    if not query or not (1 <= top_k <= 20): return []
    rows = authorized_chunks(actor)
    if not rows: return []
    lexical = _lexical(query, [r['title']+' '+r['section']+' '+r['content'] for r in rows])
    ranks_lex = sorted(range(len(rows)), key=lambda i: lexical[i][0], reverse=True)
    dense = [0.0]*len(rows)
    has_dense = False
    if use_dense and any(r['embedding_json'] and r['embedding_model'] == EMBED_MODEL for r in rows):
        try:
            from .ollama import embed_batch
            vec = embed_batch([query])[0]
            for i,r in enumerate(rows):
                if r['embedding_json'] and r['embedding_model']==EMBED_MODEL:
                    dense[i]=max(0.0, cosine(vec, json.loads(r['embedding_json'])))
            has_dense=True
        except Exception:
            # Degrade to lexical retrieval; generation will independently report Ollama errors.
            pass
    ranks_dense = sorted(range(len(rows)), key=lambda i: dense[i], reverse=True)
    lexrank = {idx:r for r,idx in enumerate(ranks_lex,1)}
    denserank = {idx:r for r,idx in enumerate(ranks_dense,1)}
    results=[]
    for i,row in enumerate(rows):
        bm25, coverage = lexical[i]
        if not bm25 and (not has_dense or dense[i] < 0.45): continue
        rrf=1/(50+lexrank[i])+(1/(50+denserank[i]) if has_dense else 0)
        # The weights are heuristic for ranking/triage only, never claim calibrated accuracy.
        semantic=max(0.0,min(1.0,(dense[i]-0.30)/0.43)) if has_dense else 0.0
        relevance=min(1.0,0.55*coverage+0.45*semantic) if has_dense else coverage
        results.append({
            'text': row['content'], 'source': row['filename'], 'title':row['title'],
            'section':row['section'], 'page':row['page'], 'chunk_id':row['id'],
            'similarity':round(dense[i],4) if has_dense else None,
            'lexical_coverage':round(coverage,4), 'evidence_score':round(relevance,4),
            '_score':rrf+0.03*relevance, 'audience':row['audience']
        })
    results.sort(key=lambda r:r['_score'],reverse=True)
    for item in results: item.pop('_score',None)
    return results[:top_k]
