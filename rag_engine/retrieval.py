"""Dense candidate retrieval + lexical fusion + MMR for diversity.

Important: cosine similarity and evidence_score are not calibrated answer accuracy.
"""
import re
from collections import Counter
from typing import Any


def tokenize(text: str) -> list[str]:
    return [s.casefold() for s in re.findall(r'[^\W_]+', text or '', flags=re.UNICODE) if len(s) > 1]


def lexical_score(query: str, document: str) -> float:
    """Weighted query-term coverage; rewards exact phrases in Vietnamese text."""
    q = tokenize(query)
    if not q:
        return 0.0
    terms = Counter(q)
    doc_terms = Counter(tokenize(document))
    coverage = sum(min(cnt, doc_terms.get(term, 0)) for term, cnt in terms.items()) / sum(terms.values())
    phrase = ' '.join(q)
    normalized = ' '.join(tokenize(document))
    phrase_bonus = 0.15 if len(q) > 1 and phrase in normalized else 0.0
    return min(1.0, coverage + phrase_bonus)


def evidence_score(similarity: float, lexical: float) -> float:
    """Heuristic relevance index for UI and triage, NOT probability of correctness.

    Mapping is deliberately documented so it can be retuned against labeled data.
    """
    semantic_signal = max(0.0, min(1.0, (similarity - 0.20) / 0.45))
    return round(max(0.0, min(1.0, 0.6 * semantic_signal + 0.4 * lexical)), 4)


def rerank_candidates(query: str, candidates: list[dict[str, Any]], top_k=5) -> list[dict]:
    q_tokens = set(tokenize(query))
    processed = []
    for c in candidates:
        hit = dict(c)
        cosine = max(0.0, min(1.0, float(hit.get('similarity', 0))))
        lexical = lexical_score(query, str(hit.get('text', '')))
        hit['lexical_similarity'] = lexical
        hit['evidence_score'] = evidence_score(cosine, lexical)
        hit['retrieval_score'] = 0.62 * cosine + 0.38 * lexical
        processed.append(hit)
    # MMR penalizes repeated near-duplicate text; preserves distinct relevant pages.
    selected, remaining = [], processed
    while remaining and len(selected) < top_k:
        def score(item):
            t = set(tokenize(item.get('text', '')))
            redundancy = max((len(t & set(tokenize(s.get('text', '')))) / max(1, len(t | set(tokenize(s.get('text', ''))))) for s in selected), default=0.0)
            query_presence = len(t & q_tokens) / max(1, len(q_tokens))
            return 0.80 * item['retrieval_score'] + 0.10 * query_presence - 0.10 * redundancy
        winner = max(range(len(remaining)), key=lambda i: score(remaining[i]))
        selected.append(remaining.pop(winner))
    return selected


def retrieve_context(query: str, top_k: int = 5, min_similarity: float = 0.0) -> list[dict]:
    query = query.strip()
    if not query:
        return []
    if not 1 <= top_k <= 30 or not 0 <= min_similarity <= 1:
        raise ValueError('Tham số truy xuất không hợp lệ.')
    from .database import get_collection
    from .embeddings import encode_query
    collection = get_collection()
    count = collection.count()
    if not count:
        return []
    result = collection.query(query_embeddings=[encode_query(query)],
                              n_results=min(max(8 * top_k, 30), count),
                              include=['documents', 'metadatas', 'distances'])
    candidates = []
    for doc, meta, distance in zip(result.get('documents', [[]])[0],
                                   result.get('metadatas', [[]])[0],
                                   result.get('distances', [[]])[0]):
        cosine = max(0.0, min(1.0, 1.0 - float(distance)))
        if cosine < min_similarity:
            continue
        meta = meta or {}
        candidates.append({'text': doc or '', 'source': meta.get('source', 'Không rõ'),
                           'source_id': meta.get('source_id', ''),
                           'page': meta.get('page', 0), 'chunk': meta.get('chunk', 0),
                           'similarity': cosine})
    return rerank_candidates(query, candidates, top_k)
