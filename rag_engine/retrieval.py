"""Dense retrieval, lexical reranking, and MMR-style diversity selection."""
import re
from typing import Any
from .database import get_collection
from .embeddings import encode_query

def _tokenize_for_matching(text: str) -> set[str]:
    return {
        token.casefold()
        for token in re.findall(r"[^\W_]+", text, flags=re.UNICODE)
        if len(token) > 1
    }

def _lexical_similarity(query_tokens: set[str], document: str) -> float:
    if not query_tokens:
        return 0.0
    document_tokens = _tokenize_for_matching(document)
    return len(query_tokens & document_tokens) / len(query_tokens)

def retrieve_context(
    query: str, top_k: int = 5, min_similarity: float = 0.25
) -> list[dict[str, Any]]:
    query = query.strip()
    if not query:
        return []
    if top_k < 1:
        raise ValueError("top_k phải lớn hơn 0.")
    if not 0.0 <= min_similarity <= 1.0:
        raise ValueError("min_similarity phải nằm trong khoảng từ 0 đến 1.")

    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []

    result = collection.query(
        query_embeddings=[encode_query(query)],
        n_results=min(max(top_k * 4, top_k), count),
        include=["documents", "metadatas", "distances"],
    )
    candidates = []
    query_tokens = _tokenize_for_matching(query)
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]

    for doc, meta, distance in zip(docs, metas, distances):
        similarity = max(0.0, min(1.0, 1.0 - float(distance)))
        if similarity < min_similarity:
            continue
        lexical = _lexical_similarity(query_tokens, doc or "")
        meta = meta or {}
        candidates.append({
            "text": doc, "source": meta.get("source", "Tài liệu không rõ"),
            "page": meta.get("page", "?"), "chunk": meta.get("chunk", "?"),
            "similarity": similarity, "lexical_similarity": lexical,
            "retrieval_score": 0.8 * similarity + 0.2 * lexical,
            "source_id": meta.get("source_id", ""),
        })

    selected, remaining = [], candidates[:]
    while remaining and len(selected) < top_k:
        best_index, best_score = 0, float("-inf")
        for index, candidate in enumerate(remaining):
            candidate_tokens = _tokenize_for_matching(candidate["text"] or "")
            redundancy = max(
                (
                    len(candidate_tokens & _tokenize_for_matching(prev["text"] or ""))
                    / max(1, len(candidate_tokens))
                    for prev in selected
                ),
                default=0.0,
            )
            score = 0.8 * candidate["retrieval_score"] - 0.2 * redundancy
            if score > best_score:
                best_index, best_score = index, score
        selected.append(remaining.pop(best_index))
    return selected
