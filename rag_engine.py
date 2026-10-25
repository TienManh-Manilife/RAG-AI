"""PDF ingestion, chunking, multilingual embeddings, ChromaDB retrieval, and Ollama generation."""
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import requests

ROOT = Path(__file__).resolve().parent
CHROMA_DIR = ROOT / "data" / "chroma"
CHROMA_DIR.mkdir(parents=True, exist_ok=True)

COLLECTION_NAME = "company_policy"
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")

_client = None
_collection = None
_embedder = None


def get_collection():
    global _client, _collection
    if _collection is None:
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)
    return _embedder


def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[\t ]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 160) -> list[str]:
    """Split text into character-based chunks, preferring paragraph/sentence boundaries."""
    text = normalize_text(text)
    if not text:
        return []
    if chunk_size < 100:
        raise ValueError("chunk_size phải từ 100 ký tự trở lên.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap phải >= 0 và nhỏ hơn chunk_size.")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[str] = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        # Long paragraphs are broken near sentence punctuation, then by fixed length if needed.
        pieces = []
        if len(para) <= chunk_size:
            pieces = [para]
        else:
            sentences = re.split(r"(?<=[.!?。！？])\s+", para)
            buf = ""
            for sentence in sentences:
                if len(sentence) > chunk_size:
                    if buf:
                        pieces.append(buf)
                        buf = ""
                    start = 0
                    while start < len(sentence):
                        pieces.append(sentence[start:start + chunk_size])
                        start += chunk_size - overlap
                elif not buf or len(buf) + 1 + len(sentence) <= chunk_size:
                    buf = (buf + " " + sentence).strip()
                else:
                    pieces.append(buf)
                    buf = sentence
            if buf:
                pieces.append(buf)

        for piece in pieces:
            if not current:
                current = piece
            elif len(current) + 2 + len(piece) <= chunk_size:
                current += "\n\n" + piece
            else:
                previous = current
                flush()
                tail = previous[-overlap:] if overlap else ""
                current = (tail + "\n" + piece).strip() if tail else piece
                if len(current) > chunk_size:
                    current = current[-chunk_size:]
    flush()
    # De-duplicate identical chunks caused by short overlap/boundary handling.
    result = []
    seen = set()
    for chunk in chunks:
        key = chunk.strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result


def extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Return one (1-based page number, text) tuple for every page with extractable text."""
    import io
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            pages.append((page_number, text))
    return pages


def ingest_pdf_bytes(filename: str, pdf_bytes: bytes, chunk_size: int = 900, overlap: int = 160) -> dict[str, Any]:
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Chỉ hỗ trợ PDF.")
    pages = extract_pdf_pages(pdf_bytes)
    if not pages:
        raise ValueError(
            "Không trích xuất được văn bản. Nếu PDF là bản scan, cần OCR trước khi nạp."
        )

    collection = get_collection()
    source_id = hashlib.sha256((filename + hashlib.sha256(pdf_bytes).hexdigest()).encode("utf-8")).hexdigest()[:20]

    documents, metadatas, ids = [], [], []
    for page_num, page_text in pages:
        page_chunks = chunk_text(page_text, chunk_size=chunk_size, overlap=overlap)
        for local_idx, chunk in enumerate(page_chunks, start=1):
            chunk_id = f"{source_id}-p{page_num}-c{local_idx}"
            documents.append(chunk)
            metadatas.append({
                "source": filename,
                "source_id": source_id,
                "page": int(page_num),
                "chunk": int(local_idx),
            })
            ids.append(chunk_id)

    if not documents:
        raise ValueError("PDF không có đoạn văn bản đủ điều kiện để lập chỉ mục.")

    # Idempotent for the same file contents: remove previous copy, then add current chunks.
    try:
        collection.delete(where={"source_id": source_id})
    except Exception:
        pass

    embeddings = get_embedder().encode(
        documents,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings,
    )
    return {
        "filename": filename,
        "source_id": source_id,
        "pages": len(pages),
        "chunks": len(documents),
    }


def retrieve_context(query: str, top_k: int = 5, min_similarity: float = 0.25) -> list[dict[str, Any]]:
    query = query.strip()
    if not query:
        return []
    collection = get_collection()
    if collection.count() == 0:
        return []

    query_embedding = get_embedder().encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).tolist()[0]
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(max(1, top_k), collection.count()),
        include=["documents", "metadatas", "distances"],
    )
    hits = []
    docs = result.get("documents", [[]])[0]
    metas = result.get("metadatas", [[]])[0]
    distances = result.get("distances", [[]])[0]
    for doc, meta, distance in zip(docs, metas, distances):
        # Chroma cosine distance: similarity = 1 - distance.
        similarity = 1.0 - float(distance)
        if similarity < min_similarity:
            continue
        hits.append({
            "text": doc,
            "source": (meta or {}).get("source", "Tài liệu không rõ"),
            "page": (meta or {}).get("page", "?"),
            "chunk": (meta or {}).get("chunk", "?"),
            "similarity": similarity,
            "source_id": (meta or {}).get("source_id", ""),
        })
    return hits


def generate_answer(
    query: str,
    hits: list[dict[str, Any]],
    model: str,
    temperature: float = 0.2,
    num_ctx: int = 4096,
) -> tuple[str, dict[str, Any]]:
    if not hits:
        return (
            "Mình chưa tìm thấy đoạn tài liệu đủ liên quan để trả lời chắc chắn. "
            "Bạn có thể diễn đạt câu hỏi cụ thể hơn hoặc bổ sung tài liệu. "
            "Nếu đây là vấn đề về quyền lợi cá nhân hay thông tin chưa có trong tài liệu, "
            "hãy liên hệ bộ phận phụ trách.",
            {"no_context": True},
        )

    context_blocks = []
    for i, hit in enumerate(hits, start=1):
        context_blocks.append(
            f"[Nguồn {i}: {hit['source']}, trang {hit['page']}]\n{hit['text']}"
        )
    context = "\n\n---\n\n".join(context_blocks)

    system = (
        "Bạn là trợ lý hỏi đáp quy định công ty. Trả lời bằng tiếng Việt. "
        "Chỉ sử dụng thông tin trong NGỮ CẢNH TÀI LIỆU được cung cấp. "
        "Nếu tài liệu không đủ căn cứ, hãy nói rõ chưa tìm thấy thông tin trong tài liệu và đề nghị liên hệ bộ phận phụ trách. "
        "Không tự bịa chính sách, mức tiền, thời hạn hoặc quyền lợi. "
        "Trích dẫn nguồn theo dạng [Tên tài liệu, trang N] cho các ý chính. "
        "Không tuyên bố đã đọc tài liệu ngoài phần ngữ cảnh. Chỉ xuất câu trả lời cuối cùng, không xuất suy luận."
    )
    user_content = (
        "NGỮ CẢNH TÀI LIỆU:\n"
        f"{context}\n\n"
        f"CÂU HỎI:\n{query}\n\n"
        "Hãy trả lời ngắn gọn, rõ ràng. Mỗi thông tin chính cần có trích dẫn trang nếu phù hợp. "
        "Nếu thiếu dữ kiện, hãy nêu rõ giới hạn."
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": "/no_think\n" + user_content},
        ],
        "stream": False,
        "think": False,
        "options": {
            "temperature": float(temperature),
            "num_ctx": int(num_ctx),
            "num_predict": 800,
        },
    }
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        timeout=(5, 240),
    )
    response.raise_for_status()
    data = response.json()
    message = data.get("message") or {}
    content = (message.get("content") or "").strip()
    if not content:
        # Do not expose the model's internal thinking field to end users.
        reason = data.get("done_reason", "unknown")
        content = (
            "Model Ollama chưa trả về câu trả lời cuối cùng (trường content rỗng; "
            f"done_reason={reason}). Hãy thử model khác hoặc kiểm tra cấu hình chế độ suy luận của model."
        )
    return content, data


def get_collection_stats() -> dict[str, int]:
    collection = get_collection()
    count = collection.count()
    sources = list_indexed_sources()
    return {"chunks": count, "sources": len(sources)}


def list_indexed_sources() -> list[dict[str, Any]]:
    collection = get_collection()
    count = collection.count()
    if count == 0:
        return []
    data = collection.get(include=["metadatas"])
    grouped: dict[str, dict[str, Any]] = {}
    for meta in data.get("metadatas") or []:
        if not meta:
            continue
        source_id = meta.get("source_id", "")
        item = grouped.setdefault(source_id, {
            "source_id": source_id,
            "source": meta.get("source", "Unknown"),
            "chunks": 0,
            "pages": set(),
        })
        item["chunks"] += 1
        item["pages"].add(meta.get("page"))
    return [
        {
            "source_id": item["source_id"],
            "source": item["source"],
            "chunks": item["chunks"],
            "pages": len(item["pages"]),
        }
        for item in sorted(grouped.values(), key=lambda x: x["source"].lower())
    ]


def delete_source(source_id: str) -> None:
    get_collection().delete(where={"source_id": source_id})


def clear_collection() -> None:
    global _collection
    try:
        get_collection().delete(where={})
    except Exception:
        # Empty collection deletion semantics vary slightly by Chroma version.
        try:
            _client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        _collection = None
        get_collection()
