"""Session-isolated indexes for user-supplied documents. Never insert into system Chroma."""
import hashlib
import io
from pathlib import Path
import numpy as np
from .text_processing import chunk_text, extract_pdf_pages, normalize_text
from .retrieval import rerank_candidates

ALLOWED_TYPES = {'.pdf', '.txt', '.docx'}
MAX_FILE_BYTES = 12 * 1024 * 1024
MAX_PAGES = 250
MAX_CHARS = 1_200_000
MAX_CHUNKS = 1800


def extract_pages(filename: str, raw: bytes):
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_TYPES:
        raise ValueError('Chỉ hỗ trợ PDF, DOCX, TXT.')
    if not raw or len(raw) > MAX_FILE_BYTES:
        raise ValueError('Tệp rỗng hoặc vượt quá giới hạn 12 MB.')
    if suffix == '.pdf':
        pages = extract_pdf_pages(raw)
    elif suffix == '.txt':
        pages = [(1, normalize_text(raw.decode('utf-8-sig', errors='replace')))]
    else:
        from docx import Document
        document = Document(io.BytesIO(raw))
        text = '\n\n'.join(p.text for p in document.paragraphs if p.text.strip())
        for table in document.tables:
            text += '\n' + '\n'.join(' | '.join(cell.text for cell in row.cells) for row in table.rows)
        pages = [(1, normalize_text(text))]
    if not pages:
        raise ValueError('Không trích xuất được chữ. PDF scan cần OCR riêng.')
    if len(pages) > MAX_PAGES or sum(len(text) for _, text in pages) > MAX_CHARS:
        raise ValueError('Tài liệu vượt giới hạn số trang/ký tự của phiên làm việc.')
    return pages


def index_upload(filename: str, raw: bytes, chunk_size=850, overlap=140):
    pages = extract_pages(filename, raw)
    chunks = []
    for page, text in pages:
        for i, piece in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap), 1):
            chunks.append({'text': piece, 'source': Path(filename).name, 'page': page,
                           'chunk': i, 'similarity': 0})
    if not chunks or len(chunks) > MAX_CHUNKS:
        raise ValueError('Không có nội dung hoặc quá nhiều đoạn dữ liệu.')
    from .embeddings import encode_documents
    vectors = np.asarray(encode_documents([c['text'] for c in chunks]), dtype=np.float32)
    return {'id': hashlib.sha256(raw).hexdigest(), 'filename': Path(filename).name,
            'chunks': chunks, 'embeddings': vectors, 'pages': len(pages)}


def retrieve_uploaded(query: str, indices: list[dict], top_k=5):
    if not query.strip() or not indices:
        return []
    from .embeddings import encode_query
    query_vec = np.asarray(encode_query(query), dtype=np.float32)
    query_vec = query_vec / max(np.linalg.norm(query_vec), 1e-8)
    candidates = []
    for index in indices:
        scores = index['embeddings'] @ query_vec
        n = min(len(scores), max(8 * top_k, 30))
        if n:
            positions = np.argpartition(-scores, n - 1)[:n]
            for i in positions:
                hit = dict(index['chunks'][int(i)])
                hit['similarity'] = float(np.clip(scores[i], 0, 1))
                candidates.append(hit)
    candidates.sort(key=lambda h: h['similarity'], reverse=True)
    return rerank_candidates(query, candidates[:max(80, 12*top_k)], top_k)
