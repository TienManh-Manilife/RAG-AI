"""Index vetted system PDFs into Chroma. Caller must enforce staff/admin access."""
import hashlib
from pathlib import Path
from typing import Any
from .database import get_collection
from .embeddings import encode_documents
from .text_processing import chunk_text, extract_pdf_pages

MAX_SYSTEM_PDF_BYTES = 30 * 1024 * 1024


def ingest_pdf_bytes(filename: str, pdf_bytes: bytes, chunk_size=850, overlap=140) -> dict[str, Any]:
    filename = Path(filename).name
    if not filename.lower().endswith('.pdf') or not pdf_bytes or len(pdf_bytes) > MAX_SYSTEM_PDF_BYTES:
        raise ValueError('Chỉ chấp nhận PDF hợp lệ không quá 30 MB.')
    pages = extract_pdf_pages(pdf_bytes)
    if not pages:
        raise ValueError('PDF không có văn bản trích xuất. Hãy OCR bản scan trước.')
    source_id = hashlib.sha256(filename.casefold().encode()).hexdigest()[:20]
    documents, metas, ids = [], [], []
    for page, text in pages:
        for i, chunk in enumerate(chunk_text(text, chunk_size, overlap), 1):
            documents.append(chunk)
            metas.append({'source': filename, 'source_id': source_id, 'page': int(page), 'chunk': i})
            ids.append(f'{source_id}-p{page}-c{i}')
    if not documents or len(documents) > 10000:
        raise ValueError('PDF không có nội dung phù hợp hoặc quá nhiều đoạn.')
    embeds = encode_documents(documents)
    coll = get_collection()
    # Chroma upsert for changes to the same document; delete old chunks after embedding completes.
    coll.delete(where={'source_id': source_id})
    for start in range(0, len(ids), 200):
        stop = start+200
        coll.add(ids=ids[start:stop], documents=documents[start:stop],
                 metadatas=metas[start:stop], embeddings=embeds[start:stop])
    return {'filename': filename, 'pages': len(pages), 'chunks': len(documents), 'source_id': source_id}
