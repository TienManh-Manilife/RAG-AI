"""PDF ingestion and indexing into ChromaDB."""
import hashlib
from typing import Any
from .database import get_collection
from .embeddings import encode_documents
from .text_processing import chunk_text, extract_pdf_pages

def ingest_pdf_bytes(
    filename: str, pdf_bytes: bytes, chunk_size: int = 900, overlap: int = 160
) -> dict[str, Any]:
    if not filename.lower().endswith(".pdf"):
        raise ValueError("Chỉ hỗ trợ PDF.")
    pages = extract_pdf_pages(pdf_bytes)
    if not pages:
        raise ValueError(
            "Không trích xuất được văn bản. Nếu PDF là bản scan, cần OCR trước khi nạp."
        )

    collection = get_collection()
    source_id = hashlib.sha256(filename.strip().encode("utf-8")).hexdigest()[:20]
    documents, metadatas, ids = [], [], []
    for page_num, page_text in pages:
        for local_idx, chunk in enumerate(
            chunk_text(page_text, chunk_size=chunk_size, overlap=overlap), start=1
        ):
            documents.append(chunk)
            metadatas.append({
                "source": filename, "source_id": source_id,
                "page": int(page_num), "chunk": int(local_idx),
            })
            ids.append(f"{source_id}-p{page_num}-c{local_idx}")

    if not documents:
        raise ValueError("PDF không có đoạn văn bản đủ điều kiện để lập chỉ mục.")

    # Replace prior chunks for this filename, preventing stale versions.
    collection.delete(where={"source": filename})
    embeddings = encode_documents(documents)
    collection.add(
        ids=ids, documents=documents, metadatas=metadatas, embeddings=embeddings
    )
    return {
        "filename": filename, "source_id": source_id,
        "pages": len(pages), "chunks": len(documents),
    }
