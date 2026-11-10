"""Text normalization, PDF page extraction, and character-based chunking."""
import re
from pypdf import PdfReader

def normalize_text(text: str) -> str:
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def chunk_text(text: str, chunk_size: int = 900, overlap: int = 160) -> list[str]:
    """Split into character chunks, preferring paragraph and sentence boundaries."""
    text = normalize_text(text)
    if not text:
        return []
    if chunk_size < 100:
        raise ValueError("chunk_size phải từ 100 ký tự trở lên.")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap phải >= 0 và nhỏ hơn chunk_size.")

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    current = ""

    def flush():
        nonlocal current
        if current.strip():
            chunks.append(current.strip())
        current = ""

    for para in paragraphs:
        if len(para) <= chunk_size:
            pieces = [para]
        else:
            pieces, buf = [], ""
            sentences = re.split(r"(?<=[.!?。！？])\s+", para)
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
                if overlap:
                    tail_size = max(0, chunk_size - len(piece) - 1)
                    tail = previous[-min(overlap, tail_size):]
                    current = (tail + "\n" + piece).strip() if tail else piece
                else:
                    current = piece
    flush()

    result, seen = [], set()
    for chunk in chunks:
        key = chunk.strip()
        if key and key not in seen:
            seen.add(key)
            result.append(key)
    return result

def extract_pdf_pages(pdf_bytes: bytes) -> list[tuple[int, str]]:
    """Return (1-based page number, extracted text) for pages containing text."""
    import io
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            pages.append((page_number, text))
    return pages
