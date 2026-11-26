"""Staff-managed knowledge base stored in SQLite; source visibility enforced server-side."""
from __future__ import annotations
import hashlib
import io
import json
import re
import secrets
import sqlite3
from pathlib import Path
from typing import Any
from .auth import db, require_role, utcnow
from .config import KNOWLEDGE_DIR

MAX_BYTES = 15 * 1024 * 1024
ALLOWED = {'.md', '.txt', '.pdf', '.docx'}

def init_docs():
    with db() as c:
        c.executescript('''
        CREATE TABLE IF NOT EXISTS documents(
            id TEXT PRIMARY KEY, title TEXT NOT NULL, filename TEXT NOT NULL,
            audience TEXT NOT NULL CHECK(audience IN ('customer','staff')),
            sha256 TEXT NOT NULL, owner_id TEXT, created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL, seeded INTEGER NOT NULL DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS chunks(
            id TEXT PRIMARY KEY, document_id TEXT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
            section TEXT NOT NULL, page INTEGER NOT NULL, content TEXT NOT NULL,
            embedding_json TEXT, embedding_model TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_chunks_doc ON chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_docs_audience ON documents(audience);
        CREATE TABLE IF NOT EXISTS app_state(key TEXT PRIMARY KEY, value TEXT NOT NULL);
        ''')

def extract_file(filename: str, payload: bytes) -> list[tuple[int, str]]:
    extension = Path(filename).suffix.lower()
    if extension not in ALLOWED or not payload or len(payload) > MAX_BYTES:
        raise ValueError('Chỉ hỗ trợ PDF, DOCX, TXT, MD tối đa 15 MB.')
    if extension == '.pdf':
        from pypdf import PdfReader
        try:
            pdf = PdfReader(io.BytesIO(payload), strict=False)
            if len(pdf.pages) > 200:
                raise ValueError('PDF vượt quá 200 trang.')
            return [(i, p.extract_text() or '') for i,p in enumerate(pdf.pages, 1)]
        except Exception as exc:
            raise ValueError('Không đọc được PDF hoặc PDF được bảo vệ.') from exc
    if extension == '.docx':
        from docx import Document
        try:
            d = Document(io.BytesIO(payload))
            return [(1, '\n'.join(p.text for p in d.paragraphs if p.text.strip()))]
        except Exception as exc:
            raise ValueError('DOCX không hợp lệ.') from exc
    try:
        return [(1, payload.decode('utf-8-sig'))]
    except UnicodeDecodeError as exc:
        raise ValueError('TXT/MD yêu cầu mã UTF-8.') from exc

def _chunk_pages(pages: list[tuple[int, str]]) -> list[tuple[int, str, str]]:
    result = []
    for page, text in pages:
        text = text.replace('\x00', ' ').strip()
        # Markdown sections are indivisible fact units for accurate retrieval and citation.
        sections = re.split(r'(?m)(?=^##\s+)', text)
        for raw in sections:
            raw = raw.strip()
            if not raw:
                continue
            heading = next((line.lstrip('# ').strip() for line in raw.splitlines() if line.startswith('## ')), 'Thông tin chung')
            # Fallback for long PDFs: 1100 chars with 150 chars of overlap.
            if len(raw) <= 1500:
                if len(raw) >= 30: result.append((page, heading, raw))
                continue
            step = 1100
            for start in range(0, len(raw), step):
                piece = raw[max(0, start - 130): start+step].strip()
                if len(piece) >= 30: result.append((page, heading, piece))
    if not result or len(result) > 1500:
        raise ValueError('Tài liệu không có văn bản trích xuất hoặc quá lớn. PDF scan cần OCR trước.')
    return result

def ingest(actor: dict|None, filename: str, payload: bytes, audience='customer', *, seeded=False, title=None):
    if not seeded:
        require_role(actor)
    if audience not in {'customer','staff'}:
        raise ValueError('Phạm vi tài liệu không hợp lệ.')
    filename = Path(filename).name
    if len(filename) > 180:
        raise ValueError('Tên file quá dài.')
    chunks = _chunk_pages(extract_file(filename, payload))
    sha = hashlib.sha256(payload).hexdigest()
    doc_id = hashlib.sha256(('seed:' if seeded else 'upload:').encode() + filename.casefold().encode()).hexdigest()[:24]
    now = utcnow()
    with db() as c:
        # Replace atomically; no partial ingestion on error.
        c.execute('''INSERT INTO documents(id,title,filename,audience,sha256,owner_id,created_at,updated_at,seeded)
                     VALUES(?,?,?,?,?,?,?,?,?)
                     ON CONFLICT(id) DO UPDATE SET title=excluded.title,filename=excluded.filename,
                     audience=excluded.audience,sha256=excluded.sha256,owner_id=excluded.owner_id,
                     updated_at=excluded.updated_at''',
                  (doc_id, (title or Path(filename).stem)[:180], filename, audience, sha,
                   actor['id'] if actor else None, now, now, int(seeded)))
        c.execute('DELETE FROM chunks WHERE document_id=?', (doc_id,))
        c.executemany('''INSERT INTO chunks(id,document_id,section,page,content,embedding_json,embedding_model)
                         VALUES(?,?,?,?,?,NULL,NULL)''',
                      [(f'{doc_id}:{i}', doc_id, heading[:180], page, text)
                       for i,(page,heading,text) in enumerate(chunks,1)])
    return {'id': doc_id, 'filename': filename, 'chunks': len(chunks), 'audience': audience}

def seed_documents():
    """Idempotent; never overwrites staff modifications, even if the same seeded file exists."""
    manifest = json.loads((KNOWLEDGE_DIR / 'manifest.json').read_text(encoding='utf-8'))
    added = 0
    with db() as c:
        if c.execute("SELECT 1 FROM app_state WHERE key='demo_seed_v1'").fetchone():
            return 0
        existing = {row['filename'] for row in c.execute('SELECT filename FROM documents')}
    for item in manifest:
        if item['filename'] in existing:
            continue
        filename = item['filename']
        payload = (KNOWLEDGE_DIR / filename).read_bytes()
        ingest(None, filename, payload, item['audience'], seeded=True, title=item['title'])
        added += 1
    with db() as c:
        c.execute("INSERT OR IGNORE INTO app_state(key,value) VALUES('demo_seed_v1',?)",(utcnow(),))
    return added

def list_docs(actor=None):
    staff = bool(actor and actor.get('role') in {'staff','admin'})
    with db() as c:
        rows = c.execute('''SELECT d.*,count(ch.id) AS chunks_count,
            sum(CASE WHEN ch.embedding_json IS NOT NULL THEN 1 ELSE 0 END) AS embedded_count
            FROM documents d LEFT JOIN chunks ch ON ch.document_id=d.id
            WHERE (?=1 OR d.audience='customer')
            GROUP BY d.id ORDER BY d.created_at,d.filename''', (int(staff),)).fetchall()
    return [dict(r) for r in rows]

def remove(actor, doc_id):
    require_role(actor)
    with db() as c:
        row = c.execute('SELECT filename FROM documents WHERE id=?', (doc_id,)).fetchone()
        if not row: raise ValueError('Tài liệu không tồn tại.')
        c.execute('DELETE FROM documents WHERE id=?', (doc_id,))
    return row['filename']

def authorized_chunks(actor=None):
    staff = bool(actor and actor.get('role') in {'staff','admin'})
    with db() as c:
        rows = c.execute('''SELECT ch.*,d.title,d.filename,d.audience FROM chunks ch
                    JOIN documents d ON d.id=ch.document_id
                    WHERE (?=1 OR d.audience='customer') ORDER BY d.filename,ch.id''', (int(staff),)).fetchall()
    return [dict(r) for r in rows]

def refresh_embeddings(batch_size=12, max_batches=None):
    """Persist embeddings; return counters; safe to rerun after restart."""
    from .config import EMBED_MODEL
    from .ollama import embed_batch
    with db() as c:
        pending = [dict(r) for r in c.execute('''SELECT id,content FROM chunks
                WHERE embedding_json IS NULL OR embedding_model!=? ORDER BY id''', (EMBED_MODEL,)).fetchall()]
    completed = 0
    for offset in range(0, len(pending), batch_size):
        if max_batches is not None and offset // batch_size >= max_batches: break
        batch = pending[offset: offset+batch_size]
        vectors = embed_batch([x['content'] for x in batch])
        with db() as c:
            c.executemany('UPDATE chunks SET embedding_json=?,embedding_model=? WHERE id=?',
                          [(json.dumps(v, separators=(',', ':')), EMBED_MODEL, r['id']) for r,v in zip(batch,vectors)])
        completed += len(batch)
    return {'processed': completed, 'pending_before': len(pending), 'remaining': len(pending)-completed}
