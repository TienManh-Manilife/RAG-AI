"""ChromaDB client and collection lifecycle."""
import chromadb
from chromadb.config import Settings
from .config import CHROMA_DIR, COLLECTION_NAME

_client = None
_collection = None

def get_collection():
    global _client, _collection
    if _collection is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(
            path=str(CHROMA_DIR),
            settings=Settings(anonymized_telemetry=False),
        )
        _collection = _client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection

def reset_collection_cache():
    global _client, _collection
    _client = None
    _collection = None

def clear_collection():
    global _collection
    collection = get_collection()
    try:
        collection.delete(where={})
    except Exception:
        try:
            _client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        _collection = None
        get_collection()
