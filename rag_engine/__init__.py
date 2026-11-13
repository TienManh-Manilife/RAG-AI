"""Modular RAG engine public API."""

from .config import (
    ROOT,
    DATA_DIR,
    CHROMA_DIR,
    HANDOFF_FILE,
    METRICS_FILE,
    COLLECTION_NAME,
    DEFAULT_EMBEDDING_MODEL,
    EMBEDDING_MODEL_PATH,
    OLLAMA_BASE_URL,
    get_ai_server_config,
)

from .text_processing import normalize_text, chunk_text, extract_pdf_pages
from .ingestion import ingest_pdf_bytes
from .retrieval import retrieve_context
from .generation import clean_model_answer, generate_answer
from .handoff import should_handoff, create_handoff_request, list_handoff_requests
from .metrics import record_interaction, get_operational_metrics
from .sources import get_collection_stats, list_indexed_sources, delete_source
from .database import get_collection, clear_collection
from .embeddings import get_embedder

__all__ = [
    "normalize_text", "chunk_text", "extract_pdf_pages", "ingest_pdf_bytes",
    "retrieve_context", "clean_model_answer", "generate_answer",
    "should_handoff", "create_handoff_request", "list_handoff_requests",
    "record_interaction", "get_operational_metrics", "get_collection_stats",
    "list_indexed_sources", "delete_source", "get_collection",
    "clear_collection", "get_embedder",
]
__all__ += [
    "ROOT",
    "DATA_DIR",
    "CHROMA_DIR",
    "HANDOFF_FILE",
    "METRICS_FILE",
    "COLLECTION_NAME",
    "DEFAULT_EMBEDDING_MODEL",
    "EMBEDDING_MODEL_PATH",
    "OLLAMA_BASE_URL",
    "get_ai_server_config",
]
