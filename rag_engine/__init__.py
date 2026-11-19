"""Public imports; heavy embedding and Chroma dependencies are loaded lazily."""
from importlib import import_module
from .config import ROOT, DATA_DIR, CHROMA_DIR, COLLECTION_NAME, DEFAULT_EMBEDDING_MODEL, get_ai_server_config, get_auth_config
from .auth import init_db, create_account, login, get_user, require_role, change_password
from .service import (new_conversation, list_conversations, get_conversation, list_messages,
                      add_message, log_event, submit_feedback, feedback_for_message,
                      request_handoff, get_handoff_for_message, list_handoffs,
                      update_handoff, metrics)
from .handoff import should_handoff
from .generation import generate_answer, clean_model_answer
from .evaluation import evaluate_rows, answer_scores, bool_value, suggest_evidence_threshold

_LAZY = {
    'retrieve_context': 'retrieval', 'rerank_candidates': 'retrieval', 'evidence_score': 'retrieval',
    'index_upload': 'private_docs', 'retrieve_uploaded': 'private_docs',
    'ingest_pdf_bytes': 'ingestion', 'get_collection_stats': 'sources',
    'list_indexed_sources': 'sources', 'delete_source': 'sources',
}

def __getattr__(name):
    if name in _LAZY:
        return getattr(import_module('.'+_LAZY[name], __name__), name)
    raise AttributeError(name)

def record_interaction(*, actor: dict, mode: str, handoff_required: bool, best_similarity: float):
    return log_event(actor, mode, best_similarity, handoff_required)


def get_operational_metrics(actor: dict):
    return metrics(actor)
