"""Application paths and configuration. Secrets must never be committed."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv('RAG_DATA_DIR', str(ROOT / 'data'))).resolve()
CHROMA_DIR = DATA_DIR / 'chroma'
SQLITE_PATH = DATA_DIR / 'rag_service.sqlite3'
COLLECTION_NAME = os.getenv('CHROMA_COLLECTION', 'company_policy')
DEFAULT_EMBEDDING_MODEL = os.getenv('EMBEDDING_MODEL', 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
EMBEDDING_MODEL_PATH = Path(os.getenv('EMBEDDING_MODEL_PATH', str(ROOT / 'embedding_model')))


def _secrets(section: str) -> dict:
    try:
        import streamlit as st
        values = st.secrets.get(section, {})
        return dict(values)
    except Exception:
        return {}


def get_ai_server_config() -> dict:
    s = _secrets('ai_server')
    return {
        'base_url': str(s.get('base_url') or os.getenv('AI_SERVER_BASE_URL', '')).strip().rstrip('/'),
        'api_key': str(s.get('api_key') or os.getenv('AI_SERVER_API_KEY', '')).strip(),
        'model': str(s.get('model') or os.getenv('AI_SERVER_MODEL', 'qwen3:4b')).strip(),
    }


def get_auth_config() -> dict:
    s = _secrets('auth')
    return {
        'staff_invite_code': str(s.get('staff_invite_code') or os.getenv('STAFF_INVITE_CODE', '')),
        'bootstrap_admin_email': str(s.get('bootstrap_admin_email') or os.getenv('BOOTSTRAP_ADMIN_EMAIL', '')),
        'bootstrap_admin_password': str(s.get('bootstrap_admin_password') or os.getenv('BOOTSTRAP_ADMIN_PASSWORD', '')),
        'bootstrap_admin_name': str(s.get('bootstrap_admin_name') or os.getenv('BOOTSTRAP_ADMIN_NAME', 'System Administrator')),
    }
