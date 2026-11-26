"""Local-only settings. Network boundary: Streamlit container -> Ollama container."""
import os
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = Path(os.getenv('RAG_DATA_DIR', str(ROOT / 'data')))
SQLITE_PATH = DATA_DIR / 'rag_service.sqlite3'
KNOWLEDGE_DIR = ROOT / 'knowledge_base'
OLLAMA_URL = os.getenv('OLLAMA_BASE_URL', 'http://ollama:11434').rstrip('/')
CHAT_MODEL = os.getenv('OLLAMA_CHAT_MODEL', 'qwen3:4b')
EMBED_MODEL = os.getenv('OLLAMA_EMBED_MODEL', 'bge-m3')
RAG_MIN_RELEVANCE = float(os.getenv('RAG_MIN_RELEVANCE', '0.38'))

def get_auth_config():
    return {
        'staff_invite_code': '', # Admin provisions staff; staff self-enrollment disabled.
        'bootstrap_admin_email': os.getenv('BOOTSTRAP_ADMIN_EMAIL', '').strip(),
        'bootstrap_admin_password': os.getenv('BOOTSTRAP_ADMIN_PASSWORD', ''),
        'bootstrap_admin_name': os.getenv('BOOTSTRAP_ADMIN_NAME', 'Quản trị An Phát'),
    }
