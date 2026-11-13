"""Central configuration and filesystem paths."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CHROMA_DIR = DATA_DIR / "chroma"
HANDOFF_FILE = DATA_DIR / "handoff_requests.json"
METRICS_FILE = DATA_DIR / "metrics.json"

COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "company_policy")
DEFAULT_EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
)
EMBEDDING_MODEL_PATH = Path(
    os.getenv("EMBEDDING_MODEL_PATH", str(ROOT / "embedding_model"))
)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")


def get_ai_server_config():
    """Read remote inference settings from Streamlit secrets or environment."""
    secrets_config = {}
    try:
        import streamlit as st
        secrets_config = st.secrets.get("ai_server", {})
    except Exception:
        # Allows local tooling/imports when Streamlit secrets are unavailable.
        secrets_config = {}

    base_url = (
        secrets_config.get("base_url")
        or os.getenv("AI_SERVER_BASE_URL", "")
    ).strip().rstrip("/")
    api_key = (
        secrets_config.get("api_key")
        or os.getenv("AI_SERVER_API_KEY", "")
    ).strip()
    model = (
        secrets_config.get("model")
        or os.getenv("AI_SERVER_MODEL", "qwen3:4b")
    ).strip()

    return {"base_url": base_url, "api_key": api_key, "model": model}
