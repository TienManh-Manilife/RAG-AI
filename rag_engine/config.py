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
