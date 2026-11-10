"""Local SentenceTransformer embedding model."""
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL_PATH

_embedder = None

def get_embedder():
    global _embedder
    if _embedder is None:
        model_path = EMBEDDING_MODEL_PATH
        if not model_path.is_dir():
            raise FileNotFoundError(f"Không tìm thấy thư mục embedding model: {model_path}")
        if not (model_path / "modules.json").is_file():
            raise FileNotFoundError(
                f"Không tìm thấy modules.json trong: {model_path}. "
                "Hãy kiểm tra model đã tải đầy đủ chưa."
            )
        _embedder = SentenceTransformer(str(model_path), local_files_only=True)
        print(f"[RAG] Đã tải embedding model từ: {model_path}")
    return _embedder

def encode_documents(texts):
    return get_embedder().encode(
        texts, normalize_embeddings=True, convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()

def encode_query(query):
    return get_embedder().encode(
        [query], normalize_embeddings=True, convert_to_numpy=True
    ).tolist()[0]
