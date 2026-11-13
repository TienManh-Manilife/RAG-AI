"""SentenceTransformer embeddings with local and Hugging Face fallback.

On a developer machine, use ROOT/embedding_model when it is present.
On Streamlit Cloud, where that large folder is usually not committed to Git,
download/load the same model from Hugging Face Hub on first use.
"""
from sentence_transformers import SentenceTransformer

from .config import DEFAULT_EMBEDDING_MODEL, EMBEDDING_MODEL_PATH

_embedder = None


def get_embedder():
    global _embedder

    if _embedder is not None:
        return _embedder

    model_path = EMBEDDING_MODEL_PATH
    modules_file = model_path / "modules.json"

    if model_path.is_dir() and modules_file.is_file():
        print(f"[RAG] Loading local embedding model: {model_path}")
        _embedder = SentenceTransformer(
            str(model_path),
            local_files_only=True,
        )
    else:
        # Do not fail just because the local model directory is absent on
        # Streamlit Cloud. Load the configured Hub model and let HF cache it.
        print(
            "[RAG] Local embedding model not found at "
            f"'{model_path}'. Loading Hugging Face model: "
            f"'{DEFAULT_EMBEDDING_MODEL}'"
        )
        _embedder = SentenceTransformer(DEFAULT_EMBEDDING_MODEL)

    print("[RAG] Embedding model is ready.")
    return _embedder


def encode_documents(texts):
    return get_embedder().encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()


def encode_query(query):
    return get_embedder().encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ).tolist()[0]
