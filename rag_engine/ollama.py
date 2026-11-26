"""Ollama client; all inference stays inside the Compose network."""
from __future__ import annotations
import requests
from .config import OLLAMA_URL, CHAT_MODEL, EMBED_MODEL

class OllamaUnavailable(RuntimeError):
    pass

def models():
    try:
        result = requests.get(f'{OLLAMA_URL}/api/tags', timeout=5)
        result.raise_for_status()
        return [m.get('name', '') for m in result.json().get('models', [])]
    except (requests.RequestException, ValueError) as exc:
        raise OllamaUnavailable('Không kết nối được Ollama nội bộ. Kiểm tra container ollama.') from exc

def ready():
    try:
        names = models()
        return {'online': True, 'chat': CHAT_MODEL in names,
                'embed': any(n == EMBED_MODEL or n == EMBED_MODEL+':latest' for n in names), 'models': names}
    except OllamaUnavailable:
        return {'online': False, 'chat': False, 'embed': False, 'models': []}

def embed_batch(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    try:
        r = requests.post(f'{OLLAMA_URL}/api/embed', json={'model': EMBED_MODEL, 'input': texts,
                          'truncate': True, 'keep_alive': '1m'}, timeout=(8, 240))
        r.raise_for_status()
        result = r.json().get('embeddings') or []
        if len(result) != len(texts):
            raise OllamaUnavailable('Ollama trả về số vector embedding không chính xác.')
        return result
    except (requests.RequestException, ValueError) as exc:
        raise OllamaUnavailable('Không tạo được embedding. Kiểm tra model bge-m3.') from exc

def chat(messages, temperature=0.05):
    try:
        r = requests.post(f'{OLLAMA_URL}/api/chat', json={
            'model': CHAT_MODEL, 'messages': messages, 'stream': False, 'think': False,
            'format': 'json', 'options': {'temperature': temperature, 'num_ctx': 8192, 'num_predict': 900}
        }, timeout=(8, 240))
        r.raise_for_status()
        return r.json().get('message', {}).get('content', '')
    except (requests.RequestException, ValueError) as exc:
        raise OllamaUnavailable('Mô hình Qwen3 chưa sẵn sàng. Kiểm tra docker compose logs ollama.') from exc
