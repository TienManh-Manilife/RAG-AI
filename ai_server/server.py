"""Minimal protected Ollama inference gateway used by Streamlit Cloud.

Launch locally: uvicorn ai_server.server:app --host 127.0.0.1 --port 8000
Publish through an HTTPS tunnel if needed. Never expose Ollama directly.
"""
import hmac
import os
from typing import Any

import requests
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field

app = FastAPI(title='RAG-AI Inference Gateway', docs_url=None, redoc_url=None)

OLLAMA_URL = os.getenv('OLLAMA_BASE_URL', 'http://127.0.0.1:11434').rstrip('/')
GATEWAY_KEY = os.getenv('AI_SERVER_API_KEY', '')
MODEL = os.getenv('AI_SERVER_MODEL', 'qwen3:4b')


class ChatMessage(BaseModel):
    role: str
    content: str = Field(min_length=1, max_length=50_000)


class ChatRequest(BaseModel):
    messages: list[ChatMessage] = Field(min_length=1, max_length=12)
    temperature: float = Field(default=0.1, ge=0, le=1)


@app.get('/health')
def health():
    return {'status': 'online', 'service': 'RAG AI Server'}


@app.post('/v1/chat')
def chat(request: ChatRequest, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    # Fail closed: without an explicitly configured key, inference is disabled.
    if not GATEWAY_KEY or len(GATEWAY_KEY) < 24 or GATEWAY_KEY.startswith(('REPLACE_', 'CHANGE_', 'YOUR_')):
        raise HTTPException(status_code=503, detail='Gateway authentication is not configured')
    expected = 'Bearer ' + GATEWAY_KEY
    if not authorization or not hmac.compare_digest(authorization, expected):
        raise HTTPException(status_code=401, detail='Unauthorized')
    if any(item.role not in {'system','user','assistant'} for item in request.messages):
        raise HTTPException(status_code=400, detail='Invalid message role')
    try:
        response = requests.post(f'{OLLAMA_URL}/api/chat', json={
            'model': MODEL, 'messages': [m.model_dump() for m in request.messages],
            'stream': False, 'think': False,
            'options': {'temperature': request.temperature, 'num_ctx': 4096,
                        'num_predict': 1100},
        }, timeout=(5, 180))
        response.raise_for_status()
        data = response.json()
    except (requests.RequestException, ValueError):
        # Do not leak internal network locations or raw upstream error bodies.
        raise HTTPException(status_code=502, detail='Local model unavailable')
    return {'model': MODEL, 'answer': data.get('message', {}).get('content', '')}
