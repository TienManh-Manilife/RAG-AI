
import os
import secrets

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

load_dotenv()

app = FastAPI(title="RAG AI Inference Server")

OLLAMA_URL = "http://127.0.0.1:11434"
MODEL_NAME = "qwen3:4b"
API_KEY = os.getenv("AI_SERVER_API_KEY")


class ChatRequest(BaseModel):
    messages: list[dict[str, str]]
    temperature: float = 0.3


@app.get("/health")
async def health():
    return {
        "status": "online",
        "service": "RAG AI Server"
    }


@app.post("/v1/chat")
async def chat(
    request: ChatRequest,
    authorization: str | None = Header(default=None)
):
    if not API_KEY:
        raise HTTPException(
            status_code=500,
            detail="Server API key is not configured"
        )

    expected = f"Bearer {API_KEY}"

    if not authorization or not secrets.compare_digest(
        authorization, expected
    ):
        raise HTTPException(
            status_code=401,
            detail="Unauthorized"
        )

    try:
        async with httpx.AsyncClient(timeout=180.0) as client:
            response = await client.post(
                f"{OLLAMA_URL}/api/chat",
                json={
                    "model": MODEL_NAME,
                    "messages": request.messages,
                    "stream": False,
                    "options": {
                        "temperature": request.temperature
                    }
                }
            )

            response.raise_for_status()
            result = response.json()

        return {
            "model": MODEL_NAME,
            "answer": result["message"]["content"]
        }

    except httpx.TimeoutException:
        raise HTTPException(
            status_code=504,
            detail="Ollama inference timed out"
        )

    except httpx.HTTPError:
        raise HTTPException(
            status_code=502,
            detail="Cannot communicate with Ollama"
        )
