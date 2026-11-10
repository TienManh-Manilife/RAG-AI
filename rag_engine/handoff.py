"""Human-review decision and persistent handoff queue."""
import hashlib
import json
from datetime import datetime, timezone
from typing import Any
from .config import DATA_DIR, HANDOFF_FILE

def should_handoff(
    hits: list[dict[str, Any]], answer_metadata: dict[str, Any],
    min_similarity: float = 0.45,
) -> tuple[bool, str]:
    if not hits:
        return True, "Không tìm thấy đoạn tài liệu phù hợp."
    best_similarity = max(
        (float(hit.get("similarity", 0.0)) for hit in hits), default=0.0
    )
    if best_similarity < min_similarity:
        return (
            True,
            f"Độ tương đồng cao nhất ({best_similarity:.2f}) thấp hơn "
            f"ngưỡng an toàn ({min_similarity:.2f}).",
        )
    if answer_metadata.get("no_context"):
        return True, "Mô hình không có đủ ngữ cảnh để trả lời."
    return False, ""

def create_handoff_request(
    question: str, answer: str, reason: str, hits: list[dict[str, Any]]
) -> dict[str, Any]:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    requests_data = []
    if HANDOFF_FILE.exists():
        try:
            loaded = json.loads(HANDOFF_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded, list):
                requests_data = loaded
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Không đọc được hàng đợi chuyển nhân viên: {HANDOFF_FILE}"
            ) from exc

    now = datetime.now(timezone.utc).isoformat()
    record = {
        "id": hashlib.sha256(f"{question}:{now}".encode("utf-8")).hexdigest()[:16],
        "question": question, "answer": answer, "reason": reason,
        "sources": [
            {
                "source": hit.get("source"), "page": hit.get("page"),
                "similarity": hit.get("similarity"),
            }
            for hit in hits
        ],
        "status": "pending", "created_at": now,
    }
    requests_data.append(record)
    HANDOFF_FILE.write_text(
        json.dumps(requests_data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return record

def list_handoff_requests() -> list[dict[str, Any]]:
    if not HANDOFF_FILE.exists():
        return []
    try:
        loaded = json.loads(HANDOFF_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Không đọc được hàng đợi chuyển nhân viên: {HANDOFF_FILE}"
        ) from exc
    return loaded if isinstance(loaded, list) else []
