"""Persist and calculate operational metrics."""
import json
from typing import Any
from .config import DATA_DIR, METRICS_FILE

def _load_metrics() -> dict[str, Any]:
    metrics = {"questions": 0, "handoffs": 0, "similarity_sum": 0.0}
    if METRICS_FILE.exists():
        try:
            loaded = json.loads(METRICS_FILE.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                metrics.update(loaded)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Không đọc được số liệu vận hành: {METRICS_FILE}"
            ) from exc
    return metrics

def record_interaction(*, handoff_required: bool, best_similarity: float) -> None:
    """Record interactions; handoffs currently counts required/suggested handoffs."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    metrics = _load_metrics()
    metrics["questions"] = int(metrics["questions"]) + 1
    metrics["handoffs"] = int(metrics["handoffs"]) + int(handoff_required)
    metrics["similarity_sum"] = (
        float(metrics["similarity_sum"]) + float(best_similarity)
    )
    METRICS_FILE.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

def get_operational_metrics() -> dict[str, Any]:
    metrics = _load_metrics()
    questions = int(metrics["questions"])
    handoffs = int(metrics["handoffs"])
    return {
        "questions": questions,
        "handoffs": handoffs,
        "handoff_rate": handoffs / questions if questions else 0.0,
        "average_similarity": (
            float(metrics["similarity_sum"]) / questions if questions else 0.0
        ),
    }
