"""Deterministic escalation rule; score is a heuristic, not calibrated probability."""
from typing import Any


def should_handoff(hits: list[dict[str, Any]], answer_metadata: dict[str, Any],
                   min_similarity: float = 0.45) -> tuple[bool, str, float]:
    """Backward-compatible name/return shape; third value is evidence index."""
    best = max((float(h.get('evidence_score', h.get('similarity', 0))) for h in hits), default=0.0)
    if not hits:
        return True, 'Chưa tìm thấy nguồn tài liệu phù hợp.', best
    if answer_metadata.get('no_context') or answer_metadata.get('insufficient_evidence'):
        return True, 'Tài liệu hoặc câu trả lời chưa đủ căn cứ xác minh.', best
    if answer_metadata.get('answer_filtered'):
        return True, 'Mô hình không cung cấp câu trả lời cuối cùng hợp lệ.', best
    if answer_metadata.get('missing_citation'):
        return True, 'Câu trả lời chưa có trích dẫn nguồn hợp lệ.', best
    if best < min_similarity:
        return True, f'Chỉ số phù hợp bằng chứng {best:.0%} thấp hơn ngưỡng {min_similarity:.0%}.', best
    return False, '', best
