"""Compatibility helpers; operational events now require an authenticated actor."""
from .service import log_event, metrics


def record_interaction(*, actor: dict, mode: str, handoff_required: bool, best_similarity: float):
    """Write an operational event with the authenticated user identity."""
    log_event(actor, mode, best_similarity, handoff_required)


def get_operational_metrics(actor: dict):
    """Staff-only global dashboard; no anonymous access."""
    return metrics(actor)
