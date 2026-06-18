"""
admin_confidence.py
===================
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

_THRESHOLD_HIGH = 0.75
_THRESHOLD_MEDIUM = 0.50


_INTENT_LABEL_TO_FLOAT: Dict[str, float] = {
    "high": 0.90,
    "medium": 0.65,
    "low": 0.35,
}


@dataclass
class AdminConfidence:
    score: float
    label: str

    @property
    def is_high(self) -> bool:
        return self.score >= _THRESHOLD_HIGH

    @property
    def is_medium(self) -> bool:
        return _THRESHOLD_MEDIUM <= self.score < _THRESHOLD_HIGH

    @property
    def is_low(self) -> bool:
        return self.score < _THRESHOLD_MEDIUM

    def to_dict(self) -> Dict[str, Any]:
        return {"score": round(self.score, 3), "label": self.label}


def _label_from_score(score: float) -> str:
    if score >= _THRESHOLD_HIGH:
        return "high"
    if score >= _THRESHOLD_MEDIUM:
        return "medium"
    return "low"


def _score_from_label(label: str) -> float:
    return _INTENT_LABEL_TO_FLOAT.get((label or "").lower(), 0.35)


def compute_confidence(
    *,
    intent_result: Optional[Dict[str, Any]] = None,
    plan: Any = None,
) -> AdminConfidence:
    intent_score: Optional[float] = None
    plan_score: Optional[float] = None

    if intent_result:
        label = intent_result.get("confidence") or "low"
        intent_score = _score_from_label(label)
        if intent_result.get("category") and intent_result.get("subcategory"):
            intent_score = min(1.0, intent_score + 0.05)

    if plan is not None:
        plan_score = float(getattr(plan, "confidence", 0.0))

    if intent_score is not None and plan_score is not None:
        raw = 0.60 * intent_score + 0.40 * plan_score
    elif intent_score is not None:
        raw = intent_score
    elif plan_score is not None:
        raw = plan_score
    else:
        raw = 0.30

    raw = max(0.0, min(1.0, raw))
    label = _label_from_score(raw)
    return AdminConfidence(score=round(raw, 3), label=label)


def normalise_intent_confidence(intent_result: Dict[str, Any]) -> Dict[str, Any]:
    conf = compute_confidence(intent_result=intent_result)
    updated = dict(intent_result)
    updated["confidence"] = conf.to_dict()
    return updated
