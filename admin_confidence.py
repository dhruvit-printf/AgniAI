"""
admin_confidence.py
===================
Unified confidence framework for the AgniAI Admin Chatbot.

Problem
-------
v1 has two separate confidence systems:
  - `classify_admin_intent()` returns a string: "high" / "medium" / "low"
  - `plan_query()` returns a float: 0.0 – 1.0

These are inconsistent and callers must handle both separately.

Solution
--------
`AdminConfidence` is a single dataclass that:
  1. Holds both representations (float score + string label).
  2. Is produced by `compute_confidence()` from any combination of
     intent result and/or query plan.
  3. Can be serialised directly into API responses.

The float thresholds mirror what the existing code already uses implicitly:
  high   ≥ 0.75
  medium ≥ 0.50
  low    <  0.50

Usage
-----
    from admin_confidence import compute_confidence

    conf = compute_confidence(intent_result=intent, plan=query_plan)
    # conf.score  → float
    # conf.label  → "high" | "medium" | "low"
    # conf.to_dict() → {"score": 0.85, "label": "high"}
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional


# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_THRESHOLD_HIGH   = 0.75
_THRESHOLD_MEDIUM = 0.50


# ---------------------------------------------------------------------------
# Label → float mapping (for the intent string from classify_admin_intent)
# ---------------------------------------------------------------------------

_INTENT_LABEL_TO_FLOAT: Dict[str, float] = {
    "high":   0.90,
    "medium": 0.65,
    "low":    0.35,
}


# ---------------------------------------------------------------------------
# Main dataclass
# ---------------------------------------------------------------------------

@dataclass
class AdminConfidence:
    score: float  # 0.0 – 1.0
    label: str    # "high" | "medium" | "low"

    # Convenience predicates
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


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def compute_confidence(
    *,
    intent_result: Optional[Dict[str, Any]] = None,
    plan: Any = None,          # QueryPlan dataclass — imported lazily to avoid cycles
) -> AdminConfidence:
    """
    Produce a unified AdminConfidence from any combination of intent result
    and/or query plan.

    Priority order:
    1. If both are supplied → average their scores, then weight by whether
       intent has a valid category (adds 0.05 bonus).
    2. If only intent_result → convert string label to float.
    3. If only plan → use plan.confidence directly.
    4. Otherwise → low confidence.
    """
    intent_score: Optional[float] = None
    plan_score:   Optional[float] = None

    if intent_result:
        label        = (intent_result.get("confidence") or "low")
        intent_score = _score_from_label(label)
        # Bonus for having a valid category + subcategory
        if intent_result.get("category") and intent_result.get("subcategory"):
            intent_score = min(1.0, intent_score + 0.05)

    if plan is not None:
        plan_score = float(getattr(plan, "confidence", 0.0))

    if intent_score is not None and plan_score is not None:
        # Weighted average: intent = 60%, plan = 40%
        raw = 0.60 * intent_score + 0.40 * plan_score
    elif intent_score is not None:
        raw = intent_score
    elif plan_score is not None:
        raw = plan_score
    else:
        raw = 0.30

    raw   = max(0.0, min(1.0, raw))
    label = _label_from_score(raw)
    return AdminConfidence(score=round(raw, 3), label=label)


def normalise_intent_confidence(intent_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of *intent_result* with the confidence field replaced by
    a unified float+label dict, for inclusion in API responses.
    """
    conf = compute_confidence(intent_result=intent_result)
    updated = dict(intent_result)
    updated["confidence"] = conf.to_dict()
    return updated