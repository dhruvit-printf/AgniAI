"""
intelligence_common.py
=======================
Shared statistical/record helpers for the Intelligence Layer
(analysis_engine.py, prediction_engine.py, conclusion_engine.py).

This module consolidates the pieces that were confirmed byte-identical (or
functionally equivalent) across those engines. It intentionally does NOT
attempt to consolidate everything the engines have in common — some
similarly-named logic carries real behavioral differences that must not be
silently merged:

  - `_named_score_records` is NOT here. It has three genuinely different
    implementations: analysis_engine.py's adds a "unit" field and uses
    nested-aware score extraction; prediction_engine.py's is nested-aware
    without a unit field; conclusion_engine.py's uses the SIMPLE (non-nested)
    score extraction without a unit field. Merging these would change output
    for at least one caller — left as three separate, deliberately-diverged
    functions.
  - `_extract_scores` here is the NESTED-AWARE variant (checks
    attempts/sections/subItems before falling back to a flat score field),
    used only by analysis_engine.py and prediction_engine.py.
    conclusion_engine.py and suggested_question_engine.py each keep their own
    SIMPLE (non-nested) `_extract_scores` — merging them into this version
    would newly make those two engines pick up nested attempt scores they
    currently ignore, a real behavior change.
  - conclusion_engine.py's own trend detection (`_detect_trend`, linear
    regression) and prediction_engine.py's own trend detection (`_momentum`,
    half-split average delta) are two different algorithms answering a
    similar question — not consolidated here; see the refactor plan's
    Tier 2 for why that's a bigger, separately-scoped decision.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from utils import get_score as _get_score
from utils import safe_float as _safe_float

# ── Tunable thresholds ───────────────────────────────────────────────────────
# Shared by analysis_engine.py and prediction_engine.py. conclusion_engine.py
# and suggested_question_engine.py still use their own inline 50/75 literals
# in most places — left alone (see refactor plan) except where explicitly
# migrated.
LOW_SCORE_THRESHOLD: float = 50.0
HIGH_SCORE_THRESHOLD: float = 75.0


# ── Named record helpers ──────────────────────────────────────────────────────

_NAME_FIELDS = ("fullName", "name", "agniveerName", "recruiterName")
_ID_FIELDS = ("agniveerNo", "agniveerNumber", "enrollmentNo")


def _record_label(record: Dict[str, Any]) -> Optional[str]:
    for f in _NAME_FIELDS:
        if record.get(f):
            return str(record[f])
    for f in _ID_FIELDS:
        if record.get(f):
            return str(record[f])
    return None


# ── Statistical helpers ────────────────────────────────────────────────────────


def _percentile(sorted_data: List[float], pct: float) -> float:
    """Return the pct-th percentile (0-100) of a pre-sorted list."""
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    idx = (pct / 100) * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return round(sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo]), 2)


def _extract_nested_scores(record: Dict[str, Any]) -> List[float]:
    """
    Recursively pull scores from the .NET nested structure:
      attempts -> sections -> subItems -> grading -> score
    Falls back to top-level score via get_score().
    """
    collected: List[float] = []
    top = _get_score(record)
    if top is not None:
        collected.append(top)

    for attempt in record.get("attempts") or []:
        for section in attempt.get("sections") or []:
            s = _safe_float(section.get("score") or section.get("totalScore"))
            if s is not None:
                collected.append(s)
            for sub in section.get("subItems") or []:
                sub_s = _safe_float(sub.get("score") or sub.get("marksObtained"))
                if sub_s is not None:
                    collected.append(sub_s)
    return collected


def _extract_scores(records: List[Any], preserve_order: bool = True) -> List[float]:
    """
    Extract scores from a list of records, checking nested attempt structures
    first and falling back to a flat score field (see _extract_nested_scores).
    preserve_order=True  -> encounter order (for momentum / trend use)
    preserve_order=False -> sorted ascending (for percentile / stats use)

    NOT used by conclusion_engine.py / suggested_question_engine.py — they
    keep their own simpler, non-nested-aware variant (see module docstring).
    """
    scores: List[float] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        nested = _extract_nested_scores(r)
        if nested:
            scores.append(max(nested))
        else:
            s = _get_score(r)
            if s is not None:
                scores.append(s)
    if not preserve_order:
        scores.sort()
    return scores
