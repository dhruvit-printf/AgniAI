"""
conclusion_engine.py
====================
Generates a concise conclusion with at most three bullet points.
Pure Python — no LLM dependency.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from feature_flags import get_flags
from grounding_utils import ground_and_sanitize as _ground_and_sanitize
from utils import get_score as _get_score
from utils import safe_float as _safe_float


def _build_conclusion_grounding_text(answer: Dict[str, Any], query_type: str) -> str:
    lines = []
    sections = answer.get("sections") or []

    if query_type == "compare":
        left = answer.get("left") or {}
        right = answer.get("right") or {}
        comp = answer.get("comparison") or {}
        if left:
            lines.append(f"{left.get('label')} count is {len(left.get('data', []))}.")
        if right:
            lines.append(f"{right.get('label')} count is {len(right.get('data', []))}.")
        for k, v in comp.items():
            if isinstance(v, dict):
                lines.append(
                    f"{k} difference is {v.get('difference')} and percentage is {v.get('percentage')}."
                )
    else:
        records = sections[0].get("data") if sections else []
        lines.append(f"Count: {len(records)}")
        scores = []
        for r in records:
            score = _get_score(r)
            if score is not None:
                scores.append(score)
        if scores:
            lines.append(f"Average Score: {round(sum(scores) / len(scores), 2)}")
            lines.append(f"Top Score: {max(scores)}")
            lines.append(f"Bottom Score: {min(scores)}")

    return "\n".join(lines)


def generate_conclusion(
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate a short conclusion with at most three bullet points.
    Pure Python — no LLM calls. Never raises.
    """
    try:
        from normalized_models import extract_records as _extract_records

        records = _extract_records(combined_result)

        if isinstance(combined_result, dict):
            sections = combined_result.get("sections") or []
            left = combined_result.get("left") or {}
            right = combined_result.get("right") or {}
            comp = combined_result.get("comparison") or {}
        else:
            sections = []
            left = {}
            right = {}
            comp = {}

        if not sections and not (left or right):
            sections = [{"label": "Result", "data": records}]

        is_empty = True
        if query_type in ("compare", "comparison"):
            left_data = left.get("data") or []
            right_data = right.get("data") or []
            if left_data or right_data:
                is_empty = False
        else:
            for sec in sections:
                if sec.get("data"):
                    is_empty = False
                    break

        category = intent.get("category") or "Agniveer"

        if is_empty:
            return {"summary": "No matching records found.", "bullets": []}

        # ── Pure Python conclusion generation (max 3 bullets) ────────────
        bullets: List[str] = []

        if query_type in ("compare", "comparison"):
            left_label = left.get("label", "Side 1")
            right_label = right.get("label", "Side 2")
            left_cnt = len(left.get("data", []))
            right_cnt = len(right.get("data", []))
            summary = (
                f"Comparative review of {left_label} and {right_label} is complete."
            )
            bullets.append(
                f"Comparison evaluated {left_cnt} records on {left_label} and {right_cnt} records on {right_label}."
            )
            left_scores = [
                s
                for s in (
                    _get_score(r)
                    for r in (left.get("data") or [])
                    if isinstance(r, dict)
                )
                if s is not None
            ]
            right_scores = [
                s
                for s in (
                    _get_score(r)
                    for r in (right.get("data") or [])
                    if isinstance(r, dict)
                )
                if s is not None
            ]
            if left_scores and right_scores:
                l_avg = round(sum(left_scores) / len(left_scores), 2)
                r_avg = round(sum(right_scores) / len(right_scores), 2)
                bullets.append(
                    f"{left_label} average: {l_avg}, {right_label} average: {r_avg}."
                )

        elif query_type == "cross_filter":
            cnt = len(records)
            summary = "Cross-filter analysis successfully completed."
            bullets.append(
                f"Found {cnt} records satisfying all combined filter conditions."
            )
            scores = [
                s
                for s in (_get_score(r) for r in records if isinstance(r, dict))
                if s is not None
            ]
            if scores:
                bullets.append(
                    f"Average score across the matched subset: {round(sum(scores) / len(scores), 2)}."
                )

        elif query_type == "multi_independent":
            summary = f"Consolidation of multi-section dataset is complete."
            bullets.append(
                f"Consolidated data covers {len(sections)} independent sections."
            )
            for sec in sections[:2]:
                label = sec.get("label", "Section")
                cnt = len(sec.get("data", []))
                bullets.append(f"{label} contains {cnt} records.")

        else:
            # simple / trend / distribution
            cnt = len(records)
            summary = f"Query lookup for {category.lower()} records is complete."
            bullets.append(f"Found {cnt} matching {category.lower()} records.")
            scores = [
                s
                for s in (_get_score(r) for r in records if isinstance(r, dict))
                if s is not None
            ]
            if scores:
                avg = round(sum(scores) / len(scores), 2)
                bullets.append(
                    f"Average score: {avg} (range: {round(min(scores), 2)} to {round(max(scores), 2)})."
                )

        # Cap at 3 bullets
        bullets = bullets[:3]
        return {"summary": summary, "bullets": bullets}

    except Exception as exc:
        logger.error(
            "conclusion_engine.generate_conclusion failed: %s", exc, exc_info=True
        )
        category = intent.get("category") or "Agniveer"
        fallback = f"The review of {category.lower()} records is complete."
        return {"summary": fallback, "bullets": [fallback]}
