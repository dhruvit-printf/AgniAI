"""
conclusion_engine.py
====================
Generates a concise conclusion with at most three bullet points.
Pure Python — no LLM dependency.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

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
                lines.append(f"{k} difference is {v.get('difference')} and percentage is {v.get('percentage')}.")
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
    answer: Dict[str, Any],
    query_type: str,
    intent: Dict[str, Any],
    trace_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate a short conclusion with at most three bullet points.
    Pure Python — no LLM calls.  Never raises.
    """
    try:
        sections = answer.get("sections") or []
        is_empty = True
        if query_type in ("compare", "comparison"):
            left_data = answer.get("left", {}).get("data") or []
            right_data = answer.get("right", {}).get("data") or []
            if left_data or right_data:
                is_empty = False
        else:
            for sec in sections:
                if sec.get("data"):
                    is_empty = False
                    break

        category = intent.get("category") or "Agniveer"

        if is_empty:
            if query_type == "cross_filter":
                msg = "The cross-filter search did not return any matching records. You may want to broaden the criteria and try again."
            elif query_type in ("compare", "comparison"):
                msg = "The side-by-side comparison could not be completed because no matching records were found for either side."
            elif query_type == "multi_independent":
                msg = "The consolidated report is empty because no matching records were found across the requested sections."
            else:
                msg = f"The search returned zero matching {category.lower()} records. Try broadening the criteria and search again."
            return {"summary": msg, "bullets": [msg[:120]]}

        # ── Pure Python conclusion generation (max 3 bullets) ────────────
        bullets: List[str] = []

        if query_type in ("compare", "comparison"):
            left = answer.get("left") or {}
            right = answer.get("right") or {}
            left_label = left.get("label", "Side 1")
            right_label = right.get("label", "Side 2")
            left_cnt = len(left.get("data", []))
            right_cnt = len(right.get("data", []))
            bullets.append(f"Comparison of {left_label} ({left_cnt} records) vs {right_label} ({right_cnt} records) is complete.")
            left_scores = [s for s in (_get_score(r) for r in (left.get("data") or []) if isinstance(r, dict)) if s is not None]
            right_scores = [s for s in (_get_score(r) for r in (right.get("data") or []) if isinstance(r, dict)) if s is not None]
            if left_scores and right_scores:
                l_avg = round(sum(left_scores) / len(left_scores), 2)
                r_avg = round(sum(right_scores) / len(right_scores), 2)
                bullets.append(f"{left_label} average: {l_avg}, {right_label} average: {r_avg}.")
            summary = f"The comparative review of {left_label} and {right_label} is complete."

        elif query_type == "cross_filter":
            records = sections[0].get("data") if sections else []
            cnt = len(records)
            bullets.append(f"Cross-filter query matched {cnt} records satisfying all conditions.")
            scores = [s for s in (_get_score(r) for r in records if isinstance(r, dict)) if s is not None]
            if scores:
                bullets.append(f"Average score of matched records: {round(sum(scores) / len(scores), 2)}.")
            summary = f"Cross-filter query isolated {cnt} matching records."

        elif query_type == "multi_independent":
            bullets.append(f"Consolidated report covers {len(sections)} independent sections.")
            for sec in sections[:2]:
                label = sec.get("label", "Section")
                cnt = len(sec.get("data", []))
                bullets.append(f"{label}: {cnt} records.")
            summary = f"Consolidation of {len(sections)} independent modules is complete."

        else:
            # simple / trend / distribution
            records = sections[0].get("data", []) if sections else []
            cnt = len(records)
            scores = [s for s in (_get_score(r) for r in records if isinstance(r, dict)) if s is not None]
            bullets.append(f"Query returned {cnt} {category.lower()} records.")
            if scores:
                avg = round(sum(scores) / len(scores), 2)
                bullets.append(f"Average score: {avg} (range: {round(min(scores), 2)} to {round(max(scores), 2)}).")
            summary = f"The {category.lower()} query returned {cnt} records and is ready for review."

        # Cap at 3 bullets
        bullets = bullets[:3]
        return {"summary": summary, "bullets": bullets}

    except Exception as exc:
        logger.warning("conclusion_engine.generate_conclusion failed: %s", exc, exc_info=True)
        category = intent.get("category") or "Agniveer"
        fallback = f"The review of {category.lower()} records is complete."
        return {"summary": fallback, "bullets": [fallback]}
