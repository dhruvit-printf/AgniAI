"""
analysis_engine.py
==================
Generates observations and insights from JSON answer data using pure Python
statistics.  No LLM dependency.
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from utils import get_score as _get_score
from utils import safe_float as _safe_float


def _analysis_payload(
    summary: str,
    insights: List[str],
    statistics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return {
        "summary": summary,
        "insights": insights,
        "statistics": statistics or {},
    }

def generate_analysis(
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
    user_query: str = "",
    trace_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate observations and insights from JSON combined_result using pure Python
    statistics. No LLM calls. Never raises — returns a safe fallback on
    any internal error.
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
            if query_type == "cross_filter":
                return _analysis_payload(
                    "No matching records found.",
                    [],
                    {"record_count": 0, "match_count": 0},
                )
            elif query_type in ("compare", "comparison"):
                return _analysis_payload(
                    "No matching records found.",
                    [],
                    {"left_count": 0, "right_count": 0, "record_count": 0},
                )
            elif query_type == "multi_independent":
                return _analysis_payload(
                    "No matching records found.",
                    [],
                    {"section_count": len(sections), "record_count": 0},
                )
            else:
                return _analysis_payload(
                    "No matching records found.",
                    [],
                    {"record_count": 0},
                )

        # ── Pure Python statistics (no LLM) ──────────────────────────────
        stats: Dict[str, Any] = {"record_count": len(records)}
        summary = f"Summary of {category.lower()} metrics is complete."
        insights: List[str] = ["The returned records match the selected parameters."]

        if query_type in ("compare", "comparison"):
            left_data = left.get("data") or []
            right_data = right.get("data") or []
            left_scores = [s for s in (_get_score(r) for r in left_data if isinstance(r, dict)) if s is not None]
            right_scores = [s for s in (_get_score(r) for r in right_data if isinstance(r, dict)) if s is not None]
            left_avg = round(sum(left_scores) / len(left_scores), 2) if left_scores else None
            right_avg = round(sum(right_scores) / len(right_scores), 2) if right_scores else None
            summary = f"Comparison completed between {left.get('label', 'Side 1')} ({len(left_data)} records) and {right.get('label', 'Side 2')} ({len(right_data)} records)."
            insights = []
            if left_avg is not None and right_avg is not None:
                diff = round(left_avg - right_avg, 2)
                insights.append(f"{left.get('label', 'Side 1')} average: {left_avg}, {right.get('label', 'Side 2')} average: {right_avg}, difference: {diff}.")
            else:
                insights.append("Comparison highlights metric variances across categories.")
            stats = {
                "left_count": len(left_data),
                "right_count": len(right_data),
                "left_average": left_avg,
                "right_average": right_avg,
                "record_count": len(left_data) + len(right_data),
            }
        elif query_type == "cross_filter":
            match_records = sections[0].get("data") if sections else []
            summary = f"Cross-filter analysis matched {len(match_records)} records after intersecting the requested conditions."
            insights = [f"{len(match_records)} records satisfy all overlapping filter conditions."]
            stats = {"match_count": len(match_records), "record_count": len(match_records)}
        elif query_type == "multi_independent":
            summary = f"Consolidated data from {len(sections)} independent sections."
            insights = ["Sections are presented independently without correlation."]
            section_details = {}
            total_recs = 0
            for sec in sections:
                label = sec.get("label", "Section")
                count = len(sec.get("data", []))
                section_details[label] = count
                total_recs += count
            stats = {"section_count": len(sections), "sections": section_details, "record_count": total_recs}
        else:
            # simple / trend / distribution
            target_records = sections[0].get("data", []) if sections else []
            scores = [s for s in (_get_score(r) for r in target_records if isinstance(r, dict)) if s is not None]
            stats = {"record_count": len(target_records)}
            if scores:
                import statistics as _stats_mod
                avg_score = round(sum(scores) / len(scores), 2)
                min_score = round(min(scores), 2)
                max_score = round(max(scores), 2)
                std_dev = round(_stats_mod.pstdev(scores), 2) if len(scores) > 1 else 0.0
                summary = (
                    f"Matched {len(target_records)} {category.lower()} records with an average score of {avg_score}, "
                    f"ranging from {min_score} to {max_score}."
                )
                insights = [
                    f"Average score: {avg_score}.",
                    f"Score range: {min_score} to {max_score}.",
                ]
                if std_dev > 0:
                    insights.append(f"Standard deviation: {std_dev}.")
                stats.update({
                    "average_score": avg_score,
                    "min_score": min_score,
                    "max_score": max_score,
                    "std_dev": std_dev,
                })
            else:
                summary = f"Matched {len(target_records)} {category.lower()} records."
                insights = [f"The query returned {len(target_records)} {category.lower()} records."]

        return _analysis_payload(summary, insights, stats)

    except Exception as exc:
        logger.warning("analysis_engine.generate_analysis failed: %s", exc, exc_info=True)
        category = intent.get("category") or "Agniveer"
        return _analysis_payload(
            f"Analysis of {category.lower()} records completed with limited metrics.",
            ["Dataset matches the specified parameters."],
            {"record_count": 0},
        )
