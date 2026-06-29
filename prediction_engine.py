"""
prediction_engine.py
====================
Generates data-grounded predictions (short term and future trends).
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from grounding_utils import ground_and_sanitize as _ground_and_sanitize
from utils import get_score as _get_score
from utils import safe_float as _safe_float


def _build_prediction_grounding_text(combined_result: Any, query_type: str) -> str:
    lines = []
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

    from normalized_models import extract_records as _extract_records

    records = _extract_records(combined_result)
    if not sections and not (left or right):
        sections = [{"label": "Result", "data": records}]

    if query_type == "compare":
        if left:
            for k, v in left.get("metrics", {}).items():
                lines.append(f"{k}: {v}")
        if right:
            for k, v in right.get("metrics", {}).items():
                lines.append(f"{k}: {v}")
        for k, v in comp.items():
            if isinstance(v, dict):
                lines.append(f"{k} difference: {v.get('difference')}")
    else:
        target_records = sections[0].get("data") if sections else []
        lines.append(f"Record Count: {len(target_records)}")
        scores = []
        for r in target_records:
            for score_field in (
                "bestTotal",
                "totalMarks",
                "score",
                "Score",
                "omrInputTotal",
                "marksObtained",
            ):
                v = _safe_float(r.get(score_field))
                if v is not None:
                    scores.append(v)
                    break
        if scores:
            lines.append(f"Average Score: {round(sum(scores) / len(scores), 2)}")
            lines.append(f"Top Score: {max(scores)}")
            lines.append(f"Bottom Score: {min(scores)}")

    return "\n".join(lines)


def _build_fallback_prediction(
    combined_result: Any, query_type: str, intent: Dict[str, Any]
) -> Dict[str, Any]:
    category = intent.get("category") or "Agniveer"
    from normalized_models import extract_records as _extract_records

    records = _extract_records(combined_result)
    record_count = len(records)

    if record_count <= 0:
        projection = f"Future projection is unavailable because no {category.lower()} records were returned."
        trend = "Insufficient Data"
        return {
            "trend": trend,
            "projection": projection,
            "heuristicEstimate": "Unavailable",
            "shortTerm": "insufficient data",
            "futureTrends": [],
        }
    else:
        projection = f"Performance metrics for {category.lower()} are projected to remain stable and consistent with current baselines."
        trend = "Stable"

    return {
        "trend": trend,
        "projection": projection,
        "heuristicEstimate": projection,
        "shortTerm": trend.lower(),
        "futureTrends": [projection],
    }


def generate_predictions(
    combined_result: Any, query_type: str, intent: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Generate shortTerm and futureTrends predictions from JSON data.
    """
    from normalized_models import extract_records as _extract_records

    records = _extract_records(combined_result)

    if isinstance(combined_result, dict):
        sections = combined_result.get("sections") or []
        left = combined_result.get("left") or {}
        right = combined_result.get("right") or {}
        comp = combined_result.get("comparison") or {}
        trend_direction = combined_result.get("trendDirection")
    else:
        sections = []
        left = {}
        right = {}
        comp = {}
        trend_direction = None

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
        return _build_fallback_prediction(combined_result, query_type, intent)

    grounding_text = _build_prediction_grounding_text(combined_result, query_type)
    total_points = len(records)

    # 1. Determine shortTerm direction (stable, increasing, decreasing)
    short_term = "stable"
    future_trends = []

    if query_type == "trend":
        if trend_direction:
            short_term = trend_direction
        else:
            short_term = "stable"
        future_trends.append(
            f"The short-term trend is projected as {short_term} based on historical metrics."
        )

    elif query_type in ("compare", "comparison"):
        left_label = left.get("label", "Side 1")
        right_label = right.get("label", "Side 2")
        if comp and ("difference" in comp or "higher" in comp):
            first_metric = comp
        else:
            first_metric = next(iter(comp.values()), {}) if comp else {}
        diff = None
        higher_label = None
        if isinstance(first_metric, dict):
            diff = _safe_float(first_metric.get("difference"))
            higher_label = first_metric.get("higher")
        elif isinstance(first_metric, (int, float)):
            diff = float(first_metric)

        if diff and diff > 10:
            short_term = "increasing" if higher_label == left_label else "decreasing"
        else:
            short_term = "stable"

        future_trends.append(
            f"Performance differences observed between {left_label} and {right_label} are projected to persist in future training cycles."
        )

    elif query_type == "cross_filter":
        short_term = "stable"
        future_trends.append(
            f"Subsequent runs of this cross-filter query are highly likely to return a matching count of approximately {total_points} records."
        )

    elif query_type == "multi_independent":
        short_term = "stable"
        future_trends.append(
            "Each section is expected to maintain its current performance level, with no immediate correlation expected between independent categories."
        )

    else:
        # simple / other
        target_records = sections[0].get("data") if sections else []
        scores = []
        for r in target_records:
            score = _get_score(r)
            if score is not None:
                scores.append(score)

        if scores:
            avg_score = round(sum(scores) / len(scores), 2)
            if avg_score > 75:
                short_term = "stable"
                future_trends.append(
                    f"Future scores are projected to remain stable around the average of {avg_score}."
                )
            elif avg_score < 50:
                short_term = "decreasing"
                future_trends.append(
                    f"Trainees averaging {avg_score} are projected to require extra training to improve standard scores."
                )
            else:
                short_term = "stable"
                future_trends.append(
                    f"Performance is expected to continue near the average score of {avg_score}."
                )
        else:
            short_term = "stable"
            future_trends.append(
                f"The {category.lower()} records count of {total_points} is expected to remain stable."
            )

    # Sanitize trends against grounding numbers to ensure no hallucinations
    sanitized_trends = []
    for trend in future_trends:
        san = _ground_and_sanitize(trend, grounding_text)
        if san:
            sanitized_trends.append(san)
        else:
            sanitized_trends.append(
                f"Metrics for {category.lower()} are expected to align with historical baseline."
            )

    # Strict enum check for short_term (examples: Stable, Increasing, Decreasing)
    trend_val = "Stable"
    st_lower = str(short_term).lower()
    if "increas" in st_lower or "up" in st_lower:
        trend_val = "Increasing"
    elif "decreas" in st_lower or "down" in st_lower:
        trend_val = "Decreasing"

    projection_text = f"Performance metrics for {category.lower()} are projected to remain {trend_val.lower()} and consistent with current baselines."
    heuristic_text = projection_text

    return {
        "trend": trend_val,
        "projection": projection_text,
        "heuristicEstimate": heuristic_text,
        "shortTerm": trend_val.lower(),
        "futureTrends": sanitized_trends[:3],
    }
