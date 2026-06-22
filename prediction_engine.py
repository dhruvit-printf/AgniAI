"""
prediction_engine.py
====================
Generates data-grounded predictions (short term and future trends).
"""

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from grounding_utils import extract_numbers_from_text as _extract_numbers_from_text
from grounding_utils import ground_and_sanitize as _ground_and_sanitize
from utils import get_score as _get_score
from utils import safe_float as _safe_float

def _build_prediction_grounding_text(answer: Dict[str, Any], query_type: str) -> str:
    lines = []
    sections = answer.get("sections") or []

    if query_type == "compare":
        left = answer.get("left") or {}
        right = answer.get("right") or {}
        comp = answer.get("comparison") or {}
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
        records = sections[0].get("data") if sections else []
        lines.append(f"Record Count: {len(records)}")
        scores = []
        for r in records:
            for score_field in ("bestTotal", "totalMarks", "score", "Score", "omrInputTotal", "marksObtained"):
                v = _safe_float(r.get(score_field))
                if v is not None:
                    scores.append(v)
                    break
        if scores:
            lines.append(f"Average Score: {round(sum(scores) / len(scores), 2)}")
            lines.append(f"Top Score: {max(scores)}")
            lines.append(f"Bottom Score: {min(scores)}")

    return "\n".join(lines)

def generate_predictions(
    answer: Dict[str, Any],
    query_type: str,
    intent: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Generate shortTerm and futureTrends predictions from JSON data.
    """
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
        trend_val = "Insufficient Data"
        if query_type == "cross_filter":
            proj = "Given the current database state, subsequent runs of this specific cross-filter query are expected to continue yielding zero matches unless the underlying records or the filter conditions are updated."
            trends = [
                "Subsequent queries with the same criteria will likely yield zero matches.",
                "Adjusting filter criteria will be necessary to obtain non-empty result sets."
            ]
        elif query_type in ("compare", "comparison"):
            proj = "Future comparisons will remain unavailable until records are successfully logged for the target categories in the database."
            trends = [
                "Comparisons will continue to show no data until records are populated.",
                "Verifying database connection and record ingestion is recommended."
            ]
        elif query_type == "multi_independent":
            proj = "The consolidated report will continue to show empty sections in subsequent runs unless data is populated in the source modules."
            trends = [
                "Report sections will remain empty until data ingestion occurs.",
                "Query parameters should be verified to confirm they match existing records."
            ]
        else:
            proj = f"No future trend projection can be generated since there is no historical or current {category.lower()} data available in the database at this time."
            trends = [
                "Projections will remain unavailable until active records are populated.",
                "Ensure record logging is functional before planning trend analysis."
            ]
        return {
            "trend": trend_val,
            "projection": proj,
            "heuristicEstimate": proj,
            "shortTerm": "stable",
            "futureTrends": trends,
        }

    grounding_text = _build_prediction_grounding_text(answer, query_type)
    grounded_numbers = _extract_numbers_from_text(grounding_text)

    # 1. Determine shortTerm direction (stable, increasing, decreasing)
    short_term = "stable"
    future_trends = []

    if query_type == "trend":
        # Check sections for trend metadata (or if it exists)
        # If we have chartData in answer, we can compute trend
        records = sections[0].get("data") if sections else []
        # If there is trend direction in answer
        if "trendDirection" in answer:
            short_term = answer["trendDirection"]
        else:
            short_term = "stable"
        
        future_trends.append(f"The short-term trend is projected as {short_term} based on historical metrics.")

    elif query_type in ("compare", "comparison"):
        comp = answer.get("comparison") or {}
        # if comparison difference is high
        diff = _safe_float(comp.get("difference"))
        if diff and diff > 10:
            short_term = "increasing" if comp.get("higher") == "Side 1" else "decreasing"
        else:
            short_term = "stable"

        left_label = answer.get("left", {}).get("label", "Side 1")
        right_label = answer.get("right", {}).get("label", "Side 2")
        future_trends.append(f"Based on the side-by-side evaluation, the current performance differences observed between {left_label} and {right_label} are projected to persist in future training cycles unless targeted training interventions are implemented.")

    elif query_type == "cross_filter":
        records = sections[0].get("data") if sections else []
        short_term = "stable"
        future_trends.append(f"Based on the current intersection results, subsequent runs of this cross-filter query are highly likely to return a matching count of approximately {len(records)} records, assuming the criteria and the underlying dataset remain stable.")

    elif query_type == "multi_independent":
        short_term = "stable"
        future_trends.append("The consolidated metrics from the multiple independent modules are projected to follow current baseline trends. Each section is expected to maintain its current performance level, with no immediate correlation expected between the independent categories.")

    else:
        # simple / other
        records = sections[0].get("data") if sections else []
        scores = []
        for r in records:
            score = _get_score(r)
            if score is not None:
                scores.append(score)

        if scores:
            avg_score = round(sum(scores) / len(scores), 2)
            if avg_score > 75:
                short_term = "stable"
                future_trends.append(f"Future scores are projected to remain high and stable around the average of {avg_score}.")
            elif avg_score < 50:
                short_term = "decreasing"
                future_trends.append(f"Trainees averaging {avg_score} are projected to require extra training to improve standard scores.")
            else:
                short_term = "stable"
                future_trends.append(f"Performance is expected to continue near the average score of {avg_score}.")
        else:
            short_term = "stable"
            future_trends.append(f"The {category.lower()} records count of {len(records)} is expected to remain stable.")

    # Sanitize trends against grounding numbers to ensure no hallucinations
    sanitized_trends = []
    for trend in future_trends:
        san = _ground_and_sanitize(trend, grounding_text)
        if san:
            sanitized_trends.append(san)
        else:
            # fallback that doesn't contain ungrounded numbers
            sanitized_trends.append(f"Metrics for {category.lower()} are expected to align with historical baseline.")

    # Strict enum check for short_term (examples: Stable, Increasing, Decreasing, Insufficient Data)
    trend_val = "Stable"
    if is_empty:
        trend_val = "Insufficient Data"
    else:
        st_lower = str(short_term).lower()
        if "increase" in st_lower or "up" in st_lower:
            trend_val = "Increasing"
        elif "decrease" in st_lower or "down" in st_lower:
            trend_val = "Decreasing"
        elif "insufficient" in st_lower or "no data" in st_lower:
            trend_val = "Insufficient Data"

    projection_text = sanitized_trends[0] if sanitized_trends else "Metrics are expected to align with historical standards."
    heuristic_text = projection_text

    return {
        "trend": trend_val,
        "projection": projection_text,
        "heuristicEstimate": heuristic_text,
        "shortTerm": trend_val.lower() if trend_val != "Insufficient Data" else "stable",
        "futureTrends": sanitized_trends[:3],
    }

def _extract_records_from_combined(data: Any) -> List[Dict]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "Data", "result", "Result", "records", "Records", "persons", "personnel"):
            val = data.get(key)
            if isinstance(val, list):
                return val
    return []

def generate_rule_based_predictions(
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
    aggregate_text: str,
) -> List[str]:
    predictions = []
    grounded_numbers = _extract_numbers_from_text(aggregate_text)

    if query_type == "cross_filter":
        match_count = (
            combined_result.get("matchCount", 0)
            if isinstance(combined_result, dict)
            else 0
        )
        total_before = (
            combined_result.get("totalBeforeFilter", 0)
            if isinstance(combined_result, dict)
            else 0
        )
        if total_before > 0 and match_count > 0:
            pct = round((match_count / total_before) * 100, 1)
            pct_str = f"{pct}"
            if pct_str in grounded_numbers:
                predictions.append(
                    f"If the current match rate of {pct}% holds, future cross-filters are expected to return a similar proportion of records."
                )
        if not predictions:
            predictions.append(
                "Tighter cross-filtering criteria are expected to reduce the match count in future query runs."
            )

    elif query_type == "comparison":
        sides = (
            combined_result.get("sides", [])
            if isinstance(combined_result, dict)
            else []
        )
        if len(sides) >= 2:
            label1 = sides[0].get("label", "Side 1")
            label2 = sides[1].get("label", "Side 2")
            predictions.append(
                f"Performance or status variations between {label1} and {label2} are expected to persist in subsequent evaluations."
            )
        else:
            predictions.append(
                "Future comparison cycles are expected to show similar variances across the selected metrics."
            )

    elif query_type == "multi_independent":
        predictions.append(
            "Consolidated section metrics are projected to follow current trends across all monitored categories."
        )

    else:
        # simple
        records = _extract_records_from_combined(combined_result)
        cnt = len(records)
        cnt_str = f"{cnt}"
        category = intent.get("category") or "Agniveer"

        scores = []
        if not scores:
            for r in records:
                for score_field in ("bestTotal", "totalMarks", "score", "Score", "omrInputTotal", "marksObtained"):
                    v = _safe_float(r.get(score_field))
                    if v is not None:
                        scores.append(v)
                        break

        if scores:
            avg_score = round(sum(scores) / len(scores), 2)
            avg_str = f"{avg_score}"
            if avg_str in grounded_numbers:
                predictions.append(
                    f"Based on the average score of {avg_score}, future group performance is projected to remain stable."
                )

        if cnt > 0 and cnt_str in grounded_numbers:
            predictions.append(
                f"With {cnt} active records detected, subsequent evaluations will likely require comparable tracking resources."
            )

        if not predictions:
            predictions.append(
                f"The {category.lower()} metrics are expected to align with historical standards in the next evaluation."
            )

    grounded_predictions = []
    for pred in predictions:
        san = _ground_and_sanitize(pred, aggregate_text)
        if san:
            grounded_predictions.append(san)

    return grounded_predictions[:3]
