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


def _build_aggregate_text(answer: Dict[str, Any], query_type: str, intent: Dict[str, Any]) -> str:
    lines = []
    category = intent.get("category") or "Agniveer"
    sections = answer.get("sections") or []

    if query_type == "cross_filter":
        records = sections[0].get("data") if sections else []
        lines.append(f"Query Type: cross_filter")
        lines.append(f"Match Count: {len(records)}")
        names = [r.get("fullName") or r.get("name") for r in records if r.get("fullName") or r.get("name")]
        if names:
            lines.append(f"Matched Agniveers: {', '.join(names)}")
    elif query_type == "compare":
        left = answer.get("left") or {}
        right = answer.get("right") or {}
        comp = answer.get("comparison") or {}
        lines.append(f"Query Type: comparison")
        if left:
            lines.append(f"Side 1: {left.get('label')} - Count: {len(left.get('data', []))}")
            for k, v in left.get("metrics", {}).items():
                lines.append(f"  Side 1 {k}: {v}")
        if right:
            lines.append(f"Side 2: {right.get('label')} - Count: {len(right.get('data', []))}")
            for k, v in right.get("metrics", {}).items():
                lines.append(f"  Side 2 {k}: {v}")
        for k, v in comp.items():
            if isinstance(v, dict):
                lines.append(f"  Comparison {k}: difference={v.get('difference')}, percentage={v.get('percentage')}, higher={v.get('higher')}, lower={v.get('lower')}")
    elif query_type == "multi_independent":
        lines.append(f"Query Type: multi_independent")
        lines.append(f"Section Count: {len(sections)}")
        for sec in sections:
            lines.append(f"  Section: {sec.get('label')} - {len(sec.get('data', []))} records")
    else:
        # simple/trend/distribution
        records = sections[0].get("data") if sections else []
        lines.append(f"Query Type: simple")
        lines.append(f"Category: {category}")
        lines.append(f"Record Count: {len(records)}")
        scores = []
        for r in records:
            score = _get_score(r)
            if score is not None:
                scores.append(score)
        if scores:
            lines.append(f"Average Score: {round(sum(scores) / len(scores), 2)}")
            lines.append(f"Top Score: {max(scores)}")
            lines.append(f"Bottom Score: {min(scores)}")
        names = [r.get("fullName") or r.get("name") for r in records if r.get("fullName") or r.get("name")]
        if names:
            lines.append(f"Records: {', '.join(names[:20])}")
            if len(names) > 20:
                lines.append(f"...and {len(names) - 20} more")

    return "\n".join(lines)


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
    answer: Dict[str, Any],
    query_type: str,
    intent: Dict[str, Any],
    user_query: str = "",
    trace_id: Optional[str] = None
) -> Dict[str, Any]:
    """
    Generate observations and insights from JSON answer using pure Python
    statistics.  No LLM calls.  Never raises — returns a safe fallback on
    any internal error.
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
                return _analysis_payload(
                    "A detailed cross-filter query was executed across multiple datasets to identify common personnel matching all filter parameters. The resulting intersection yielded zero common records, demonstrating that the overlapping conditions specified in the query filter out all available Agniveers in the system database.",
                    [
                        "A cross-filter search was performed across all selected category sets, but no overlapping records were retrieved."
                    ],
                    {"record_count": 0, "match_count": 0},
                )
            elif query_type in ("compare", "comparison"):
                return _analysis_payload(
                    "A side-by-side comparison was initiated to evaluate the metrics of the selected categories. However, since the query returned no records for any of the groups, a comparative breakdown cannot be compiled. There is no active data to compare across the selected dimensions.",
                    [
                        "Both comparison groups returned empty datasets from the primary database query."
                    ],
                    {"left_count": 0, "right_count": 0},
                )
            elif query_type == "multi_independent":
                return _analysis_payload(
                    "The multi-section consolidation process compiled results from all requested data modules. Unfortunately, none of the query paths returned any active records, meaning that all sections in this report are currently empty and there are no statistics available for analysis.",
                    [
                        "All requested independent sections returned zero active records from the system."
                    ],
                    {"section_count": len(sections)},
                )
            else:
                return _analysis_payload(
                    f"No matching data was found for the requested {category.lower()} query filter criteria. The database search completed successfully but returned an empty dataset, meaning there are no records available to display or analyze at this time.",
                    [
                        f"The search returned 0 records matching the {category.lower()} query parameters."
                    ],
                    {"record_count": 0},
                )

        # ── Pure Python statistics (no LLM) ──────────────────────────────
        stats: Dict[str, Any] = {}
        summary = f"Summary of {category.lower()} metrics completed."
        insights: List[str] = ["Dataset matches the specified parameters."]

        if query_type in ("compare", "comparison"):
            left = answer.get("left") or {}
            right = answer.get("right") or {}
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
            }
        elif query_type == "cross_filter":
            records = sections[0].get("data") if sections else []
            summary = f"Cross-filter analysis matched {len(records)} records after intersecting the requested conditions."
            insights = [f"{len(records)} records satisfy all overlapping filter conditions."]
            stats = {"match_count": len(records)}
        elif query_type == "multi_independent":
            summary = f"Consolidated data from {len(sections)} independent sections."
            insights = ["Sections are presented independently without correlation."]
            section_details = {}
            for sec in sections:
                label = sec.get("label", "Section")
                count = len(sec.get("data", []))
                section_details[label] = count
            stats = {"section_count": len(sections), "sections": section_details}
        else:
            # simple / trend / distribution
            records = sections[0].get("data", []) if sections else []
            scores = [s for s in (_get_score(r) for r in records if isinstance(r, dict)) if s is not None]
            stats = {"record_count": len(records)}
            if scores:
                import statistics as _stats_mod
                avg_score = round(sum(scores) / len(scores), 2)
                min_score = round(min(scores), 2)
                max_score = round(max(scores), 2)
                std_dev = round(_stats_mod.pstdev(scores), 2) if len(scores) > 1 else 0.0
                summary = (
                    f"Matched {len(records)} {category.lower()} records with an average score of {avg_score}, "
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
                summary = f"Matched {len(records)} {category.lower()} records from the returned JSON payload."
                insights = [f"The query returned {len(records)} {category.lower()} records."]

        return _analysis_payload(summary, insights, stats)

    except Exception as exc:
        logger.warning("analysis_engine.generate_analysis failed: %s", exc, exc_info=True)
        category = intent.get("category") or "Agniveer"
        return _analysis_payload(
            f"Analysis of {category.lower()} records completed with limited metrics.",
            ["Dataset matches the specified parameters."],
            {},
        )

