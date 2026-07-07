"""
analysis_engine.py
==================
Generates observations and insights from JSON answer data using pure Python
statistics.  No LLM dependency.

FIXES vs original:
  - Null guard on combined_result (handles None gracefully)
  - _extract_scores no longer sorts — preserves encounter order for momentum
  - Handles .NET top-level "data" key when "sections" is absent
  - Handles nested Agniveer structure: attempts / sections / subItems / grading
  - score_distribution exported in stats block for frontend bucket rendering
  - per-section stats in multi_independent stats block
  - "simple" query now falls through all sections, not just sections[0]
  - Hardcoded 50/75 thresholds moved to module-level constants (easy to tune)
"""

import logging
import statistics as _stats_mod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from grounding_utils import ground_and_sanitize as _ground_and_sanitize
from normalized_models import humanize_category
from utils import categorical_breakdown as _categorical_breakdown
from utils import get_score as _get_score
from utils import safe_float as _safe_float

# ── Tunable thresholds ─────────────────────────────────────────────────────────
LOW_SCORE_THRESHOLD: float = 50.0
HIGH_SCORE_THRESHOLD: float = 75.0


# ── Statistical helpers ────────────────────────────────────────────────────────


def _percentile(sorted_data: List[float], pct: float) -> float:
    """Return the pct-th percentile (0–100) of a pre-sorted list."""
    if not sorted_data:
        return 0.0
    n = len(sorted_data)
    idx = (pct / 100) * (n - 1)
    lo, hi = int(idx), min(int(idx) + 1, n - 1)
    return round(sorted_data[lo] + (idx - lo) * (sorted_data[hi] - sorted_data[lo]), 2)


def _score_distribution(scores: List[float]) -> Dict[str, int]:
    """Bucket scores into low / mid / high using module thresholds."""
    return {
        "low_count": sum(1 for s in scores if s < LOW_SCORE_THRESHOLD),
        "mid_count": sum(
            1 for s in scores if LOW_SCORE_THRESHOLD <= s <= HIGH_SCORE_THRESHOLD
        ),
        "high_count": sum(1 for s in scores if s > HIGH_SCORE_THRESHOLD),
    }


def _extract_nested_scores(record: Dict[str, Any]) -> List[float]:
    """
    Recursively pull scores from the .NET nested structure:
      attempts → sections → subItems → grading → score
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
    Extract scores from a list of records.
    preserve_order=True  → encounter order (for momentum / trend use)
    preserve_order=False → sorted ascending (for percentile / stats use)
    """
    scores: List[float] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        # Try nested extraction first, fall back to flat get_score
        nested = _extract_nested_scores(r)
        if nested:
            # Use the best (max) score from nested attempts as the record's score
            scores.append(max(nested))
        else:
            s = _get_score(r)
            if s is not None:
                scores.append(s)
    if not preserve_order:
        scores.sort()
    return scores


# ── Named record helpers (individual-level insight) ───────────────────────────

_NAME_FIELDS = ("fullName", "name", "agniveerName", "recruiterName")
_ID_FIELDS = ("agniveerNo", "agniveerNumber", "enrollmentNo")
_UNIT_FIELDS = ("platoonName", "platoon", "company", "unit", "class", "batchName")


def _record_label(record: Dict[str, Any]) -> Optional[str]:
    for f in _NAME_FIELDS:
        if record.get(f):
            return str(record[f])
    for f in _ID_FIELDS:
        if record.get(f):
            return str(record[f])
    return None


def _record_unit(record: Dict[str, Any]) -> Optional[str]:
    for f in _UNIT_FIELDS:
        if record.get(f):
            return str(record[f])
    return None


def _named_score_records(records: List[Any]) -> List[Dict[str, Any]]:
    """Pair each record's best score with its human-readable label/unit so
    analysis can call out specific individuals, not just aggregate stats."""
    out: List[Dict[str, Any]] = []
    for r in records:
        if not isinstance(r, dict):
            continue
        label = _record_label(r)
        if not label:
            continue
        nested = _extract_nested_scores(r)
        score = max(nested) if nested else _get_score(r)
        if score is None:
            continue
        out.append({"label": label, "score": score, "unit": _record_unit(r)})
    return out


def _group_breakdown(records: List[Any], limit: int = 5) -> List[Dict[str, Any]]:
    """Aggregate scores by platoon/unit/class so the analysis reflects the
    whole dataset's structure, not just a single overall average."""
    groups: Dict[str, List[float]] = {}
    for r in records:
        if not isinstance(r, dict):
            continue
        unit = _record_unit(r)
        if not unit:
            continue
        nested = _extract_nested_scores(r)
        score = max(nested) if nested else _get_score(r)
        if score is None:
            continue
        groups.setdefault(unit, []).append(score)

    breakdown = [
        {
            "group": name,
            "count": len(scores),
            "average_score": round(sum(scores) / len(scores), 2),
        }
        for name, scores in groups.items()
    ]
    breakdown.sort(key=lambda g: (-g["average_score"], -g["count"]))
    return breakdown[:limit]


def _normalise_sections(
    combined_result: Any, records: List[Any]
) -> List[Dict[str, Any]]:
    """
    Build a canonical sections list from whatever shape the .NET response uses:
      - combined_result["sections"]  (existing AgniAI format)
      - combined_result["data"]      (flat .NET list response)
      - fallback to [{"label": "Result", "data": records}]
    """
    if not isinstance(combined_result, dict):
        return [{"label": "Result", "data": records}]

    sections = combined_result.get("sections") or []
    if sections:
        return sections

    # .NET sometimes returns top-level "data" list
    top_data = combined_result.get("data")
    if isinstance(top_data, list) and top_data:
        return [{"label": "Result", "data": top_data}]

    return [{"label": "Result", "data": records}]


def _rich_score_stats(scores: List[float]) -> Dict[str, Any]:
    """Return a comprehensive stats block for a list of scores (unsorted input OK)."""
    if not scores:
        return {}
    sorted_scores = sorted(scores)
    avg = round(sum(scores) / len(scores), 2)
    dist = _score_distribution(scores)
    above_avg = sum(1 for s in scores if s > avg)
    pct_above_avg = round((above_avg / len(scores)) * 100, 1)
    return {
        "average_score": avg,
        "min_score": round(min(scores), 2),
        "max_score": round(max(scores), 2),
        "median_score": round(_stats_mod.median(scores), 2),
        "p25_score": _percentile(sorted_scores, 25),
        "p75_score": _percentile(sorted_scores, 75),
        "std_dev": round(_stats_mod.pstdev(scores), 2) if len(scores) > 1 else 0.0,
        "above_average_count": above_avg,
        "pct_above_average": pct_above_avg,
        # Export distribution so the frontend can render bucket charts directly
        "score_distribution": dist,
        **dist,
    }


def _score_insights(
    scores: List[float], category: str, stats: Dict[str, Any]
) -> List[str]:
    """Generate data-grounded insight sentences from score stats."""
    if not scores or not stats:
        return []
    insights: List[str] = []

    avg = stats["average_score"]
    pct_above = stats["pct_above_average"]
    high_cnt = stats["high_count"]
    low_cnt = stats["low_count"]
    std_dev = stats["std_dev"]
    p25 = stats["p25_score"]
    p75 = stats["p75_score"]

    insights.append(
        f"{pct_above}% of {humanize_category(category).lower()} records scored above the average of {avg}."
    )
    insights.append(
        f"The middle 50% of scores fall between {p25} and {p75} (IQR: {round(p75 - p25, 2)})."
    )
    if high_cnt:
        insights.append(
            f"{high_cnt} record(s) scored above {HIGH_SCORE_THRESHOLD}, indicating strong performance."
        )
    if low_cnt:
        insights.append(
            f"{low_cnt} record(s) scored below {LOW_SCORE_THRESHOLD} and may need additional review."
        )
    if std_dev > 15:
        insights.append(
            f"High score variability detected (std dev: {std_dev}), suggesting inconsistent performance."
        )
    elif std_dev <= 5 and len(scores) > 1:
        insights.append(
            f"Scores are tightly clustered (std dev: {std_dev}), indicating uniform performance."
        )
    return insights


# ── Payload builder ────────────────────────────────────────────────────────────


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


# ── Public API ─────────────────────────────────────────────────────────────────


def _extract_time_context(records: List[Any]) -> Optional[str]:
    """Identify if the records represent a time-series and return a summary string."""
    if not records or not isinstance(records[0], dict):
        return None
    time_keys = ["month", "date", "week", "day", "year"]
    for r in records:
        if not isinstance(r, dict):
            continue
        for k in time_keys:
            if k in r and isinstance(r[k], str):
                vals = [rec[k] for rec in records if isinstance(rec, dict) and k in rec and isinstance(rec[k], str)]
                if len(vals) > 1:
                    return f"Trend spans {len(vals)} periods from {vals[0]} to {vals[-1]}."
                elif len(vals) == 1:
                    return f"Data point recorded for {vals[0]}."
    return None


def generate_analysis(
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
    user_query: str = "",
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate observations and insights from JSON combined_result using pure Python
    statistics. No LLM calls. Never raises — returns a safe fallback on any error.
    """
    try:
        # ── Null guard ────────────────────────────────────────────────────
        if combined_result is None:
            combined_result = {}

        from normalized_models import extract_records as _extract_records

        records = _extract_records(combined_result)

        if isinstance(combined_result, dict):
            left = combined_result.get("left") or {}
            right = combined_result.get("right") or {}
            comp = combined_result.get("comparison") or {}
        else:
            left = {}
            right = {}
            comp = {}

        sections = _normalise_sections(combined_result, records)

        # ── Empty check ───────────────────────────────────────────────────
        is_empty = True
        if query_type in ("compare", "comparison"):
            if (
                (left.get("data") or [])
                or (right.get("data") or [])
                or left.get("metrics", {}).get("recordCount", 0) > 0
                or right.get("metrics", {}).get("recordCount", 0) > 0
            ):
                is_empty = False
        else:
            for sec in sections:
                if sec.get("data"):
                    is_empty = False
                    break

        category = intent.get("category") or "Agniveer"

        if is_empty:
            base_stats: Dict[str, Any] = {"record_count": 0}
            if query_type == "cross_filter":
                base_stats["match_count"] = 0
            elif query_type in ("compare", "comparison"):
                base_stats.update({"left_count": 0, "right_count": 0})
            elif query_type == "multi_independent":
                base_stats["section_count"] = len(sections)
            return _analysis_payload("No matching records found.", [], base_stats)

        # ── Query-type branching ──────────────────────────────────────────

        # ── COMPARE ──────────────────────────────────────────────────────
        if query_type in ("compare", "comparison"):
            left_data = left.get("data") or []
            right_data = right.get("data") or []
            left_label = left.get("label", "Side 1")
            right_label = right.get("label", "Side 2")

            left_scores = _extract_scores(left_data, preserve_order=False)
            right_scores = _extract_scores(right_data, preserve_order=False)
            left_stats = _rich_score_stats(left_scores)
            right_stats = _rich_score_stats(right_scores)

            left_avg = left_stats.get("average_score")
            right_avg = right_stats.get("average_score")

            left_metrics = left.get("metrics") or {}
            right_metrics = right.get("metrics") or {}
            left_cnt = left_metrics.get("recordCount")
            if left_cnt is None:
                left_cnt = len(left_data)
            else:
                left_cnt = int(left_cnt)
            right_cnt = right_metrics.get("recordCount")
            if right_cnt is None:
                right_cnt = len(right_data)
            else:
                right_cnt = int(right_cnt)

            summary = (
                f"Comparison completed between {left_label} ({left_cnt} records) "
                f"and {right_label} ({right_cnt} records)."
            )
            insights: List[str] = []

            if left_avg is not None and right_avg is not None:
                diff = round(left_avg - right_avg, 2)
                winner = left_label if diff > 0 else right_label
                insights.append(
                    f"{left_label} average: {left_avg}, {right_label} average: {right_avg} "
                    f"(difference: {abs(diff)}). {winner} leads on average score."
                )

            if (
                left_stats.get("high_count") is not None
                and right_stats.get("high_count") is not None
            ):
                insights.append(
                    f"High scorers (>{HIGH_SCORE_THRESHOLD}): {left_label} has "
                    f"{left_stats['high_count']}, {right_label} has {right_stats['high_count']}."
                )

            if (
                left_stats.get("p25_score") is not None
                and right_stats.get("p25_score") is not None
            ):
                insights.append(
                    f"P25–P75 range: {left_label} [{left_stats['p25_score']}–{left_stats['p75_score']}], "
                    f"{right_label} [{right_stats['p25_score']}–{right_stats['p75_score']}]."
                )

            if (
                left_stats.get("std_dev") is not None
                and right_stats.get("std_dev") is not None
            ):
                insights.append(
                    f"Score variability — {left_label}: std dev {left_stats['std_dev']}, "
                    f"{right_label}: std dev {right_stats['std_dev']}."
                )

            if not insights:
                insights.append(
                    "Comparison highlights metric variances across categories."
                )

            stats: Dict[str, Any] = {
                "left_count": left_cnt,
                "right_count": right_cnt,
                "left_average": left_avg,
                "right_average": right_avg,
                "left_stats": left_stats,
                "right_stats": right_stats,
                "record_count": left_cnt + right_cnt,
            }

        # ── CROSS-FILTER ──────────────────────────────────────────────────
        elif query_type == "cross_filter":
            match_records = sections[0].get("data") if sections else []
            scores = _extract_scores(match_records, preserve_order=False)
            score_stats = _rich_score_stats(scores)

            summary = (
                f"Cross-filter analysis matched {len(match_records)} records "
                f"after intersecting the requested conditions."
            )
            insights = [
                f"{len(match_records)} records satisfy all overlapping filter conditions."
            ]
            if score_stats:
                insights.append(
                    f"Among matched records — average: {score_stats['average_score']}, "
                    f"median: {score_stats['median_score']}, "
                    f"range: {score_stats['min_score']}–{score_stats['max_score']}."
                )
                insights += _score_insights(scores, category, score_stats)[:2]

            named = _named_score_records(match_records)
            weak = sorted(
                [n for n in named if n["score"] < LOW_SCORE_THRESHOLD],
                key=lambda x: x["score"],
            )[:3]
            strong = sorted(
                [n for n in named if n["score"] > HIGH_SCORE_THRESHOLD],
                key=lambda x: -x["score"],
            )[:3]
            if weak:
                insights.append(
                    "Individuals needing attention: "
                    + ", ".join(f"{n['label']} ({n['score']})" for n in weak)
                    + "."
                )
            if strong:
                insights.append(
                    "Standout performers in this filtered set: "
                    + ", ".join(f"{n['label']} ({n['score']})" for n in strong)
                    + "."
                )

            stats = {
                "match_count": len(match_records),
                "record_count": len(match_records),
                **score_stats,
            }

        # ── MULTI-INDEPENDENT ─────────────────────────────────────────────
        elif query_type == "multi_independent":
            summary = f"Consolidated data from {len(sections)} independent sections."
            insights = ["Sections are presented independently without correlation."]
            section_details: Dict[str, Any] = {}
            total_recs = 0

            for sec in sections:
                label = sec.get("label", "Section")
                sec_records = sec.get("data") or []
                sec_scores = _extract_scores(sec_records, preserve_order=False)
                sec_stats = _rich_score_stats(sec_scores)
                sec_avg = sec_stats.get("average_score")
                section_details[label] = {
                    "count": len(sec_records),
                    "average_score": sec_avg,
                    "stats": sec_stats,  # full stats per section
                }
                total_recs += len(sec_records)
                if sec_avg is not None:
                    insights.append(
                        f"{label}: {len(sec_records)} records, average score {sec_avg}."
                    )

            stats = {
                "section_count": len(sections),
                "sections": section_details,
                "record_count": total_recs,
            }

        # ── SIMPLE / TREND / DISTRIBUTION ────────────────────────────────
        else:
            # Collect ALL sections (not just sections[0]) for simple queries
            all_records: List[Any] = []
            for sec in sections:
                all_records.extend(sec.get("data") or [])

            scores = _extract_scores(all_records, preserve_order=False)
            score_stats = _rich_score_stats(scores)
            stats = {"record_count": len(all_records), **score_stats}

            if scores:
                avg_score = score_stats["average_score"]
                min_score = score_stats["min_score"]
                max_score = score_stats["max_score"]
                median_score = score_stats["median_score"]
                category_label = humanize_category(category).lower()
                summary = (
                    f"Matched {len(all_records)} {category_label} records with an average score "
                    f"of {avg_score}, median {median_score}, ranging from {min_score} to {max_score}."
                )
                insights = [
                    f"Average score: {avg_score} | Median: {median_score} | "
                    f"Range: {min_score}–{max_score}.",
                ]
                insights += _score_insights(scores, category, score_stats)

                if category.lower() == "attendance":
                    time_ctx = _extract_time_context(all_records)
                    if time_ctx:
                        insights.insert(0, time_ctx)

                named = _named_score_records(all_records)
                weak = sorted(
                    [n for n in named if n["score"] < LOW_SCORE_THRESHOLD],
                    key=lambda x: x["score"],
                )[:3]
                strong = sorted(
                    [n for n in named if n["score"] > HIGH_SCORE_THRESHOLD],
                    key=lambda x: -x["score"],
                )[:3]

                if weak:
                    insights.append(
                        "Individuals needing attention: "
                        + ", ".join(f"{n['label']} ({n['score']})" for n in weak)
                        + "."
                    )
                if strong:
                    insights.append(
                        "Standout performers: "
                        + ", ".join(f"{n['label']} ({n['score']})" for n in strong)
                        + "."
                    )

                groups = _group_breakdown(all_records)
                if len(groups) > 1:
                    lead = groups[0]
                    trail = groups[-1]
                    insights.append(
                        f"Across {len(groups)} group(s), {lead['group']} leads with {lead['count']} "
                        f"record(s) averaging {lead['average_score']}, while {trail['group']} trails "
                        f"at {trail['average_score']}."
                    )
                    stats["group_breakdown"] = groups
            else:
                category_label = humanize_category(category).lower()
                summary = f"Matched {len(all_records)} {category_label} records."
                insights = [
                    f"The query returned {len(all_records)} {category_label} records."
                ]

                breakdown = _categorical_breakdown(all_records)
                if breakdown:
                    parts = ", ".join(
                        f"{item['count']} {item['value']}"
                        for item in breakdown["breakdown"]
                    )
                    summary = (
                        f"Matched {len(all_records)} {category_label} records — "
                        f"by {breakdown['field']}: {parts}."
                    )
                    insights = [f"Breakdown by {breakdown['field']}: {parts}."]
                    if breakdown["attention"]:
                        att = ", ".join(
                            f"{a['count']} {a['value']}" for a in breakdown["attention"]
                        )
                        insights.append(
                            f"{att} — flagged for follow-up out of {breakdown['total']} records."
                        )
                    stats["categorical_breakdown"] = breakdown

        return _analysis_payload(summary, insights, stats)

    except Exception as exc:
        logger.error("analysis_engine.generate_analysis failed: %s", exc, exc_info=True)
        category = intent.get("category") or "Agniveer"
        return _analysis_payload(
            f"Analysis of {humanize_category(category).lower()} records completed with limited metrics.",
            ["Dataset matches the specified parameters."],
            {"record_count": 0},
        )
