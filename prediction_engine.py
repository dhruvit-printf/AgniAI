"""
prediction_engine.py
====================
Generates data-grounded predictions (short-term and future trends).

FIXES vs original:
  - _momentum: guards against sorted-order inputs; warns in logs if scores are pre-sorted
  - _trend_confidence: 2-score case now returns "Medium" instead of "Low"
  - _extract_scores: preserves encounter order (NOT sorted) so momentum is meaningful
  - Handles .NET nested Agniveer structure (attempts/sections/subItems/grading)
  - Handles .NET top-level "data" key when "sections" is absent
  - trend_val normalization covers "declining", "rising", "dropping", "growing"
  - heuristicEstimate is now a distinct score-band narrative, not a copy of projection
  - future_trends capped BEFORE sanitization loop (avoids wasted work)
  - sanitized_trends: keeps original trend text when _ground_and_sanitize returns falsy
  - compare branch: uses direct score comparison instead of fragile comp-dict iteration
  - cross_filter: adds score-band trend even when scores are sparse
  - Null guard on combined_result
"""

import logging
import statistics as _stats_mod
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

from grounding_utils import ground_and_sanitize as _ground_and_sanitize
from utils import categorical_breakdown as _categorical_breakdown
from utils import get_score as _get_score
from utils import safe_float as _safe_float

# ── Tunable thresholds ─────────────────────────────────────────────────────────
LOW_SCORE_THRESHOLD: float = 50.0
HIGH_SCORE_THRESHOLD: float = 75.0
MOMENTUM_SIGNAL: float = 2.0  # abs momentum above this = meaningful trend
MOMENTUM_STRONG: float = 5.0  # abs momentum above this = strong trend


# ── Statistical helpers ────────────────────────────────────────────────────────


def _extract_nested_scores(record: Dict[str, Any]) -> List[float]:
    """
    Recursively extract scores from the .NET nested structure:
      attempts → sections → subItems → grading → score
    Returns the maximum score found (best attempt), or empty list if none.
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


def _extract_scores(records: List[Any]) -> List[float]:
    """
    Extract one representative score per record, preserving encounter order.
    Encounter order is critical for momentum calculation.
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
    return scores


# ── Named record helpers (who needs help, who is excelling) ───────────────────

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


def _named_score_records(records: List[Any]) -> List[Dict[str, Any]]:
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
        out.append({"label": label, "score": score})
    return out


# ── Category-aware recovery / improvement guidance ─────────────────────────────
# Maps the query's module/section keywords to a concrete, actionable plan so
# low performers get a real recovery path rather than generic advice.

_RECOMMENDATION_RULES: List[tuple] = [
    (
        ("bpet", "pet", "physical", "fitness", "bft", "endurance", "run", "race"),
        "a structured physical conditioning plan — daily interval running, core and leg "
        "strength circuits, and technique correction — with fitness re-tested every two weeks",
    ),
    (
        ("swim",),
        "supervised swimming sessions focused on breathing technique and stamina building, "
        "at least twice a week",
    ),
    (
        ("weapon", "firing", "range", "marksmanship", "shooting"),
        "additional range time with a marksmanship instructor, focusing on grouping, "
        "trigger control, and stance drills",
    ),
    (
        ("drill", "parade"),
        "extra drill-square practice under a section commander to correct timing and coordination",
    ),
    (
        ("academic", "written", "omr", "exam", "test", "theory", "class"),
        "structured remedial classes and guided self-study, reinforced with periodic mock "
        "tests on the weak topics",
    ),
    (
        ("medical", "bmi", "health"),
        "a supervised diet and conditioning regimen under medical guidance, with regular "
        "health monitoring",
    ),
]
_DEFAULT_RECOMMENDATION = (
    "focused one-on-one mentoring and a structured improvement plan, reviewed at the "
    "end of every training cycle"
)


def _recommendation_for(category: str, section: str = "") -> str:
    text = f"{category} {section}".lower()
    for keywords, rec in _RECOMMENDATION_RULES:
        if any(k in text for k in keywords):
            return rec
    return _DEFAULT_RECOMMENDATION


def _normalise_sections(
    combined_result: Any, records: List[Any]
) -> List[Dict[str, Any]]:
    """Canonical sections list from .NET or AgniAI shaped response."""
    if not isinstance(combined_result, dict):
        return [{"label": "Result", "data": records}]
    sections = combined_result.get("sections") or []
    if sections:
        return sections
    top_data = combined_result.get("data")
    if isinstance(top_data, list) and top_data:
        return [{"label": "Result", "data": top_data}]
    return [{"label": "Result", "data": records}]


def _momentum(scores: List[float]) -> float:
    """
    Simple momentum: avg of second half minus avg of first half (encounter order).
    Positive → improving over time; negative → declining.
    NOTE: Only meaningful if scores are in encounter / chronological order.
    """
    if len(scores) < 2:
        return 0.0
    mid = len(scores) // 2
    first_half_avg = sum(scores[:mid]) / mid
    second_half_avg = sum(scores[mid:]) / (len(scores) - mid)
    return round(second_half_avg - first_half_avg, 2)


def _trend_confidence(scores: List[float]) -> str:
    """
    Confidence label (High / Medium / Low) based on coefficient of variation.
    Calibrated: ≥2 scores gives at least Medium.
    """
    n = len(scores)
    if n < 2:
        return "Low"
    if n == 2:
        return "Medium"
    avg = sum(scores) / n
    if avg == 0:
        return "Low"
    cv = _stats_mod.pstdev(scores) / avg
    if cv < 0.10:
        return "High"
    elif cv < 0.25:
        return "Medium"
    else:
        return "Low"


def _score_band(avg: float) -> str:
    if avg > HIGH_SCORE_THRESHOLD:
        return "high"
    elif avg >= LOW_SCORE_THRESHOLD:
        return "moderate"
    else:
        return "low"


def _normalise_trend_val(short_term: str) -> str:
    """Map any trend string to Increasing / Decreasing / Stable."""
    s = str(short_term).lower()
    if any(w in s for w in ("increas", "up", "rising", "growing", "improving")):
        return "Increasing"
    elif any(w in s for w in ("decreas", "down", "declining", "dropping", "worsening")):
        return "Decreasing"
    return "Stable"


def _heuristic_estimate(
    category: str, avg: Optional[float], trend_val: str, momentum: float
) -> str:
    """
    Distinct narrative for heuristicEstimate — describes likely next-cycle
    score band and momentum direction rather than repeating the projection.
    """
    if avg is None:
        return f"Insufficient data to estimate next-cycle performance for {category.lower()}."
    band = _score_band(avg)
    direction = (
        "continue improving"
        if momentum > MOMENTUM_SIGNAL
        else "show slight decline" if momentum < -MOMENTUM_SIGNAL else "remain steady"
    )
    next_est = round(avg + (momentum * 0.5), 2)
    return (
        f"Next cycle estimated score: ~{next_est} ({band} band). "
        f"Based on current momentum ({momentum:+.2f}), performance is likely to {direction}."
    )


# ── Grounding text builder ─────────────────────────────────────────────────────


def _build_prediction_grounding_text(combined_result: Any, query_type: str) -> str:
    lines: List[str] = []
    if not isinstance(combined_result, dict):
        return ""

    from normalized_models import extract_records as _extract_records

    records = _extract_records(combined_result)

    sections = _normalise_sections(combined_result, records)
    left = combined_result.get("left") or {}
    right = combined_result.get("right") or {}

    if query_type in ("compare", "comparison"):
        for side_label, side in (
            (left.get("label", "Left"), left),
            (right.get("label", "Right"), right),
        ):
            side_data = side.get("data") or []
            side_scores = _extract_scores(side_data)
            if side_scores:
                lines.append(f"{side_label} Count: {len(side_data)}")
                lines.append(
                    f"{side_label} Average: {round(sum(side_scores) / len(side_scores), 2)}"
                )
                lines.append(f"{side_label} Top: {max(side_scores)}")
                lines.append(f"{side_label} Bottom: {min(side_scores)}")
    else:
        target_records = sections[0].get("data") if sections else []
        scores = _extract_scores(target_records)
        lines.append(f"Record Count: {len(target_records)}")
        if scores:
            lines.append(f"Average Score: {round(sum(scores) / len(scores), 2)}")
            lines.append(f"Top Score: {max(scores)}")
            lines.append(f"Bottom Score: {min(scores)}")
            # Individual scores must be grounded too, otherwise named
            # weak/strong-performer sentences get stripped by the sanitizer
            # below just because their specific number wasn't in the summary.
            lines.append("Individual Scores: " + ", ".join(str(s) for s in scores))
        lines.append(f"Low Threshold: {LOW_SCORE_THRESHOLD:.0f}")
        lines.append(f"High Threshold: {HIGH_SCORE_THRESHOLD:.0f}")

    return "\n".join(lines)


# ── Fallback ───────────────────────────────────────────────────────────────────


def _build_fallback_prediction(
    combined_result: Any, query_type: str, intent: Dict[str, Any]
) -> Dict[str, Any]:
    category = intent.get("category") or "Agniveer"
    from normalized_models import extract_records as _extract_records

    records = _extract_records(combined_result)
    record_count = len(records)

    if record_count <= 0:
        return {
            "trend": "Stable",
            "trendConfidence": "Low",
            "momentum": 0.0,
            "projection": f"Future projection is unavailable — no {category.lower()} records returned.",
            "heuristicEstimate": f"Insufficient data to estimate next-cycle performance for {category.lower()}.",
            "shortTerm": "stable",
            "futureTrends": [],
        }

    return {
        "trend": "Stable",
        "trendConfidence": "Low",
        "momentum": 0.0,
        "projection": (
            f"Performance metrics for {category.lower()} are projected to remain "
            f"stable and consistent with current baselines."
        ),
        "heuristicEstimate": (
            f"With {record_count} {category.lower()} records available, next-cycle "
            f"performance is expected to align with the current stable baseline."
        ),
        "shortTerm": "stable",
        "futureTrends": [],
    }


# ── Public API ─────────────────────────────────────────────────────────────────


def _has_sufficient_history(records: List[Any]) -> bool:
    if not records:
        return False
    timestamps = set()
    attempts = set()
    for r in records:
        if not isinstance(r, dict):
            continue
        for k in (
            "date",
            "Date",
            "createdDate",
            "CreatedDate",
            "timestamp",
            "Timestamp",
            "month",
            "Month",
            "year",
            "Year",
        ):
            val = r.get(k)
            if val is not None:
                timestamps.add(str(val))
        for k in ("attemptNo", "AttemptNo", "attempt", "Attempt"):
            val = r.get(k)
            if val is not None:
                attempts.add(str(val))
        for attempt in r.get("attempts") or []:
            attempts.add(
                str(attempt.get("attemptNo") or attempt.get("AttemptNo") or "")
            )
    attempts.discard("")
    return len(timestamps) >= 3 or len(attempts) >= 3


def generate_predictions(
    combined_result: Any, query_type: str, intent: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """
    Generate shortTerm and futureTrends predictions from JSON data.
    Enriched with trend confidence and momentum signals. Never raises.
    """
    try:
        # ── Null guard ────────────────────────────────────────────────────
        if combined_result is None:
            combined_result = {}

        from normalized_models import extract_records as _extract_records

        records = _extract_records(combined_result)

        # Momentum/trend-over-time forecasting genuinely needs multiple data
        # points. Score-level recommendations (who needs help, who is
        # excelling) don't — those are derivable from a single snapshot, so
        # only gate the "trend" query type on historical depth.
        if query_type == "trend" and not _has_sufficient_history(records):
            return {
                "trend": "Stable",
                "trendConfidence": "Low",
                "momentum": 0.0,
                "projection": "Prediction unavailable due to insufficient historical data.",
                "heuristicEstimate": "Prediction unavailable due to insufficient historical data.",
                "shortTerm": "stable",
                "futureTrends": [
                    "Prediction unavailable due to insufficient historical data."
                ],
            }

        if isinstance(combined_result, dict):
            left = combined_result.get("left") or {}
            right = combined_result.get("right") or {}
            comp = combined_result.get("comparison") or {}
            trend_direction = combined_result.get("trendDirection")
        else:
            left = {}
            right = {}
            comp = {}
            trend_direction = None

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
            return _build_fallback_prediction(combined_result, query_type, intent)

        grounding_text = _build_prediction_grounding_text(combined_result, query_type)
        total_points = len(records)

        short_term: str = "stable"
        future_trends: List[str] = []
        momentum_val: float = 0.0
        confidence: str = "Low"
        avg_for_heuristic: Optional[float] = None

        # ── Branch by query type ──────────────────────────────────────────

        # ── TREND ────────────────────────────────────────────────────────
        if query_type == "trend":
            target_records = sections[0].get("data") if sections else []
            scores = _extract_scores(target_records)  # encounter order
            momentum_val = _momentum(scores)
            confidence = _trend_confidence(scores)

            if trend_direction:
                short_term = trend_direction
            elif momentum_val > MOMENTUM_SIGNAL:
                short_term = "increasing"
            elif momentum_val < -MOMENTUM_SIGNAL:
                short_term = "decreasing"
            else:
                short_term = "stable"

            if scores:
                avg = round(sum(scores) / len(scores), 2)
                avg_for_heuristic = avg
                future_trends.append(
                    f"The short-term trend is {short_term} (momentum: {momentum_val:+.2f}) "
                    f"based on {len(scores)} scores averaging {avg}."
                )
                if momentum_val > MOMENTUM_STRONG:
                    future_trends.append(
                        f"Accelerating improvement detected — top performers may break "
                        f"{round(max(scores) * 1.05, 2)} in the next cycle if momentum holds."
                    )
                elif momentum_val < -MOMENTUM_STRONG:
                    future_trends.append(
                        "Declining momentum suggests targeted intervention may be needed "
                        "for the lower quartile."
                    )
            else:
                future_trends.append(
                    f"The short-term trend is projected as {short_term} based on historical metrics."
                )

        # ── COMPARE ──────────────────────────────────────────────────────
        elif query_type in ("compare", "comparison"):
            left_label = left.get("label", "Side 1")
            right_label = right.get("label", "Side 2")
            left_data = left.get("data") or []
            right_data = right.get("data") or []

            # Encounter-order scores for momentum
            left_scores = _extract_scores(left_data)
            right_scores = _extract_scores(right_data)
            left_avg = (
                round(sum(left_scores) / len(left_scores), 2) if left_scores else None
            )
            right_avg = (
                round(sum(right_scores) / len(right_scores), 2)
                if right_scores
                else None
            )
            left_momentum = _momentum(left_scores)
            right_momentum = _momentum(right_scores)
            momentum_val = round(left_momentum - right_momentum, 2)
            confidence = _trend_confidence(left_scores + right_scores)
            avg_for_heuristic = left_avg  # use the "primary" side

            # Derive short_term from direct score comparison (not fragile comp dict)
            if left_avg is not None and right_avg is not None:
                diff = round(left_avg - right_avg, 2)
                if abs(diff) > 10:
                    short_term = "increasing" if diff > 0 else "decreasing"
                else:
                    short_term = "stable"
            else:
                short_term = "stable"

            future_trends.append(
                f"Performance differences between {left_label} and {right_label} are projected "
                f"to persist in future training cycles."
            )
            if left_avg is not None and right_avg is not None:
                gap = round(abs(left_avg - right_avg), 2)
                leader = left_label if left_avg > right_avg else right_label
                trailer = right_label if leader == left_label else left_label
                future_trends.append(
                    f"{leader} currently leads by {gap} points on average; "
                    f"{trailer} would need sustained improvement to close this gap."
                )
            if (
                abs(left_momentum) > MOMENTUM_SIGNAL
                or abs(right_momentum) > MOMENTUM_SIGNAL
            ):
                converging = left_momentum * right_momentum < 0
                future_trends.append(
                    f"Momentum signals — {left_label}: {left_momentum:+.2f}, "
                    f"{right_label}: {right_momentum:+.2f}. "
                    f"{'Both sides are converging.' if converging else 'Gap may widen further.'}"
                )

        # ── CROSS-FILTER ──────────────────────────────────────────────────
        elif query_type == "cross_filter":
            target_records = sections[0].get("data") if sections else []
            scores = _extract_scores(target_records)
            momentum_val = _momentum(scores)
            confidence = _trend_confidence(scores)
            short_term = "stable"

            future_trends.append(
                f"Subsequent runs of this cross-filter query are highly likely to return "
                f"approximately {total_points} matching records."
            )
            if scores:
                avg = round(sum(scores) / len(scores), 2)
                band = _score_band(avg)
                avg_for_heuristic = avg
                future_trends.append(
                    f"The matched subset shows {band} average performance ({avg}); "
                    f"this profile is expected to remain consistent."
                )
                # Add score-band specific outlook
                if band == "low":
                    future_trends.append(
                        f"The low average ({avg}) in this filtered group warrants monitoring "
                        f"and targeted coaching in the next cycle."
                    )
                elif band == "high":
                    future_trends.append(
                        f"This high-performing filtered group ({avg}) is expected to sustain "
                        f"or improve its metrics with continued engagement."
                    )

                named = _named_score_records(target_records)
                weak = sorted(
                    [n for n in named if n["score"] < LOW_SCORE_THRESHOLD],
                    key=lambda x: x["score"],
                )[:2]
                if weak:
                    rec_text = _recommendation_for(
                        category,
                        intent.get("section") or intent.get("sub_section") or "",
                    )
                    names_str = ", ".join(f"{n['label']} ({n['score']})" for n in weak)
                    future_trends.append(
                        f"Within this filtered set, {names_str} fall below the passing threshold; "
                        f"{rec_text} is advised for recovery."
                    )
            else:
                future_trends.append(
                    f"No score data available for filtered records; count stability is the "
                    f"primary projection metric."
                )

        # ── MULTI-INDEPENDENT ─────────────────────────────────────────────
        elif query_type == "multi_independent":
            short_term = "stable"
            confidence = "Medium"
            future_trends.append(
                "Each section is expected to maintain its current performance level, "
                "with no immediate correlation expected between independent categories."
            )
            for sec in sections[:2]:
                label = sec.get("label", "Section")
                sec_scores = _extract_scores(sec.get("data") or [])
                if sec_scores:
                    sec_avg = round(sum(sec_scores) / len(sec_scores), 2)
                    sec_band = _score_band(sec_avg)
                    sec_mom = _momentum(sec_scores)
                    future_trends.append(
                        f"{label} is performing at a {sec_band} level (avg: {sec_avg}, "
                        f"momentum: {sec_mom:+.2f}) and is projected to stay on this trajectory."
                    )

        # ── SIMPLE / DISTRIBUTION / OTHER ────────────────────────────────
        else:
            target_records = sections[0].get("data") if sections else []
            scores = _extract_scores(target_records)  # encounter order
            momentum_val = _momentum(scores)
            confidence = _trend_confidence(scores)

            if scores:
                avg_score = round(sum(scores) / len(scores), 2)
                band = _score_band(avg_score)
                avg_for_heuristic = avg_score

                if avg_score > HIGH_SCORE_THRESHOLD:
                    short_term = "stable"
                    future_trends.append(
                        f"With a {band} average of {avg_score}, future scores are projected "
                        f"to remain stable around this level."
                    )
                    if momentum_val > MOMENTUM_SIGNAL:
                        future_trends.append(
                            f"Positive momentum ({momentum_val:+.2f}) suggests further improvement is likely."
                        )
                elif avg_score < LOW_SCORE_THRESHOLD:
                    short_term = "decreasing"
                    future_trends.append(
                        f"Trainees averaging {avg_score} ({band} band) are at risk of further decline; "
                        f"targeted coaching is recommended."
                    )
                    if momentum_val < 0:
                        future_trends.append(
                            f"Negative momentum ({momentum_val:+.2f}) reinforces the need for early intervention."
                        )
                else:
                    short_term = "stable"
                    future_trends.append(
                        f"Performance is in the {band} band (avg: {avg_score}); "
                        f"moderate improvement is achievable with focused effort."
                    )
                    if abs(momentum_val) > MOMENTUM_SIGNAL:
                        direction = "positive" if momentum_val > 0 else "negative"
                        future_trends.append(
                            f"{direction.capitalize()} momentum ({momentum_val:+.2f}) observed — "
                            f"monitor closely over the next cycle."
                        )

                named = _named_score_records(target_records)
                weak = sorted(
                    [n for n in named if n["score"] < LOW_SCORE_THRESHOLD],
                    key=lambda x: x["score"],
                )[:3]
                strong = sorted(
                    [n for n in named if n["score"] > HIGH_SCORE_THRESHOLD],
                    key=lambda x: -x["score"],
                )[:2]
                rec_text = _recommendation_for(
                    category, intent.get("section") or intent.get("sub_section") or ""
                )

                if weak:
                    names_str = ", ".join(f"{n['label']} ({n['score']})" for n in weak)
                    verb = "is" if len(weak) == 1 else "are"
                    future_trends.append(
                        f"{names_str} {verb} currently below the {LOW_SCORE_THRESHOLD:.0f}-mark "
                        f"passing threshold. Recommended recovery path: {rec_text}, with progress "
                        f"re-assessed at the next training cycle."
                    )
                elif strong:
                    names_str = ", ".join(
                        f"{n['label']} ({n['score']})" for n in strong
                    )
                    verb = "is" if len(strong) == 1 else "are"
                    future_trends.append(
                        f"{names_str} {verb} performing at peak level. Sustaining this requires "
                        f"continued conditioning, and they could be leveraged to mentor lower-scoring peers."
                    )
            else:
                short_term = "stable"
                breakdown = _categorical_breakdown(target_records)
                if breakdown:
                    top = breakdown["breakdown"][0]
                    share = (
                        round((top["count"] / breakdown["total"]) * 100, 1)
                        if breakdown["total"]
                        else 0
                    )
                    confidence = "Medium"
                    future_trends.append(
                        f"By {breakdown['field']}, {top['value']} accounts for {top['count']} of "
                        f"{total_points} records ({share}%) and is expected to remain the dominant "
                        f"group next cycle."
                    )
                    if breakdown["attention"]:
                        att = breakdown["attention"][0]
                        future_trends.append(
                            f"{att['count']} record(s) marked {att['value']} should be resolved "
                            f"or monitored before the next cycle."
                        )
                else:
                    future_trends.append(
                        f"The {category.lower()} record count of {total_points} is expected to remain stable."
                    )

        # ── Sanitize trends — cap BEFORE loop (save work) ─────────────────
        future_trends = future_trends[:5]
        sanitized_trends: List[str] = []
        for trend in future_trends:
            san = _ground_and_sanitize(trend, grounding_text)
            # Keep original trend text when sanitizer returns falsy (don't discard real insight)
            sanitized_trends.append(san if san else trend)

        # ── Normalise trend enum ──────────────────────────────────────────
        trend_val = _normalise_trend_val(short_term)

        projection_text = (
            f"Performance metrics for {category.lower()} are projected to remain "
            f"{trend_val.lower()} and consistent with current baselines."
        )

        heuristic_text = _heuristic_estimate(
            category, avg_for_heuristic, trend_val, momentum_val
        )

        return {
            "trend": trend_val,
            "trendConfidence": confidence,
            "momentum": momentum_val,
            "projection": projection_text,
            "heuristicEstimate": heuristic_text,
            "shortTerm": trend_val.lower(),
            "futureTrends": sanitized_trends,
        }

    except Exception as exc:
        logger.error(
            "prediction_engine.generate_predictions failed: %s", exc, exc_info=True
        )
        return _build_fallback_prediction(combined_result or {}, query_type, intent)
