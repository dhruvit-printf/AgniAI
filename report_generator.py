"""
report_generator.py
===================
Orchestration layer that delegates report generation to dedicated analysis, prediction,
and conclusion engines, maintaining compatibility with tests and old references.
"""

import logging
import time
from typing import Any, Dict, List, Optional

from analysis_engine import generate_analysis
from conclusion_engine import generate_conclusion
from grounding_utils import extract_numbers_from_text as _extract_numbers_from_text
from grounding_utils import ground_and_sanitize as _strip_ungrounded_numbers
from normalized_models import humanize_category
from prediction_engine import generate_predictions
from utils import extract_records as _extract_records
from utils import get_score as _get_score
from utils import has_any_data as _has_any_data

logger = logging.getLogger(__name__)


def _log_stage(stage: str, duration_ms: float, **extra: Any) -> None:
    event = {"stage": stage, "duration_ms": round(duration_ms, 2)}
    event.update({k: v for k, v in extra.items() if v is not None})
    logger.info(event)


from intent_engine.intent_schema import NEGATIVE_REPORT_PHRASES


def _has_negative_copy(text: Optional[str]) -> bool:
    if not text:
        return False
    lowered = text.lower()
    return any(marker in lowered for marker in NEGATIVE_REPORT_PHRASES)


def _normalize_prediction_category(category: Optional[str]) -> str:
    """Normalize a category label for fallback prediction text: falsy/
    unclear/unknown/none -> "Agniveer", otherwise Title-cased first letter.

    Shared with admin_pipeline.py's equivalent block (previously an
    independently-maintained duplicate of this exact logic).
    """
    pred_cat = (
        str(category).strip() if category and str(category).strip() else "Agniveer"
    )
    if pred_cat.lower() in ("unclear", "unknown", "none"):
        return "Agniveer"
    return pred_cat[0].upper() + pred_cat[1:]


def _build_fallback_prediction_dict(
    category: Optional[str], has_records: bool
) -> Dict[str, Any]:
    """Build the fallback "prediction" shape (trend/projection/
    heuristicEstimate/shortTerm/futureTrends) used when no real
    prediction_engine output is available.

    Shared by report_generator.py and admin_pipeline.py — previously two
    independently-maintained copies that had drifted to three different
    "no records" phrasings across their fields.
    """
    pred_cat = _normalize_prediction_category(category)
    if has_records:
        text = (
            f"Future {pred_cat} results should remain broadly stable "
            f"unless the underlying records change."
        )
    else:
        text = f"Future projection is unavailable because no {pred_cat} records were returned."
    return {
        "trend": "Stable" if has_records else "Insufficient Data",
        "projection": text,
        "heuristicEstimate": text,
        "shortTerm": "stable",
        "futureTrends": [text],
    }


def _build_data_grounded_report(
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
) -> Dict[str, Any]:
    records = _extract_records(combined_result)

    category = intent.get("category") or "Agniveer"
    category_label = humanize_category(category).lower()
    cnt = len(records)
    scores = [_get_score(record) for record in records if isinstance(record, dict)]
    scores = [score for score in scores if score is not None]

    if query_type == "cross_filter":
        match_count = (
            combined_result.get("matchCount", cnt)
            if isinstance(combined_result, dict)
            else cnt
        )
        total_before = (
            combined_result.get("totalBeforeFilter", match_count)
            if isinstance(combined_result, dict)
            else match_count
        )
        if match_count <= 0:
            return {
                "message": "No matching records found.",
                "analysis": {
                    "summary": "No matching records found.",
                    "observations": [],
                    "insights": [],
                    "predictions": [],
                },
                "prediction": {
                    "trend": "Insufficient Data",
                    "projection": "No future projection is available because no matching records were returned.",
                    "heuristicEstimate": "No future projection is available because no matching records were returned.",
                    "shortTerm": "stable",
                    "futureTrends": [],
                },
                "conclusion": {"summary": "No matching records found."},
            }
        message = (
            f"The cross-filter query completed successfully and returned {match_count} matching records "
            f"out of {total_before} candidates."
        )
        analysis_summary = f"Cross-filter analysis matched {match_count} records after intersecting the requested conditions."
        observations = [f"{match_count} records satisfied all filter conditions."]
        if total_before:
            observations.append(
                f"{match_count} of {total_before} candidates remained after filtering."
            )
        insights = [
            "The result set is grounded in the overlapping criteria that were returned."
        ]
        conclusion = f"The cross-filter result set is complete with {match_count} matching records and is ready for review."
        prediction = {
            "trend": "Stable",
            "projection": (
                f"Future cross-filter runs are expected to remain near {match_count} matches if the same criteria are reused."
            ),
            "heuristicEstimate": (
                f"Future cross-filter runs are expected to remain near {match_count} matches if the same criteria are reused."
            ),
            "shortTerm": "stable",
            "futureTrends": [
                f"Cross-filter match counts should remain close to {match_count} when criteria and source data stay unchanged."
            ],
        }
        return {
            "message": message,
            "analysis": {
                "summary": analysis_summary,
                "observations": observations,
                "insights": insights,
                "predictions": [],
            },
            "prediction": prediction,
            "conclusion": {"summary": conclusion},
        }

    if query_type in ("compare", "comparison"):
        sides = (
            combined_result.get("sides", [])
            if isinstance(combined_result, dict)
            else []
        )
        left = combined_result.get("left") or {}
        right = combined_result.get("right") or {}
        labels = [
            side.get("label", f"Side {idx + 1}") for idx, side in enumerate(sides[:2])
        ]
        if len(labels) < 2:
            labels = [left.get("label", "Side 1"), right.get("label", "Side 2")]
        left_metrics = left.get("metrics") or {}
        right_metrics = right.get("metrics") or {}
        left_cnt = left_metrics.get("recordCount")
        if left_cnt is None:
            left_cnt = len(left.get("data", [])) if isinstance(left, dict) else 0
        else:
            left_cnt = int(left_cnt)
        right_cnt = right_metrics.get("recordCount")
        if right_cnt is None:
            right_cnt = len(right.get("data", [])) if isinstance(right, dict) else 0
        else:
            right_cnt = int(right_cnt)
        comparison = (
            combined_result.get("comparison")
            if isinstance(combined_result, dict)
            else {}
        )
        diff = None
        if isinstance(comparison, dict):
            first_metric = next(iter(comparison.values()), {})
            diff = first_metric.get("difference")
        message = f"I completed a side-by-side comparison between {labels[0]} and {labels[1]}."
        analysis_summary = f"Comparison analysis reviewed {left_cnt} records on {labels[0]} and {right_cnt} records on {labels[1]}."
        observations = [
            f"{labels[0]} contains {left_cnt} records.",
            f"{labels[1]} contains {right_cnt} records.",
        ]
        if diff is not None:
            observations.append(f"The recorded comparison difference is {diff}.")
        insights = ["The comparison is grounded in the records returned for each side."]
        conclusion = f"The comparison is complete and the returned metrics show {labels[0]} versus {labels[1]} without inventing any unreturned values."
        projection = f"Future comparison results should stay close to the returned {left_cnt} and {right_cnt} record counts unless the source data changes."
        return {
            "message": message,
            "analysis": {
                "summary": analysis_summary,
                "observations": observations,
                "insights": insights,
                "predictions": [],
            },
            "prediction": {
                "trend": "Stable",
                "projection": projection,
                "heuristicEstimate": projection,
                "shortTerm": "stable",
                "futureTrends": [projection],
            },
            "conclusion": {"summary": conclusion},
        }

    if query_type == "multi_independent":
        sections = (
            combined_result.get("sections") or []
            if isinstance(combined_result, dict)
            else []
        )
        section_counts = []
        for section in sections:
            data = section.get("data") or []
            section_counts.append((section.get("label", "Section"), len(data)))
        message = (
            f"I combined {len(section_counts)} independent sections into one report."
        )
        analysis_summary = f"The multi-section report contains {len(section_counts)} independent sections with grounded record counts."
        observations = [
            f"{label} returned {count} records." for label, count in section_counts[:5]
        ]
        insights = [
            "Each section is preserved independently to prevent data loss across modules."
        ]
        conclusion = "The consolidated report is complete and reflects only the sections that were returned."
        return {
            "message": message,
            "analysis": {
                "summary": analysis_summary,
                "observations": observations,
                "insights": insights,
                "predictions": [],
            },
            "prediction": {
                "trend": "Stable",
                "projection": "Future section counts should remain aligned with the current information unless the source data changes.",
                "heuristicEstimate": "Future section counts should remain aligned with the current information unless the source data changes.",
                "shortTerm": "stable",
                "futureTrends": [
                    "Future section counts should remain aligned with the current information unless the source data changes."
                ],
            },
            "conclusion": {"summary": conclusion},
        }

    message = f"The {category_label} query completed successfully and returned {cnt} matched records."
    if cnt == 0:
        message = "No data is found for what you asked for."

    analysis_summary = f"Matched {cnt} {category_label} records."
    if scores:
        avg_score = round(sum(scores) / len(scores), 2)
        analysis_summary = (
            f"Matched {cnt} {category_label} records with an average score of {avg_score}, "
            f"ranging from {round(min(scores), 2)} to {round(max(scores), 2)}."
        )

    observations = [f"The query returned {cnt} {category_label} records."]
    if scores:
        observations.append(f"Average score: {round(sum(scores) / len(scores), 2)}.")
        observations.append(
            f"Score range: {round(min(scores), 2)} to {round(max(scores), 2)}."
        )

    insights = ["The response is grounded in the records that were returned."]
    if scores:
        insights.append(
            "The score distribution is based on the actual returned values."
        )

    return {
        "message": message,
        "summary": analysis_summary,
        "analysis": {
            "summary": analysis_summary,
            "observations": observations,
            "insights": insights,
            "predictions": [],
        },
        "prediction": _build_fallback_prediction_dict(category, cnt > 0),
        "conclusion": {
            "summary": f"The {category_label} search returned {cnt} records and the result set is ready for review.",
        },
    }


_INTRO_TEMPLATES: Dict[Any, str] = {
    (
        "Performance",
        "TopPerformers",
    ): "These assessment results highlight the strongest performers in the evaluation.",
    (
        "Performance",
        "LowestPerformers",
    ): "These results identify the individuals requiring additional training support.",
    (
        "Leave",
        "CurrentLeaveStatus",
    ): "Current leave records outline person availability across the unit.",
    (
        "Medical",
        "ActiveCases",
    ): "This summary captures current active cases undergoing medical attention.",
    (
        "Verification",
        "CompletedVerification",
    ): "These records confirm files that have cleared the verification process.",
}


def get_fallback_report(
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
) -> Dict[str, Any]:
    category = intent.get("category") or "Agniveer"
    subcategory = intent.get("subcategory") or ""
    records = _extract_records(combined_result)
    cnt = len(records)

    if query_type == "cross_filter":
        match_count = (
            combined_result.get("matchCount", cnt)
            if isinstance(combined_result, dict)
            else cnt
        )
        total_before = (
            combined_result.get("totalBeforeFilter", 0)
            if isinstance(combined_result, dict)
            else 0
        )
        if match_count > 0:
            message = f"I found {match_count} records that match all of the selected conditions."
            summary = f"The cross-filter search identified exactly {match_count} records that satisfy every selected condition."
            obs = [
                f"{match_count} records were matched out of {total_before} before filtering."
            ]
            insights = [
                "The overlap between the selected conditions identifies the records that fit every rule."
            ]
            conclusion = f"The cross-filter search is complete and the {match_count} matching records are ready for review."
        else:
            # cross_filter_datasets() already distinguishes "a condition
            # genuinely returned nothing" (message stays a plain "no records
            # found") from "every condition matched records individually but
            # none overlap" (message names what was found) — reuse it here
            # instead of a message that implies the data doesn't exist.
            no_overlap_message = (
                combined_result.get("message")
                if isinstance(combined_result, dict)
                else None
            ) or "No data is found for what you asked for."
            message = no_overlap_message
            summary = "The selected conditions may be too narrow for the current set of records."
            obs = ["The search did not return any overlapping records."]
            insights = []
            conclusion = "Try broadening one of the filters to see more results."
    elif query_type in ("comparison", "compare"):
        sides = (
            combined_result.get("sides", [])
            if isinstance(combined_result, dict)
            else []
        )
        labels = [s.get("label", "Section") for s in sides]
        labels_str = " and ".join(labels) if labels else "selected categories"
        has_data = (
            any(
                len(s.get("data", [])) > 0
                or s.get("metrics", {}).get("recordCount", 0) > 0
                for s in sides
            )
            if sides
            else (cnt > 0)
        )
        if has_data:
            message = f"I completed a side-by-side comparison between {labels_str}."
            summary = f"The comparison highlights the differences across {labels_str}."
            obs = [f"Compared {len(sides)} groups: {', '.join(labels)}."]
            insights = [
                "Looking at the groups side by side makes the main differences easy to spot."
            ]
            conclusion = (
                f"The comparison of {labels_str} is complete and ready for review."
            )
        else:
            message = "I couldn't complete that comparison because neither side returned any matching records. You might try adjusting the filters and searching again."
            summary = "The comparison could not be completed because the selected groups returned no records."
            obs = ["Both comparison groups returned no records."]
            insights = [
                "There is not enough data available yet to compare the selected groups."
            ]
            conclusion = "The comparison could not be completed because no matching records were found."
    elif query_type == "multi_independent":
        sections = (
            combined_result.get("sections") or []
            if isinstance(combined_result, dict)
            else []
        )
        has_data = any(len(s.get("data", [])) > 0 for s in sections)
        if has_data:
            message = f"I combined data from {len(sections)} independent sections into one report."
            summary = f"The consolidated report brings together {len(sections)} separate sections."
            obs = [f"Merged {len(sections)} independent sections into one view."]
            insights = [
                "Keeping each section separate makes the report easier to review."
            ]
            conclusion = f"The consolidated report is complete and includes all {len(sections)} sections."
        else:
            message = "I compiled the report, but none of the requested sections returned any records."
            summary = "The multi-section report is empty because no records were returned for the requested sections."
            obs = ["All requested sections returned no records."]
            insights = [
                "There is not enough data available in the requested sections to build a fuller report."
            ]
            conclusion = "The consolidated report is empty because no matching records were found."
    else:
        category_label = humanize_category(category).lower()
        if cnt > 0:
            message = f"I found {cnt} matching {category_label} records."
            key = (category, subcategory)
            if key in _INTRO_TEMPLATES:
                message = _INTRO_TEMPLATES[key]
            summary = (
                f"The search returned exactly {cnt} matching {category_label} records."
            )
            obs = [f"Found {cnt} matching {category_label} records."]
            insights = ["The returned records match the selected criteria."]
            conclusion = f"The search is complete and the {cnt} matching {category_label} records are ready for review."
        else:
            message = "No data is found for what you asked for."
            summary = f"No matching records were found for the selected {category_label} criteria."
            obs = [f"The search returned 0 records for the {category_label} category."]
            insights = [
                "The selected criteria may be too narrow for the available records."
            ]
            conclusion = f"I didn't find any matching {category_label} records this time — broadening the criteria may help."

    return {
        "message": message,
        "summary": summary,
        "analysis": {
            "summary": summary,
            "observations": obs,
            "insights": insights,
            "predictions": [],
        },
        "conclusion": {"summary": conclusion},
    }


def generate_report(
    combined_result: Any,
    query_type: str,
    intent: Dict[str, Any],
    user_query: str = "",
    trace_id: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Generate the structured report elements by delegating to dedicated engines.
    """
    records = _extract_records(combined_result)

    has_valid_data = _has_any_data(records)
    if (
        not has_valid_data
        and query_type in ("compare", "comparison")
        and isinstance(combined_result, dict)
    ):
        sides = combined_result.get("sides", [])
        if any(
            len(s.get("data", [])) > 0 or s.get("metrics", {}).get("recordCount", 0) > 0
            for s in sides
        ):
            has_valid_data = True

    if not has_valid_data:
        fallback = get_fallback_report(combined_result, query_type, intent)

        # cross_filter_datasets() already distinguishes "a condition
        # genuinely returned nothing" from "every condition matched records
        # individually but none overlap" and phrases the message accordingly
        # — prefer that specific message over the generic fallback text.
        no_match_message = (
            (
                combined_result.get("message")
                if query_type == "cross_filter" and isinstance(combined_result, dict)
                else None
            )
            or fallback.get("message")
            or "No data is found for what you asked for."
        )

        analysis_summary = (
            fallback.get("analysis", {}).get("summary")
            or "The selected conditions may be too narrow."
        )
        conclusion_summary = (
            fallback.get("conclusion", {}).get("summary")
            or "Try broadening the filters to find matching records."
        )

        return {
            "message": no_match_message,
            "summary": analysis_summary,
            "analysis": {
                "summary": analysis_summary,
                "insights": fallback.get("analysis", {}).get("insights", []),
                "statistics": {"record_count": 0},
            },
            "prediction": {
                "trend": "Insufficient Data",
                "projection": "Unavailable.",
                "heuristicEstimate": "Unavailable",
                "shortTerm": "insufficient data",
                "futureTrends": [],
            },
            "conclusion": {
                "summary": conclusion_summary,
                "bullets": [],
            },
            "durations": {
                "analysisDurationMs": 0.0,
                "predictionDurationMs": 0.0,
                "conclusionDurationMs": 0.0,
            },
        }

    analysis = None
    analysis_ms = 0.0
    t0 = time.time()
    try:
        analysis = generate_analysis(
            combined_result, query_type, intent, user_query, trace_id
        )
        analysis_ms = round((time.time() - t0) * 1000, 2)
        _log_stage(
            "analysis_time", analysis_ms, query_type=query_type, trace_id=trace_id
        )
    except Exception as exc:
        analysis_ms = round((time.time() - t0) * 1000, 2)
        logger.error("report_generator: analysis stage failed: %s", exc, exc_info=True)
        _log_stage(
            "analysis_time",
            analysis_ms,
            query_type=query_type,
            trace_id=trace_id,
            error=str(exc),
        )

    prediction = None
    prediction_ms = 0.0
    t0 = time.time()
    try:
        prediction = generate_predictions(combined_result, query_type, intent)
        prediction_ms = round((time.time() - t0) * 1000, 2)
        _log_stage(
            "prediction_time", prediction_ms, query_type=query_type, trace_id=trace_id
        )
    except Exception as exc:
        prediction_ms = round((time.time() - t0) * 1000, 2)
        logger.error(
            "report_generator: prediction stage failed: %s", exc, exc_info=True
        )
        _log_stage(
            "prediction_time",
            prediction_ms,
            query_type=query_type,
            trace_id=trace_id,
            error=str(exc),
        )

    conclusion = None
    conclusion_ms = 0.0
    t0 = time.time()
    try:
        conclusion = generate_conclusion(combined_result, query_type, intent, trace_id)
        conclusion_ms = round((time.time() - t0) * 1000, 2)
        _log_stage(
            "conclusion_time", conclusion_ms, query_type=query_type, trace_id=trace_id
        )
    except Exception as exc:
        conclusion_ms = round((time.time() - t0) * 1000, 2)
        logger.error(
            "report_generator: conclusion stage failed: %s", exc, exc_info=True
        )
        _log_stage(
            "conclusion_time",
            conclusion_ms,
            query_type=query_type,
            trace_id=trace_id,
            error=str(exc),
        )

    try:
        from narrative_engine import generate_narratives

        narratives = generate_narratives(
            user_query=user_query,
            combined_result=combined_result,
            query_type=query_type,
            intent=intent,
            analysis=analysis,
            prediction=prediction,
            conclusion=conclusion,
            trace_id=trace_id,
        )
        message = narratives.get("message") or ""
        analysis_str = narratives.get("analysis") or ""
        prediction_str = narratives.get("prediction") or ""
        conclusion_str = narratives.get("conclusion") or ""
        summary_str = narratives.get("summary") or ""
    except Exception as exc:
        logger.warning("report_generator: narrative_engine failed: %s", exc)
        category = intent.get("category") or "Agniveer"
        category_label = humanize_category(category).lower()
        message = (analysis or {}).get(
            "summary"
        ) or f"The {category_label} query returned {len(records)} records."
        analysis_str = (analysis or {}).get("summary", "")
        prediction_str = (prediction or {}).get("projection", "")
        conclusion_str = (
            (conclusion or {}).get("summary") or (conclusion or {}).get("message") or ""
        )
        summary_str = (analysis or {}).get("summary", "")

    intro_needs_fallback = _has_negative_copy(message) or _has_negative_copy(
        analysis_str
    )
    conclusion_needs_fallback = _has_negative_copy(conclusion_str)

    if intro_needs_fallback or conclusion_needs_fallback:
        grounded = _build_data_grounded_report(combined_result, query_type, intent)
        if intro_needs_fallback:
            message = grounded.get("message", message)
            analysis_str = grounded.get("analysis", {}).get("summary", analysis_str)
            prediction_str = grounded.get("prediction", {}).get(
                "projection", prediction_str
            )
            summary_str = grounded.get("summary", summary_str)
        if conclusion_needs_fallback:
            conclusion_str = grounded.get("conclusion", {}).get(
                "summary", conclusion_str
            )

    return {
        "message": message,
        "summary": summary_str,
        "analysis": {
            "summary": analysis_str,
            "insights": [],
            "statistics": {},
            "observations": [],
            "predictions": [],
        },
        "prediction": {
            "trend": "Stable",
            "projection": prediction_str,
            "heuristicEstimate": prediction_str,
            "shortTerm": "stable",
            "futureTrends": [prediction_str] if prediction_str else [],
        },
        "conclusion": {
            "summary": conclusion_str,
            "bullets": (
                conclusion.get("bullets")
                if (conclusion and isinstance(conclusion, dict))
                else []
            ),
        },
        "durations": {
            "analysisDurationMs": analysis_ms,
            "predictionDurationMs": prediction_ms,
            "conclusionDurationMs": conclusion_ms,
        },
    }
