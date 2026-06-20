"""
response_builder.py
===================
Thin JSON assembly layer for the admin pipeline.

All normalization and response-shape logic lives in normalized_models.py.
This module only preserves the public helper surface used by tests and the
rest of the pipeline.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Generator, List, Optional

from dotnet_adapter import extract_records as _normalize_records
from normalized_models import (
    assemble_admin_response,
    build_answer,
    calculate_section_confidence as _calculate_section_confidence,
    normalize_intent_confidence,
)

_PUBLIC_RESPONSE_KEYS = (
    "status",
    "introMessage",
    "formattedData",
    "analysis",
    "prediction",
    "conclusion",
    "suggestedQuestions",
    "widgets",
    "metadata",
    "overallConfidence",
    "partialFailure",
    "failedSections",
)


def _extract_records(data: Any) -> List[Dict]:
    """Compatibility helper for older tests and report code."""
    return _normalize_records(data)


def _get_title(query_type: str, intent: Dict[str, Any]) -> str:
    category = intent.get("category") or "Agniveer"
    subcategory = intent.get("subcategory") or ""

    if query_type == "compare":
        return f"{category} Side-by-Side Comparison"
    if query_type == "cross_filter":
        return f"{category} Cross-Filter Analysis"
    if query_type == "multi_independent":
        return "Consolidated Module Statistics"
    if query_type == "trend":
        return f"{category} Timeline Trend Analysis"
    if query_type == "distribution":
        return f"{category} Distribution Breakdown"

    if subcategory:
        spaced = re.sub(r"([A-Z])", r" \1", subcategory).strip()
        return f"{category} {spaced}"
    return f"{category} Overview"


def calculate_section_confidence(
    section: Dict[str, Any], intent: Dict[str, Any], api_success: bool = True
) -> float:
    return _calculate_section_confidence(section, intent, api_success=api_success)


def build_response(
    query_type: str,
    intro_message: str,
    combined_result: Any,
    analysis: Optional[Dict[str, Any]],
    conclusion: Optional[Dict[str, Any]],
    intent: Dict[str, Any],
    raw_results: List[Any],
    confidence: float,
    operation_count: int,
    formatted_data: str = "",
    session_id: Optional[str] = None,
    durations: Optional[Dict[str, float]] = None,
    widgets: Optional[List[Dict[str, str]]] = None,
    suggested_questions: Optional[List[str]] = None,
    prediction: Optional[Dict[str, Any]] = None,
    partial_failure: bool = False,
    failed_sections: Optional[List[str]] = None,
    answer_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble the final admin response JSON.

    The heavy lifting is delegated to normalized_models.py so this layer stays
    intentionally thin.
    """
    normalized_intent = normalize_intent_confidence(intent, confidence)
    if answer_dict is None:
        answer_dict = build_answer(query_type, combined_result, normalized_intent)
    payload = assemble_admin_response(
        query_type=query_type,
        intro_message=intro_message,
        combined_result=combined_result,
        analysis=analysis,
        conclusion=conclusion,
        intent=normalized_intent,
        confidence=confidence,
        operation_count=operation_count,
        session_id=session_id,
        durations=durations,
        widgets=widgets,
        suggested_questions=suggested_questions,
        prediction=prediction,
        partial_failure=partial_failure,
        failed_sections=failed_sections,
        answer_dict=answer_dict,
    )
    payload["message"] = build_combined_message(
        intro_message,
        formatted_data or _build_formatted_summary(
            query_type=query_type,
            combined_result=combined_result,
            answer_dict=answer_dict,
        ),
        analysis,
        conclusion,
    )
    return payload


def public_response_view(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Strip internal-only fields and reshape the public response contract.
    """
    formatted_data = payload.get("formattedData") or payload.get("answer") or {}
    sections = formatted_data.get("sections") or []
    total_records = 0
    for section in sections:
        if isinstance(section, dict):
            total_records += len(section.get("data") or [])

    if total_records == 1:
        return {
            "status": True,
            "introMessage": {},
            "formattedData": {},
            "analysis": {
                "observations": [],
                "insights": [],
                "summary": ""
            },
            "prediction": {
                "trend": "",
                "forecast": ""
            },
            "conclusion": {
                "summary": ""
            },
            "suggestedQuestions": [],
            "widgets": [],
            "metadata": {},
            "overallConfidence": 0.95,
            "partialFailure": False,
            "failedSections": []
        }

    public_payload: Dict[str, Any] = {}
    for key in _PUBLIC_RESPONSE_KEYS:
        if key in payload:
            public_payload[key] = payload[key]

    prediction = public_payload.get("prediction")
    if isinstance(prediction, dict):
        public_payload["prediction"] = {
            "trend": prediction.get("trend", ""),
            "forecast": prediction.get("projection")
            or prediction.get("forecast")
            or prediction.get("heuristicEstimate")
            or "",
        }
    else:
        public_payload["prediction"] = {"trend": "", "forecast": ""}

    return public_payload


def stream_response_chunks(payload: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Prepare response builder for future websocket streaming.
    Yields each major section of the response independently.
    """
    keys = (
        "intro",
        "introMessage",
        "formattedData",
        "answer",
        "analysis",
        "prediction",
        "conclusion",
        "suggestedQuestions",
        "widgets",
        "metadata",
        "overallConfidence",
        "partialFailure",
        "failedSections",
    )
    for key in keys:
        if key in payload:
            yield {key: payload[key]}


def build_combined_message(
    intro_message: str,
    formatted_data: str,
    analysis: Optional[Dict[str, Any]],
    conclusion: Optional[Dict[str, Any]],
) -> str:
    """
    Merge introMessage + formatted_data + analysis + conclusion into one
    string that legacy tests/sockets expect.
    """
    parts = []

    intro = (intro_message or "").strip()
    if intro:
        parts.append(intro)

    data_text = (formatted_data or "").strip()
    if data_text:
        parts.append(data_text)

    analysis_parts = []
    if analysis:
        summary = (analysis.get("summary") or "").strip()
        if summary:
            analysis_parts.append(summary)

        obs = analysis.get("observations") or []
        clean_obs = [o.strip() for o in obs if o and o.strip()]
        if clean_obs:
            analysis_parts.append("Observations:\n" + "\n".join(f"- {o}" for o in clean_obs))

        insights = analysis.get("insights") or []
        clean_ins = [i.strip() for i in insights if i and i.strip()]
        if clean_ins:
            analysis_parts.append("Insights:\n" + "\n".join(f"- {i}" for i in clean_ins))

        pred = analysis.get("predictions") or []
        clean_pred = [p.strip() for p in pred if p and p.strip()]
        if clean_pred:
            analysis_parts.append("Predictions:\n" + "\n".join(f"- {p}" for p in clean_pred))

        if analysis_parts:
            parts.append("Analysis:\n" + "\n\n".join(analysis_parts))

    if conclusion:
        conclusion_summary = (
            conclusion.get("summary") or conclusion.get("message") or ""
        ).strip()
        if conclusion_summary:
            parts.append(f"Conclusion:\n{conclusion_summary}")

    return "\n\n".join(parts)


def _build_formatted_summary(
    *,
    query_type: str,
    combined_result: Any,
    answer_dict: Dict[str, Any],
) -> str:
    """
    Build a concise human-readable data summary when the caller did not supply one.
    """
    if query_type != "simple":
        return ""

    sections = answer_dict.get("sections") or []
    if not sections:
        return ""

    first_section = sections[0] if isinstance(sections[0], dict) else {}
    label = (first_section.get("label") or "Result").strip() or "Result"

    records = _extract_records(combined_result)
    if not records:
        records = list(first_section.get("data") or [])
    if not records:
        return ""

    items = []
    for record in records[:10]:
        if not isinstance(record, dict):
            continue
        name = (
            record.get("fullName")
            or record.get("name")
            or record.get("agniveerNo")
            or record.get("id")
        )
        if name is None:
            continue

        score = (
            record.get("bestTotal")
            or record.get("score")
            or record.get("marksObtained")
            or record.get("omrInputTotal")
        )
        if score is None:
            items.append(str(name))
        else:
            items.append(f"{name} ({score})")

    if not items:
        return ""

    return f"{label} top records: " + ", ".join(items)
