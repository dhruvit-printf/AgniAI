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
from conversation_detector import build_conversational_response as _build_conversational_payload

_PUBLIC_RESPONSE_KEYS = (
    "status",
    "sessionId",
    "message",
    "widget",
    "records",
    "formattedData",
    "suggestedQuestions",
    "analysis",
    "prediction",
    "conclusion",
    "queryType",
    "answer",
    "overallConfidence",
    "metadata",
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
    formatted_data: Any = None,
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
    Assemble the final response payload strictly using the canonical schema.
    """
    from schemas import FinalResponse
    from telemetry import request_id_var, trace_id_var, session_id_var
    from normalized_models import build_answer, normalize_intent_confidence
    from visualization_intent import build_visualization_intent

    durations_dict = durations or {}
    session_value = session_id or session_id_var.get("admin-default") or "admin-default"
    request_value = durations_dict.get("requestId") or request_id_var.get("N/A")
    trace_value = durations_dict.get("traceId") or trace_id_var.get("N/A")

    metadata = {
        "requestId": request_value,
        "traceId": trace_value,
        "sessionId": session_value,
        "timings": {
            "entityResolutionMs": durations_dict.get("entityResolutionMs")
            or durations_dict.get("entity_resolution_ms", 0.0),
            "planningMs": durations_dict.get("planningMs")
            or durations_dict.get("planning_ms", 0.0),
            "plannerDurationMs": durations_dict.get("plannerDurationMs")
            or durations_dict.get("planner_duration", 0.0),
            "intentDurationMs": durations_dict.get("intentDurationMs")
            or durations_dict.get("intent_duration", 0.0),
            "dotnetDurationMs": durations_dict.get("dotnetDurationMs")
            or durations_dict.get("dotnet_duration", 0.0),
            "combineDurationMs": durations_dict.get("combineDurationMs")
            or durations_dict.get("combiner_duration", 0.0),
            "widgetMs": durations_dict.get("widgetMs")
            or durations_dict.get("widget_duration", 0.0),
            "responseAssemblyMs": durations_dict.get("responseAssemblyMs")
            or durations_dict.get("response_assembly_duration", 0.0),
            "analysisDurationMs": durations_dict.get("analysisDurationMs")
            or durations_dict.get("analysis_duration", 0.0),
            "predictionDurationMs": durations_dict.get("predictionDurationMs")
            or durations_dict.get("prediction_duration", 0.0),
            "conclusionDurationMs": durations_dict.get("conclusionDurationMs")
            or durations_dict.get("conclusion_duration", 0.0),
            "totalDurationMs": durations_dict.get("totalDurationMs")
            or durations_dict.get("total_duration", 0.0),
            "executionTimeMs": durations_dict.get("executionTimeMs")
            or durations_dict.get("execution_time_ms", 0.0),
        },
        "metrics": {
            "confidence": round(float(confidence), 2),
            "queryType": query_type,
            "operationCount": int(operation_count),
        },
        "planner_duration": durations_dict.get("planner_duration")
        or durations_dict.get("plannerDurationMs", 0.0),
        "intent_duration": durations_dict.get("intent_duration")
        or durations_dict.get("intentDurationMs", 0.0),
        "dotnet_duration": durations_dict.get("dotnet_duration")
        or durations_dict.get("dotnetDurationMs", 0.0),
        "combiner_duration": durations_dict.get("combiner_duration")
        or durations_dict.get("combineDurationMs", 0.0),
        "report_duration": durations_dict.get("report_duration")
        or durations_dict.get("analysisDurationMs", 0.0),
        "total_duration": durations_dict.get("total_duration")
        or durations_dict.get("totalDurationMs", 0.0),
        "confidence": round(float(confidence), 2),
        "queryType": query_type,
        "operationCount": int(operation_count),
    }

    normalized_intent = normalize_intent_confidence(intent, confidence)
    answer_payload = (
        answer_dict
        if answer_dict is not None
        else build_answer(query_type, combined_result, normalized_intent)
    )
    viz_intent = build_visualization_intent(
        intro_message or "",
        normalized_intent,
        combined_result,
    )

    if isinstance(formatted_data, dict):
        fd_payload = dict(formatted_data)
    else:
        fd_payload = None

    if fd_payload is None:
        fd_payload = {
            "type": "TABLE",
            "title": "Data Summary",
            "data": {"columns": [], "rows": []},
        }

    # Ensure the formatted payload never leaks analytics/report text.
    for noisy_key in ("analysis", "prediction", "conclusion"):
        fd_payload.pop(noisy_key, None)

    if "type" not in fd_payload:
        fd_payload["type"] = "TABLE"
    if "title" not in fd_payload:
        fd_payload["title"] = "Data Summary"
    if "data" not in fd_payload:
        fd_payload["data"] = {"columns": [], "rows": []}

    widget_value = viz_intent.get("presentation") or "table"

    response_model = FinalResponse(
        status=True,
        sessionId=session_value,
        message=(intro_message or "").strip(),
        queryType=query_type,
        answer=answer_payload,
        result={"processedData": combined_result},
        widgets=[
            {
                "section": fd_payload.get("title", "Result"),
                "type": (fd_payload.get("type") or "TABLE"),
                "widgetType": (fd_payload.get("type") or "TABLE"),
            }
        ],
        widget=widget_value,
        records=_extract_records(combined_result),
        analysis=analysis,
        prediction=prediction,
        conclusion=conclusion,
        intent=normalized_intent,
        formattedData=fd_payload if fd_payload else None,
        suggestedQuestions=suggested_questions or [],
        metadata=metadata,
        overallConfidence=round(float(confidence), 2),
        partialFailure=partial_failure,
        failedSections=failed_sections or [],
    )

    model_dict = response_model.model_dump(by_alias=True)
    model_dict["message"] = (intro_message or "").strip()
    model_dict["widget"] = widget_value
    model_dict["widgets"] = [
        {
            "section": fd_payload.get("title", "Result"),
            "type": (fd_payload.get("type") or "TABLE"),
            "widgetType": (fd_payload.get("type") or "TABLE"),
        }
    ]
    model_dict["sessionId"] = session_value
    return model_dict


def build_conversational_response(
    message: str,
    *,
    session_id: Optional[str] = None,
    query_type: str = "conversation",
) -> Dict[str, Any]:
    """Compatibility wrapper for conversational-only payloads."""
    return _build_conversational_payload(
        message,
        session_id=session_id or "admin-default",
        query_type=query_type,
    )


def public_response_view(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Filter the internal response dict to only expose fields matching the
    public contract, preventing security leaks of raw .NET records or LLM details.
    """
    # Old code omitted analysis, prediction, conclusion, queryType, answer, overallConfidence, and metadata keys in the public view.
    public_payload = {k: payload[k] for k in _PUBLIC_RESPONSE_KEYS if k in payload}
    
    # Filter metadata to remove internal trace ids, request ids, and timings to keep public view clean
    if "metadata" in public_payload and isinstance(public_payload["metadata"], dict):
        clean_meta = dict(public_payload["metadata"])
        clean_meta.pop("requestId", None)
        clean_meta.pop("traceId", None)
        clean_meta.pop("timings", None)
        public_payload["metadata"] = clean_meta
        
    public_payload["message"] = (payload.get("message") or payload.get("introMessage") or "").strip()
    if "widget" not in public_payload:
        widgets = payload.get("widgets") or []
        if isinstance(widgets, list) and widgets:
            first_widget = widgets[0] if isinstance(widgets[0], dict) else {}
            public_payload["widget"] = (
                payload.get("widget")
                or first_widget.get("widgetType")
                or first_widget.get("type")
                or "table"
            )
        else:
            fd = payload.get("formattedData") or {}
            if isinstance(fd, dict):
                public_payload["widget"] = fd.get("type", "table")
            else:
                public_payload["widget"] = "table"
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
