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
    combine_analysis_to_string,
    calculate_section_confidence as _calculate_section_confidence,
    combine_conclusion_to_string,
    combine_prediction_to_string,
    normalize_intent_confidence,
)
from conversation_detector import build_conversational_response as _build_conversational_payload

_PUBLIC_RESPONSE_KEYS = (
    "status",
    "sessionId",
    "message",
    "widget",
    "formattedData",
    "suggestedQuestions",
    "dotnetPayload",
    "metadata",
)

_DEFAULT_ANALYSIS = {"summary": "", "insights": [], "statistics": {}}
_DEFAULT_CONCLUSION = {"summary": "", "bullets": []}
_DEFAULT_FORMATTED_DATA = {
    "type": "TABLE",
    "title": "Data Summary",
    "data": {"columns": [], "rows": []},
    "analysis": dict(_DEFAULT_ANALYSIS),
    "prediction": None,
    "conclusion": dict(_DEFAULT_CONCLUSION),
}
_NO_DATA_FALLBACKS = {
    "compare": "I could not complete the comparison because there was not enough matching information on one or both sides. Please try again with broader filters.",
    "cross_filter": "I checked the selected conditions, but nothing matched all of them. Please try a wider search.",
    "multi_independent": "I gathered the sections you asked for, but none of them returned anything to show.",
    "trend": "I could not find enough matching information to describe a clear pattern. Please try a broader search.",
    "distribution": "I could not find enough matching information to break down the result set. Please try a broader search.",
}


def _normalize_analysis_block(analysis: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not analysis:
        return {"summary": "", "insights": [], "statistics": {}}
    if isinstance(analysis, str):
        return {"summary": analysis.strip(), "insights": [], "statistics": {}}

    insights = analysis.get("insights") or []
    if isinstance(insights, list):
        clean_insights = [str(item).strip() for item in insights if str(item).strip()]
    elif isinstance(insights, str):
        clean_insights = [insights.strip()] if insights.strip() else []
    else:
        clean_insights = []

    statistics = analysis.get("statistics") or {}
    if not isinstance(statistics, dict):
        statistics = {}

    return {
        "summary": (analysis.get("summary") or "").strip(),
        "insights": clean_insights,
        "statistics": statistics,
    }


def _normalize_conclusion_block(conclusion: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if not conclusion:
        return {"summary": "", "bullets": []}
    if isinstance(conclusion, str):
        return {"summary": conclusion.strip(), "bullets": []}

    bullets = conclusion.get("bullets") or []
    if isinstance(bullets, list):
        clean_bullets = [str(item).strip() for item in bullets if str(item).strip()]
    elif isinstance(bullets, str):
        clean_bullets = [bullets.strip()] if bullets.strip() else []
    else:
        clean_bullets = []

    return {
        "summary": (conclusion.get("summary") or conclusion.get("message") or "").strip(),
        "bullets": clean_bullets,
    }


def _normalize_prediction_block(prediction: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not prediction:
        return {
            "summary": "I do not have enough information to say what comes next.",
            "confidence": 0.0,
            "forecast": ["I do not have enough information to say what comes next."],
        }
    if isinstance(prediction, str):
        summary = prediction.strip()
        if not summary:
            return {
                "summary": "I do not have enough information to say what comes next.",
                "confidence": 0.0,
                "forecast": ["I do not have enough information to say what comes next."],
            }
        return {"summary": summary, "confidence": 0.0, "forecast": []}

    forecast = prediction.get("forecast") or prediction.get("futureTrends") or []
    if isinstance(forecast, list):
        clean_forecast = [str(item).strip() for item in forecast if str(item).strip()]
    elif isinstance(forecast, str):
        clean_forecast = [forecast.strip()] if forecast.strip() else []
    else:
        clean_forecast = []

    confidence = prediction.get("confidence")
    try:
        confidence_value = float(confidence) if confidence is not None else 0.0
    except (TypeError, ValueError):
        confidence_value = 0.0

    summary = (prediction.get("summary") or prediction.get("projection") or "").strip()
    if not summary and clean_forecast:
        summary = clean_forecast[0]

    if not summary and not clean_forecast:
        return {
            "summary": "I do not have enough information to say what comes next.",
            "confidence": round(max(0.0, min(1.0, confidence_value)), 2),
            "forecast": ["I do not have enough information to say what comes next."],
        }

    return {
        "summary": summary,
        "confidence": round(max(0.0, min(1.0, confidence_value)), 2),
        "forecast": clean_forecast,
    }


def _has_any_records(combined_result: Any, answer_payload: Optional[Dict[str, Any]] = None) -> bool:
    if isinstance(answer_payload, dict):
        sections = answer_payload.get("sections") or []
        for section in sections:
            if isinstance(section, dict) and section.get("data"):
                return True
    return bool(_extract_records(combined_result))


def _build_no_data_message(query_type: str, intent: Dict[str, Any]) -> str:
    query_key = (query_type or "").strip().lower()
    if query_key in _NO_DATA_FALLBACKS:
        return _NO_DATA_FALLBACKS[query_key]

    category = (intent.get("category") or "").strip().lower()
    if category:
        return f"I could not find any matching {category} information right now. Please try a broader search."
    return "I could not find any matching information right now. Please try a broader search."


def _build_natural_answer_message(
    *,
    query_type: str,
    intro_message: str,
    combined_result: Any,
    answer_payload: Optional[Dict[str, Any]],
    analysis: Optional[Dict[str, Any]],
    prediction: Optional[Dict[str, Any]],
    conclusion: Optional[Dict[str, Any]],
    intent: Dict[str, Any],
    has_data: bool,
) -> str:
    if not has_data:
        return _build_no_data_message(query_type, intent)

    parts: List[str] = []

    intro = (intro_message or "").strip()
    if intro:
        parts.append(intro)

    data_summary = _build_formatted_summary(
        query_type=query_type,
        combined_result=combined_result,
        answer_dict=answer_payload or {},
    )
    if data_summary:
        parts.append(data_summary)

    analysis_text = combine_analysis_to_string(analysis)
    if analysis_text:
        parts.append(f"Here is what stands out: {analysis_text}")

    prediction_text = combine_prediction_to_string(prediction)
    if prediction_text:
        parts.append(f"Looking ahead, {prediction_text}")

    conclusion_text = combine_conclusion_to_string(conclusion)
    if conclusion_text:
        parts.append(f"To wrap up, {conclusion_text}")

    if not parts:
        return "I have prepared a summary of the matching information."

    return "\n\n".join(parts)


def _build_exact_formatted_data(
    formatted_data: Any,
    *,
    analysis: Optional[Dict[str, Any]],
    prediction: Optional[Dict[str, Any]],
    conclusion: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    if isinstance(formatted_data, dict):
        fd_payload = dict(formatted_data)
    else:
        fd_payload = {}

    fd_payload = {
        "type": fd_payload.get("type") or _DEFAULT_FORMATTED_DATA["type"],
        "title": fd_payload.get("title") or _DEFAULT_FORMATTED_DATA["title"],
        "data": fd_payload.get("data")
        if isinstance(fd_payload.get("data"), dict)
        else _DEFAULT_FORMATTED_DATA["data"],
        "analysis": _normalize_analysis_block(analysis),
        "prediction": _normalize_prediction_block(prediction),
        "conclusion": _normalize_conclusion_block(conclusion),
    }
    return fd_payload


def _merge_answer_into_formatted_data(
    formatted_data: Dict[str, Any],
    answer_payload: Optional[Dict[str, Any]],
    *,
    intro_message: str,
    message_value: str,
) -> Dict[str, Any]:
    """
    Present one combined data object to the caller.

    The structured table/chart data stays under `data`, while the answer
    sections and the intro text are also surfaced on the same object so the
    caller does not need to juggle separate shapes.
    """
    combined = dict(formatted_data or {})
    answer_dict = answer_payload if isinstance(answer_payload, dict) else {}
    base_data = combined.get("data")
    if not isinstance(base_data, dict):
        base_data = {}
    merged_data = dict(base_data)

    if isinstance(answer_dict, dict):
        for key, value in answer_dict.items():
            if key == "sections":
                merged_data["sections"] = value
            elif key not in merged_data:
                merged_data[key] = value

    combined["data"] = merged_data

    return combined


def _build_exact_metadata(
    *,
    session_id: str,
    confidence: float,
    query_type: str,
    operation_count: int,
    durations: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    durations_dict = durations or {}
    planner_ms = durations_dict.get("plannerMs")
    if planner_ms is None:
        planner_ms = durations_dict.get("plannerDurationMs") or durations_dict.get("planningMs") or durations_dict.get("planning_ms") or 0
    intent_ms = durations_dict.get("intentMs")
    if intent_ms is None:
        intent_ms = durations_dict.get("intentDurationMs") or durations_dict.get("intent_duration") or 0
    dotnet_ms = durations_dict.get("dotnetMs")
    if dotnet_ms is None:
        dotnet_ms = durations_dict.get("dotnetDurationMs") or durations_dict.get("dotnet_duration") or 0
    combiner_ms = durations_dict.get("combinerMs")
    if combiner_ms is None:
        combiner_ms = durations_dict.get("combineDurationMs") or durations_dict.get("combiner_duration") or 0
    report_ms = durations_dict.get("reportMs")
    if report_ms is None:
        report_ms = durations_dict.get("report_duration") or durations_dict.get("analysisDurationMs") or 0
    total_ms = durations_dict.get("totalMs")
    if total_ms is None:
        total_ms = durations_dict.get("totalDurationMs") or durations_dict.get("executionTimeMs") or durations_dict.get("total_duration") or 0

    execution_ms = durations_dict.get("executionTimeMs")
    if execution_ms is None:
        execution_ms = total_ms

    return {
        "sessionId": session_id,
        "confidence": round(float(confidence), 2),
        "queryType": query_type,
        "operationCount": int(operation_count),
        "timings": {
            "plannerMs": round(float(planner_ms), 2),
            "intentMs": round(float(intent_ms), 2),
            "dotnetMs": round(float(dotnet_ms), 2),
            "combinerMs": round(float(combiner_ms), 2),
            "reportMs": round(float(report_ms), 2),
            "totalMs": round(float(total_ms), 2),
        },
        "executionTimeMs": round(float(execution_ms), 2),
    }


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
    dotnet_payload: Optional[Any] = None,
    partial_failure: bool = False,
    failed_sections: Optional[List[str]] = None,
    answer_dict: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Assemble the final response payload strictly using the canonical schema.
    Fault-tolerant: if the schema model fails, returns a minimal valid response
    preserving message and formattedData.
    """
    import logging as _log
    _logger = _log.getLogger(__name__)

    from schemas import FinalResponse
    from telemetry import session_id_var

    session_value = session_id or session_id_var.get("admin-default") or "admin-default"
    metadata = _build_exact_metadata(
        session_id=session_value,
        confidence=confidence,
        query_type=query_type,
        operation_count=operation_count,
        durations=durations,
    )
    fd_payload = _build_exact_formatted_data(
        formatted_data,
        analysis=analysis,
        prediction=prediction,
        conclusion=conclusion,
    )
    widget_value = (fd_payload.get("type") or "TABLE")
    answer_payload = (
        answer_dict
        if isinstance(answer_dict, dict)
        else build_answer(query_type, combined_result, intent)
    )
    message_value = _build_natural_answer_message(
        query_type=query_type,
        intro_message=intro_message,
        combined_result=combined_result,
        answer_payload=answer_payload,
        analysis=analysis,
        prediction=prediction,
        conclusion=conclusion,
        intent=intent,
        has_data=_has_any_records(combined_result, answer_payload),
    )
    combined_formatted_data = _merge_answer_into_formatted_data(
        fd_payload,
        answer_payload,
        intro_message=intro_message,
        message_value=message_value,
    )

    try:
        response_model = FinalResponse(
            status=True,
            sessionId=session_value,
            message=message_value,
            widget=widget_value,
            formattedData=combined_formatted_data,
            suggestedQuestions=suggested_questions or [],
            dotnetPayload=dotnet_payload if dotnet_payload is not None else {},
            metadata=metadata,
        )

        model_dict = response_model.model_dump(by_alias=True)
    except Exception as schema_exc:
        _logger.error("FinalResponse schema construction failed: %s", schema_exc, exc_info=True)
        model_dict = {
            "status": True,
            "sessionId": session_value,
            "message": message_value,
            "widget": widget_value,
            "formattedData": combined_formatted_data,
            "suggestedQuestions": suggested_questions or [],
            "dotnetPayload": dotnet_payload if dotnet_payload is not None else {},
            "metadata": metadata,
        }

    model_dict["message"] = message_value
    model_dict["introMessage"] = (intro_message or "").strip()
    model_dict["widget"] = widget_value
    model_dict["sessionId"] = session_value
    model_dict["queryType"] = query_type
    model_dict["formattedData"] = combined_formatted_data
    model_dict["answer"] = combined_formatted_data
    model_dict["analysis"] = combine_analysis_to_string(analysis)
    model_dict["prediction"] = _normalize_prediction_block(prediction)
    model_dict["conclusion"] = combine_conclusion_to_string(conclusion)
    model_dict["overallConfidence"] = round(float(confidence), 2)
    model_dict["result"] = {"processedData": combined_result}
    model_dict["intent"] = normalize_intent_confidence(intent, confidence)
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
    if not isinstance(payload, dict):
        payload = {}

    formatted = payload.get("formattedData") if isinstance(payload, dict) else None
    metadata = payload.get("metadata") if isinstance(payload, dict) else None
    if not isinstance(formatted, dict):
        formatted = {}

    clean_formatted = _build_exact_formatted_data(
        formatted,
        analysis=formatted.get("analysis"),
        prediction=formatted.get("prediction"),
        conclusion=formatted.get("conclusion"),
    )

    clean_meta = metadata if isinstance(metadata, dict) else {}
    clean_meta = {
        "sessionId": clean_meta.get("sessionId") or payload.get("sessionId") or "admin-default",
        "metrics": {
            "confidence": round(float(clean_meta.get("confidence") or 0.0), 2),
            "queryType": clean_meta.get("queryType") or "",
            "operationCount": int(clean_meta.get("operationCount") or 0),
        },
        "executionTimeMs": round(float(clean_meta.get("executionTimeMs") or 0), 2),
    }

    public_payload = {
        "status": bool(payload.get("status", True)),
        "sessionId": payload.get("sessionId") or clean_meta["sessionId"],
        "message": (payload.get("introMessage") or payload.get("message") or "").strip(),
        "formattedData": clean_formatted,
        "suggestedQuestions": list(payload.get("suggestedQuestions") or []),
        "queryType": payload.get("queryType") or clean_meta["queryType"],
        "overallConfidence": round(float(payload.get("overallConfidence") or clean_meta["confidence"]), 2),
        "metadata": clean_meta,
    }
    return public_payload


def stream_response_chunks(payload: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Prepare response builder for future websocket streaming.
    Yields each major section of the response independently.
    """
    keys = (
        "status",
        "sessionId",
        "message",
        "widget",
        "formattedData",
        "suggestedQuestions",
        "dotnetPayload",
        "metadata",
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
