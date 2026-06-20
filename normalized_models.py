"""
normalized_models.py
====================
Canonical normalization helpers for the admin pipeline.

This module keeps the shared JSON/dict-shape normalization logic in one place
so the rest of the pipeline can stay thin and predictable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


_RECORD_KEY_CANDIDATES = (
    "agniveerNo",
    "agniveerId",
    "AgniveerId",
    "AgniVeerId",
    "id",
    "Id",
)

_WRAPPER_KEY_CANDIDATES = (
    "data",
    "Data",
    "result",
    "Result",
    "records",
    "Records",
    "persons",
    "personnel",
)

_METRIC_FIELDS = (
    "bestTotal",
    "totalMarks",
    "score",
    "Score",
    "omrInputTotal",
    "marksObtained",
)


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def extract_records(data: Any) -> List[Dict]:
    """
    Canonical record unwrapping for any backend shape.

    Supports nested dict wrappers, list payloads, and a handful of legacy
    response aliases used across the repository.
    """
    if isinstance(data, dict):
        for key in _WRAPPER_KEY_CANDIDATES:
            val = data.get(key)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                nested = extract_records(val)
                if nested:
                    return nested

        teams = data.get("teams") or data.get("Teams")
        if isinstance(teams, list):
            members: List[Dict] = []
            for team in teams:
                if isinstance(team, dict):
                    team_members = team.get("members") or team.get("Members") or []
                    if isinstance(team_members, list):
                        members.extend(
                            member for member in team_members if isinstance(member, dict)
                        )
            if members:
                return members

        for value in data.values():
            if isinstance(value, (dict, list)):
                nested = extract_records(value)
                if nested:
                    return nested

    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def extract_record_id(record: Dict[str, Any]) -> Optional[str]:
    for key in _RECORD_KEY_CANDIDATES:
        value = record.get(key)
        if value is not None:
            return str(value).strip()
    return None


def normalize_dotnet_response(data: Any) -> Dict[str, Any]:
    """
    Attach a canonical `records` list to any backend response shape.
    """
    records = extract_records(data)
    if isinstance(data, dict):
        normalized = dict(data)
        normalized["records"] = records
        return normalized
    return {"records": records}


def _get_score(record: Dict[str, Any]) -> Optional[float]:
    for field in _METRIC_FIELDS:
        score = _safe_float(record.get(field))
        if score is not None:
            return score
    return None


def calculate_section_confidence(
    section: Dict[str, Any],
    intent: Dict[str, Any],
    api_success: bool = True,
) -> float:
    if not api_success:
        return 0.0

    conf_val = intent.get("confidence", 0.95)
    if isinstance(conf_val, str):
        conf_lower = conf_val.lower()
        if "high" in conf_lower:
            base_conf = 0.95
        elif "medium" in conf_lower:
            base_conf = 0.70
        elif "low" in conf_lower:
            base_conf = 0.30
        else:
            try:
                base_conf = float(conf_val)
            except ValueError:
                base_conf = 0.85
    elif isinstance(conf_val, (int, float)):
        base_conf = float(conf_val)
    else:
        base_conf = 0.85

    records = section.get("data") or []
    rec_count = len(records)
    record_factor = 0.0
    if rec_count == 0:
        record_factor = -0.05
    elif rec_count > 0:
        missing_fields = 0
        total_fields = 0
        for record in records[:5]:
            if isinstance(record, dict):
                for _, value in record.items():
                    total_fields += 1
                    if value is None or value == "":
                        missing_fields += 1
        if total_fields > 0:
            completeness = (total_fields - missing_fields) / total_fields
            record_factor = 0.05 * completeness
        else:
            record_factor = 0.05

    final_conf = base_conf + record_factor
    if section.get("failedFilters") or section.get("degraded"):
        final_conf -= 0.15

    return round(max(0.0, min(1.0, final_conf)), 2)


def normalize_intent_confidence(
    intent: Dict[str, Any],
    fallback_confidence: float,
) -> Dict[str, Any]:
    normalized = dict(intent or {})
    conf_val = normalized.get("confidence")
    if isinstance(conf_val, str):
        conf_lower = conf_val.lower()
        if "high" in conf_lower:
            normalized["confidence"] = 0.95
        elif "medium" in conf_lower:
            normalized["confidence"] = 0.70
        elif "low" in conf_lower:
            normalized["confidence"] = 0.30
        else:
            try:
                normalized["confidence"] = float(conf_val)
            except ValueError:
                normalized["confidence"] = float(fallback_confidence)
    elif conf_val is not None:
        try:
            normalized["confidence"] = float(conf_val)
        except (TypeError, ValueError):
            normalized["confidence"] = float(fallback_confidence)
    else:
        normalized["confidence"] = float(fallback_confidence)
    return normalized


def normalize_prediction(
    prediction: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not prediction:
        return None

    projection = prediction.get("projection") or prediction.get("forecast") or (
        prediction.get("futureTrends")[0] if prediction.get("futureTrends") else ""
    )
    if not projection:
        projection = "Metrics are expected to align with historical standards."

    heuristic_estimate = prediction.get("heuristicEstimate") or projection
    trend = prediction.get("trend") or "Stable"

    return {
        "trend": trend,
        "projection": projection,
        "heuristicEstimate": heuristic_estimate,
        "shortTerm": prediction.get("shortTerm") or str(trend).lower(),
        "futureTrends": list(prediction.get("futureTrends") or [heuristic_estimate]),
    }


def build_intro_message(title: str, intro_message: str, category: str) -> Dict[str, Any]:
    return {
        "title": title,
        "description": intro_message or f"Retrieved {category.lower()} records matching request.",
    }


def assemble_response_metadata(
    *,
    confidence: float,
    query_type: str,
    operation_count: int,
    durations: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    from telemetry import request_id_var, session_id_var, trace_id_var

    metadata: Dict[str, Any] = {
        "requestId": request_id_var.get("N/A"),
        "traceId": trace_id_var.get("N/A"),
        "sessionId": session_id_var.get("N/A"),
        "executionTimeMs": 0,
        "intentDurationMs": 0,
        "dotnetDurationMs": 0,
        "combineDurationMs": 0,
        "analysisDurationMs": 0,
        "predictionDurationMs": 0,
        "conclusionDurationMs": 0,
        "totalDurationMs": 0,
        "confidence": round(float(confidence), 2),
        "queryType": query_type,
        "operationCount": int(operation_count),
    }
    if durations:
        metadata.update(durations)
    return metadata


def assemble_admin_response(
    *,
    query_type: str,
    intro_message: str,
    combined_result: Any,
    analysis: Optional[Dict[str, Any]],
    conclusion: Optional[Dict[str, Any]],
    intent: Dict[str, Any],
    confidence: float,
    operation_count: int,
    session_id: Optional[str] = None,
    durations: Optional[Dict[str, float]] = None,
    widgets: Optional[List[Dict[str, Any]]] = None,
    suggested_questions: Optional[List[str]] = None,
    prediction: Optional[Dict[str, Any]] = None,
    partial_failure: bool = False,
    failed_sections: Optional[List[str]] = None,
) -> Dict[str, Any]:
    from response_builder import build_answer

    failed_sections_list = failed_sections or []
    normalized_intent = normalize_intent_confidence(intent, confidence)
    answer_dict = build_answer(query_type, combined_result, normalized_intent)
    sections = answer_dict.get("sections") or []

    section_confidences = []
    for section in sections:
        label = section.get("label") or ""
        sec_failed = label in failed_sections_list
        sec_data = section.get("data") or []
        if isinstance(sec_data, list) and len(sec_data) == 1:
            if isinstance(sec_data[0], dict) and sec_data[0].get("unavailable"):
                sec_failed = True
        sec_conf = calculate_section_confidence(
            section, normalized_intent, api_success=not sec_failed
        )
        section["confidence"] = sec_conf
        section["recordCount"] = len(sec_data) if not sec_failed else 0
        section_confidences.append(sec_conf)

    if section_confidences:
        overall_conf = round(sum(section_confidences) / len(section_confidences), 2)
    else:
        overall_conf = round(float(confidence), 2)

    analysis_dict = None
    if analysis:
        analysis_dict = {
            "summary": analysis.get("summary") or "",
            "observations": list(analysis.get("observations") or []),
            "insights": list(analysis.get("insights") or []),
        }

    conclusion_dict = None
    if conclusion:
        conclusion_dict = {
            "summary": conclusion.get("summary") or conclusion.get("message") or "",
            "message": conclusion.get("message") or conclusion.get("summary") or "",
        }

    normalized_prediction = normalize_prediction(prediction)
    metadata = assemble_response_metadata(
        confidence=confidence,
        query_type=query_type,
        operation_count=operation_count,
        durations=durations,
    )

    payload: Dict[str, Any] = {
        "status": True,
        "queryType": query_type,
        "introMessage": build_intro_message(
            _derive_title(query_type, normalized_intent), intro_message, normalized_intent.get("category") or "Agniveer"
        ),
        "formattedData": {},
        "answer": answer_dict,
        "analysis": analysis_dict,
        "prediction": normalized_prediction,
        "conclusion": conclusion_dict,
        "suggestedQuestions": suggested_questions or [],
        "widgets": widgets or [],
        "metadata": metadata,
        "overallConfidence": overall_conf,
        "partialFailure": partial_failure,
        "failedSections": failed_sections_list,
        "result": {"processedData": combined_result},
        "intent": normalized_intent,
    }

    if session_id and session_id != "admin-default":
        payload["sessionId"] = session_id

    return payload


def _derive_title(query_type: str, intent: Dict[str, Any]) -> str:
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
        import re

        spaced = re.sub(r"([A-Z])", r" \1", subcategory).strip()
        return f"{category} {spaced}"
    return f"{category} Overview"
