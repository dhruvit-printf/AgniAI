"""
response_builder.py
===================
Assembles the final JSON response payload for the admin pipeline.
Pure JSON structure, no string-based formatting or giant messages.
"""

from typing import Any, Dict, List, Optional
import re

def _extract_records(data: Any) -> List[Dict]:
    """Pull the list of records out of any wrapper shape."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "Data", "result", "Result", "records", "Records", "persons", "personnel"):
            val = data.get(key)
            if isinstance(val, list):
                return val
            if isinstance(val, dict):
                return _extract_records(val)
    return []

def _get_title(query_type: str, intent: Dict[str, Any]) -> str:
    category = intent.get("category") or "Agniveer"
    subcategory = intent.get("subcategory") or ""
    
    if query_type == "compare":
        return f"{category} Side-by-Side Comparison"
    elif query_type == "cross_filter":
        return f"{category} Cross-Filter Analysis"
    elif query_type == "multi_independent":
        return "Consolidated Module Statistics"
    elif query_type == "trend":
        return f"{category} Timeline Trend Analysis"
    elif query_type == "distribution":
        return f"{category} Distribution Breakdown"
    
    if subcategory:
        # e.g., "TopPerformers" -> "Top Performers"
        spaced = re.sub(r"([A-Z])", r" \1", subcategory).strip()
        return f"{category} {spaced}"
    return f"{category} Overview"

def build_answer(query_type: str, combined_result: Any, intent: Dict[str, Any]) -> Dict[str, Any]:
    category = intent.get("category") or "Agniveer"
    
    if query_type == "compare":
        left = combined_result.get("left") or {}
        right = combined_result.get("right") or {}
        comp = combined_result.get("comparison") or {}

        sections = [
            {
                "label": left.get("label") or "Side 1",
                "type": "compare",
                "data": left.get("data") or []
            },
            {
                "label": right.get("label") or "Side 2",
                "type": "compare",
                "data": right.get("data") or []
            }
        ]
        return {
            "sections": sections,
            "left": left,
            "right": right,
            "comparison": comp
        }
    elif query_type == "multi_independent":
        sections = combined_result.get("sections") or []
        return {
            "sections": sections
        }
    elif query_type == "cross_filter":
        records = combined_result.get("records") if isinstance(combined_result, dict) else _extract_records(combined_result)
        sections = [
            {
                "label": "Common Records",
                "type": "cross_filter",
                "data": records
            }
        ]
        return {
            "sections": sections
        }
    else:
        records = _extract_records(combined_result)
        sections = [
            {
                "label": "Result",
                "type": query_type,
                "data": records
            }
        ]
        answer_dict = {
            "sections": sections
        }
        if isinstance(combined_result, dict):
            for k in ("chartData", "granularity", "trendDirection", "labels", "values", "groupBy"):
                if k in combined_result:
                    answer_dict[k] = combined_result[k]
        return answer_dict

from typing import Any, Dict, List, Optional, Generator

def calculate_section_confidence(section: Dict[str, Any], intent: Dict[str, Any], api_success: bool = True) -> float:
    if not api_success:
        return 0.0
    
    # Base confidence from intent or default
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

    # Record count effect
    records = section.get("data") or []
    rec_count = len(records)
    
    # Adjust based on record count and completeness
    record_factor = 0.0
    if rec_count == 0:
        record_factor = -0.05
    elif rec_count > 0:
        # Check completeness of first few records
        missing_fields = 0
        total_fields = 0
        for r in records[:5]:
            if isinstance(r, dict):
                for k, v in r.items():
                    total_fields += 1
                    if v is None or v == "":
                        missing_fields += 1
        if total_fields > 0:
            completeness = (total_fields - missing_fields) / total_fields
            record_factor = 0.05 * completeness
        else:
            record_factor = 0.05

    final_conf = base_conf + record_factor
    
    # Filter matching adjustment
    if section.get("failedFilters") or section.get("degraded"):
        final_conf -= 0.15

    return round(max(0.0, min(1.0, final_conf)), 2)

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
) -> Dict[str, Any]:
    """
    Assembles the JSON response structure for the frontend.
    """
    failed_sections_list = failed_sections or []
    category = intent.get("category") or "Agniveer"
    title = _get_title(query_type, intent)

    # 1. Build introMessage
    intro_message_dict = {
        "title": title,
        "description": intro_message or f"Retrieved {category.lower()} records matching request."
    }

    # 2. Build answer object (single source of truth)
    answer_dict = build_answer(query_type, combined_result, intent)

    # Calculate section-level confidence and recordCount
    sections = answer_dict.get("sections") or []
    section_confidences = []
    for sec in sections:
        label = sec.get("label") or ""
        sec_failed = label in failed_sections_list
        sec_data = sec.get("data") or []
        if isinstance(sec_data, list) and len(sec_data) == 1:
            if isinstance(sec_data[0], dict) and sec_data[0].get("unavailable"):
                sec_failed = True

        sec_conf = calculate_section_confidence(sec, intent, api_success=not sec_failed)
        sec["confidence"] = sec_conf
        sec["recordCount"] = len(sec_data) if not sec_failed else 0
        section_confidences.append(sec_conf)

    if section_confidences:
        overall_conf = round(sum(section_confidences) / len(section_confidences), 2)
    else:
        overall_conf = round(float(confidence), 2)

    # 3. Format analysis
    analysis_dict = {
        "summary": analysis.get("summary") or "",
        "observations": list(analysis.get("observations") or []),
        "insights": list(analysis.get("insights") or [])
    } if analysis else None

    # 4. Format prediction
    prediction_dict = None
    if prediction:
        prediction_dict = {
            "trend": prediction.get("trend") or "Stable",
            "forecast": prediction.get("forecast") or (prediction.get("futureTrends")[0] if prediction.get("futureTrends") else "Metrics are expected to align with historical standards."),
            "shortTerm": prediction.get("shortTerm") or (prediction.get("trend") or "Stable").lower(),
            "futureTrends": list(prediction.get("futureTrends") or [prediction.get("forecast")])
        }

    # 5. Format conclusion
    conclusion_dict = {
        "summary": conclusion.get("summary") or conclusion.get("message") or "",
        "message": conclusion.get("message") or conclusion.get("summary") or ""
    } if conclusion else None

    # Final payload structure matching targets exactly
    from telemetry import request_id_var, trace_id_var, session_id_var
    req_id = request_id_var.get("N/A")
    tr_id = trace_id_var.get("N/A")
    sess_id = session_id_var.get("N/A")

    payload: Dict[str, Any] = {
        "status": True,
        "queryType": query_type,
        "introMessage": intro_message_dict,
        "formattedData": {},
        "answer": answer_dict,
        "analysis": analysis_dict,
        "prediction": prediction_dict,
        "conclusion": conclusion_dict,
        "suggestedQuestions": suggested_questions or [],
        "widgets": widgets or [],
        "metadata": {
            "requestId": req_id,
            "traceId": tr_id,
            "sessionId": sess_id,
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
        },
        "overallConfidence": overall_conf,
        "partialFailure": partial_failure,
        "failedSections": failed_sections_list
    }

    # Support compatibility properties
    payload["result"] = {"processedData": combined_result}
    
    # Normalize intent confidence for compatibility/tests
    intent_dict = dict(intent) if intent else {}
    conf_val = intent_dict.get("confidence")
    if isinstance(conf_val, str):
        conf_lower = conf_val.lower()
        if "high" in conf_lower:
            intent_dict["confidence"] = 0.95
        elif "medium" in conf_lower:
            intent_dict["confidence"] = 0.70
        elif "low" in conf_lower:
            intent_dict["confidence"] = 0.30
        else:
            try:
                intent_dict["confidence"] = float(conf_val)
            except ValueError:
                intent_dict["confidence"] = float(confidence)
    elif conf_val is not None:
        try:
            intent_dict["confidence"] = float(conf_val)
        except (TypeError, ValueError):
            intent_dict["confidence"] = float(confidence)
    else:
        intent_dict["confidence"] = float(confidence)
    
    payload["intent"] = intent_dict

    # Add durations to metadata
    if durations:
        payload["metadata"].update(durations)

    if session_id and session_id != "admin-default":
        payload["sessionId"] = session_id

    return payload

def stream_response_chunks(payload: Dict[str, Any]) -> Generator[Dict[str, Any], None, None]:
    """
    Prepare response builder for future websocket streaming.
    Yields each major section of the response independently.
    """
    keys = (
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
            analysis_parts.append(
                "Observations:\n" + "\n".join(f"- {o}" for o in clean_obs)
            )

        insights = analysis.get("insights") or []
        clean_ins = [i.strip() for i in insights if i and i.strip()]
        if clean_ins:
            analysis_parts.append(
                "Insights:\n" + "\n".join(f"- {i}" for i in clean_ins)
            )

        pred = analysis.get("predictions") or []
        clean_pred = [p.strip() for p in pred if p and p.strip()]
        if clean_pred:
            analysis_parts.append(
                "Predictions:\n" + "\n".join(f"- {p}" for p in clean_pred)
            )

        if analysis_parts:
            parts.append("Analysis:\n" + "\n\n".join(analysis_parts))

    if conclusion:
        conclusion_summary = (conclusion.get("summary") or conclusion.get("message") or "").strip()
        if conclusion_summary:
            parts.append(f"Conclusion:\n{conclusion_summary}")

    return "\n\n".join(parts)
