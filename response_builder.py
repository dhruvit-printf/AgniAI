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
) -> Dict[str, Any]:
    """
    Assembles the JSON response structure for the frontend.
    """
    category = intent.get("category") or "Agniveer"
    title = _get_title(query_type, intent)

    # 1. Build introMessage
    intro_message_dict = {
        "title": title,
        "description": intro_message or f"Retrieved {category.lower()} records matching request."
    }

    # 2. Build answer object (single source of truth)
    answer_dict = build_answer(query_type, combined_result, intent)

    # 3. Format analysis
    analysis_dict = {
        "summary": analysis.get("summary") or "",
        "observations": list(analysis.get("observations") or []),
        "insights": list(analysis.get("insights") or [])
    } if analysis else {
        "summary": "",
        "observations": [],
        "insights": []
    }

    # 4. Format prediction
    prediction_dict = {
        "shortTerm": prediction.get("shortTerm") or "stable",
        "futureTrends": list(prediction.get("futureTrends") or [])
    } if prediction else {
        "shortTerm": "stable",
        "futureTrends": []
    }

    # 5. Format conclusion
    conclusion_dict = {
        "message": conclusion.get("summary") or conclusion.get("message") or ""
    } if conclusion else {
        "message": ""
    }

    # Final payload structure matching targets exactly
    payload: Dict[str, Any] = {
        "status": True,
        "queryType": query_type,
        "introMessage": intro_message_dict,
        "answer": answer_dict,
        "analysis": analysis_dict,
        "prediction": prediction_dict,
        "conclusion": conclusion_dict,
        "suggestedQuestions": suggested_questions or [],
        "widgets": widgets or [],
        "metadata": {
            "executionTimeMs": 0
        }
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

    
    # Extra metadata values
    payload["metadata"]["confidence"] = round(float(confidence), 2)
    payload["metadata"]["queryType"] = query_type
    payload["metadata"]["operationCount"] = int(operation_count)

    if durations:
        payload["metadata"].update(durations)

    if session_id and session_id != "admin-default":
        payload["sessionId"] = session_id

    return payload

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
