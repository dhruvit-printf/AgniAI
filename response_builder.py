"""
response_builder.py
===================
Final response assembly layer.

formattedData   — a single widget dict when one widget exists, otherwise a list.
summary         — unified root-level narrative string.
analysis        — root-level narrative string (not inside widgets).
prediction      — root-level prediction string (not inside widgets).
conclusion      — root-level conclusion string (not inside widgets).
dotnetPayload   — the exact .NET request payload that was executed, or
                  None if no .NET query was made (conversational/greeting).
                  MUST be preserved as-is; never regenerated or defaulted.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _has_data(fd: Any) -> bool:
    if not fd:
        return False
    if isinstance(fd, list):
        return any(_has_data(w) for w in fd)
    if isinstance(fd, dict):
        d = fd.get("data")
        if not d:
            return False
        if isinstance(d, dict):
            if "row" in d and not d["row"]:
                return False
            if "rows" in d and not d["rows"]:
                return False
            if "left" in d and "right" in d:
                if not _has_data({"data": d["left"]}) and not _has_data({"data": d["right"]}):
                    return False
        return True
    return True


def _to_str(value: Any) -> str:
    """Convert an engine dict or plain string to a full readable string.

    The analysis/prediction/conclusion engines each return a structured dict
    with a headline plus supporting detail (insights, bullets, future trends).
    Previously only the headline field was kept and everything else was
    silently dropped. Compose all of it into one coherent narrative instead.
    """
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts: List[str] = []

        # analysis engine  → {"summary": "...", "insights": [...], "statistics": {...}}
        # conclusion engine → {"summary": "...", "bullets": [...]}
        headline = value.get("summary") or value.get("projection") or ""
        if headline:
            parts.append(str(headline).strip())

        for detail in value.get("insights") or value.get("bullets") or []:
            detail_str = str(detail).strip()
            if detail_str and detail_str not in parts:
                parts.append(detail_str)

        # prediction engine → {"projection": "...", "heuristicEstimate": "...", "futureTrends": [...]}
        heuristic = value.get("heuristicEstimate")
        if heuristic and str(heuristic).strip() not in parts:
            parts.append(str(heuristic).strip())

        for trend in value.get("futureTrends") or []:
            trend_str = str(trend).strip()
            if trend_str and trend_str not in parts:
                parts.append(trend_str)

        if not parts:
            return ""

        return " ".join(p if p.endswith((".", "!", "?")) else f"{p}." for p in parts)
    return str(value).strip()


def build_response(
    message: str,
    formatted_data: Any,
    metadata: Optional[Dict[str, Any]],
    session_id: str,
    analysis: Optional[Any] = None,
    prediction: Optional[Any] = None,
    conclusion: Optional[Any] = None,
    suggested_questions: Optional[List[str]] = None,
    dotnet_payload: Optional[Any] = None,
    overall_confidence: float = 0.0,
    partial_failure: bool = False,
    failed_sections: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    meta = dict(metadata) if isinstance(metadata, dict) else {}
    if "sessionId" not in meta:
        meta["sessionId"] = session_id

    summary = " ".join(
        part for part in (_to_str(analysis), _to_str(prediction), _to_str(conclusion)) if part
    ).strip()

    def _normalize_formatted_data(value: Any) -> Any:
        if isinstance(value, list):
            if len(value) == 1:
                return value[0]
            return value
        if isinstance(value, dict):
            return value
        return value

    # If it is a COMPARISON query:
    if meta.get("queryType") == "COMPARISON":
        # formattedData is a list of comparison widgets from build_comparison_widgets()
        fd = _normalize_formatted_data(formatted_data)

        response = {
            "status": True,
            "message": message or "",
            "formattedData": fd,
            "summary": summary,
            "analysis": _to_str(analysis),
            "prediction": _to_str(prediction),
            "conclusion": _to_str(conclusion),
            "suggestedQuestions": suggested_questions or [],
            "dotnetPayload": dotnet_payload,
            "metadata": meta,
            "overallConfidence": round(float(overall_confidence or 0.0), 2),
            "partialFailure": bool(partial_failure),
            "failedSections": (
                failed_sections if isinstance(failed_sections, list) else []
            ),
        }
        if not _has_data(fd):
            response.pop("formattedData", None)
        return response

    # For other query types
    fd = _normalize_formatted_data(formatted_data)

    # Preserve dotnetPayload exactly as passed:
    #   - None   → no .NET query was executed (conversational / greeting)
    #   - dict/list → the exact payload sent to the .NET API
    resolved_dotnet_payload = dotnet_payload

    response = {
        "status": True,
        "message": message or "",
        "formattedData": fd,
        "summary": summary,
        # Root-level narrative — plain strings, never inside individual widgets
        "analysis": _to_str(analysis),
        "prediction": _to_str(prediction),
        "conclusion": _to_str(conclusion),
        "suggestedQuestions": suggested_questions or [],
        # Exact .NET request payload; None when no backend call was made
        "dotnetPayload": resolved_dotnet_payload,
        "metadata": meta,
        "overallConfidence": round(float(overall_confidence or 0.0), 2),
        "partialFailure": bool(partial_failure),
        "failedSections": failed_sections if isinstance(failed_sections, list) else [],
    }
    
    if not _has_data(fd):
        response.pop("formattedData", None)
        
    return response
