"""
response_builder.py
===================
Final response assembly layer.

formattedData   — always a list of widget dicts {type, title, data}.
analysis        — root-level narrative string (not inside widgets).
prediction      — root-level prediction string (not inside widgets).
conclusion      — root-level conclusion string (not inside widgets).
dotnetPayload   — the exact .NET request payload that was executed, or
                  None if no .NET query was made (conversational/greeting).
                  MUST be preserved as-is; never regenerated or defaulted.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def _to_str(value: Any) -> str:
    """Convert an engine dict or plain string to a single readable string."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        # analysis engine  → {"summary": "...", "observations": [...], ...}
        # prediction engine → {"projection": "...", "heuristicEstimate": "...", ...}
        # conclusion engine → {"summary": "...", "bullets": [...]}
        text = (
            value.get("summary")
            or value.get("projection")
            or value.get("heuristicEstimate")
            or ""
        )
        return str(text).strip()
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

    # If it is a COMPARISON query:
    if meta.get("queryType") == "COMPARISON":
        # formattedData is a list of comparison widgets from build_comparison_widgets()
        if isinstance(formatted_data, list):
            fd = formatted_data
        elif isinstance(formatted_data, dict) and formatted_data:
            fd = [formatted_data]
        else:
            fd = []

        return {
            "status":             True,
            "message":            message or "",
            "formattedData":      fd,
            "analysis":           _to_str(analysis),
            "prediction":         _to_str(prediction),
            "conclusion":         _to_str(conclusion),
            "suggestedQuestions": suggested_questions or [],
            "dotnetPayload":      dotnet_payload or [],
            "metadata":           meta,
            "overallConfidence":  round(float(overall_confidence or 0.0), 2),
            "partialFailure":     bool(partial_failure),
            "failedSections":     failed_sections if isinstance(failed_sections, list) else [],
        }

    # For other query types
    if isinstance(formatted_data, list):
        fd: List[Any] = formatted_data
    elif isinstance(formatted_data, dict) and formatted_data:
        fd = [formatted_data]
    else:
        fd = []

    # Preserve dotnetPayload exactly as passed:
    #   - None   → no .NET query was executed (conversational / greeting)
    #   - dict/list → the exact payload sent to the .NET API
    resolved_dotnet_payload = dotnet_payload

    return {
        "status": True,
        "message": message or "",
        "formattedData": fd,
        # Root-level narrative — plain strings, never inside individual widgets
        "analysis":   _to_str(analysis),
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
