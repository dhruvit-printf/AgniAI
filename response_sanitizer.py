"""Response sanitization helpers for public API outputs.

Universal response contract (all fields mandatory, no extras):

{
    "status":             bool,
    "message":            str,
    "formattedData":      { "type": str, "title": str, "data": dict } | [ ... ],
    "summary":            str,
    "analysis":           str,
    "prediction":         str,
    "conclusion":         str,
    "suggestedQuestions": [ str ],
    "dotnetPayload":      list | dict | null,
    "sessionId":          str
}

Internal-only fields (metadata, overallConfidence, partialFailure,
failedSections, comparisonMetrics) stay on the pipeline's internal payload
for telemetry/validation but are never exposed to the frontend.
"""

from __future__ import annotations

from typing import Any, Dict, List, Union

# Only these three keys are exposed per widget.
# id, analysis, prediction, conclusion are internal — never sent to frontend.
_ALLOWED_WIDGET_KEYS = ("type", "title", "data")


def _normalize_text(value: Any) -> Any:
    """Normalize string-like values without changing the public response shape."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return value


def _clean_widget(widget: Any) -> Dict[str, Any]:
    if not isinstance(widget, dict):
        return {}
    return {key: widget[key] for key in _ALLOWED_WIDGET_KEYS if key in widget}


def _clean_formatted_data(
    formatted: Any,
) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Preserve single-widget objects and lists of widgets."""
    if isinstance(formatted, list):
        cleaned = [
            _clean_widget(widget) for widget in formatted if isinstance(widget, dict)
        ]
        if len(cleaned) == 1:
            return cleaned[0]
        return cleaned
    if isinstance(formatted, dict) and formatted:
        return _clean_widget(formatted)
    return []


def public_response_view(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}

    formatted = _clean_formatted_data(payload.get("formattedData"))
    metadata = payload.get("metadata")
    metadata = metadata if isinstance(metadata, dict) else {}
    session_id = metadata.get("sessionId") or payload.get("sessionId") or ""

    return {
        "status": bool(payload.get("status", True)),
        "message": _normalize_text(payload.get("message")),
        "formattedData": formatted,
        "summary": _normalize_text(payload.get("summary")),
        "analysis": _normalize_text(payload.get("analysis")),
        "prediction": _normalize_text(payload.get("prediction")),
        "conclusion": _normalize_text(payload.get("conclusion")),
        "suggestedQuestions": list(payload.get("suggestedQuestions") or []),
        "dotnetPayload": payload.get("dotnetPayload"),
        "sessionId": session_id,
    }
