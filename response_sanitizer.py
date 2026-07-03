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


def _clean_widget(widget: Any) -> Dict[str, Any]:
    if not isinstance(widget, dict):
        return {}
    return {key: widget[key] for key in _ALLOWED_WIDGET_KEYS if key in widget}


def _clean_formatted_data(formatted: Any) -> Union[List[Dict[str, Any]], Dict[str, Any]]:
    """Preserve single-widget objects and lists of widgets."""
    if isinstance(formatted, list):
        cleaned = [_clean_widget(w) for w in formatted if isinstance(w, dict)]
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
    raw_meta = (
        payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}
    )
    session_id = raw_meta.get("sessionId") or payload.get("sessionId") or ""

    return {
        "status": bool(payload.get("status", True)),
        "message": (payload.get("message") or "").strip(),
        "formattedData": formatted,
        "summary": (payload.get("summary") or ""),
        "analysis": (payload.get("analysis") or ""),
        "prediction": (payload.get("prediction") or ""),
        "conclusion": (payload.get("conclusion") or ""),
        "suggestedQuestions": list(payload.get("suggestedQuestions") or []),
        "dotnetPayload": payload.get("dotnetPayload"),
        "sessionId": session_id,
    }
