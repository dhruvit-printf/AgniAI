"""Response sanitization helpers for public API outputs.

Universal response contract (all fields mandatory, no extras):

{
    "status":             bool,
    "message":            str,
    "formattedData":      [ { "type": str, "title": str, "data": dict } ],
    "analysis":           str,
    "prediction":         str,
    "conclusion":         str,
    "suggestedQuestions": [ str ],
    "metadata": {
        "sessionId":       str,
        "confidence":      float,
        "queryType":       str,
        "operationCount":  int,
        "executionTimeMs": float
    }
}
"""

from __future__ import annotations

from typing import Any, Dict, List

# Only these three keys are exposed per widget.
# id, analysis, prediction, conclusion are internal — never sent to frontend.
_ALLOWED_WIDGET_KEYS = ("type", "title", "data")


def _clean_widget(widget: Any) -> Dict[str, Any]:
    if not isinstance(widget, dict):
        return {}
    return {key: widget[key] for key in _ALLOWED_WIDGET_KEYS if key in widget}


def _clean_formatted_data(formatted: Any) -> List[Dict[str, Any]]:
    """Always returns a list of cleaned widget dicts."""
    if isinstance(formatted, list):
        return [_clean_widget(w) for w in formatted if isinstance(w, dict)]
    if isinstance(formatted, dict) and formatted:
        # Legacy single-widget shape — wrap in list
        return [_clean_widget(formatted)]
    return []


def public_response_view(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}

    formatted = _clean_formatted_data(payload.get("formattedData"))
    raw_meta = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

    # Flat metadata — no nested "metrics" object
    clean_meta = {
        "sessionId":      raw_meta.get("sessionId") or payload.get("sessionId") or "",
        "confidence":     round(float(raw_meta.get("confidence") or 0.0), 2),
        "queryType":      raw_meta.get("queryType") or "",
        "operationCount": int(raw_meta.get("operationCount") or 0),
        "executionTimeMs": round(float(raw_meta.get("executionTimeMs") or 0.0), 2),
    }

    return {
        "status":             bool(payload.get("status", True)),
        "message":            (payload.get("message") or "").strip(),
        "formattedData":      formatted,
        "analysis":           (payload.get("analysis")   or ""),
        "prediction":         (payload.get("prediction") or ""),
        "conclusion":         (payload.get("conclusion") or ""),
        "suggestedQuestions": list(payload.get("suggestedQuestions") or []),
        "metadata":           clean_meta,
    }
