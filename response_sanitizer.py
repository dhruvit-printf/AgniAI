"""Response sanitization helpers for public API outputs."""

from __future__ import annotations

from typing import Any, Dict, List

_ALLOWED_ROOT_KEYS = (
    "status",
    "message",
    "formattedData",
    "suggestedQuestions",
    "dotnetPayload",
    "metadata",
)

# Keys allowed per widget in the formattedData list.
# 'id' is required by the frontend for widget keying.
_ALLOWED_WIDGET_KEYS = (
    "id",
    "type",
    "title",
    "data",
    "analysis",
    "prediction",
    "conclusion",
)


def _clean_widget(widget: Any) -> Dict[str, Any]:
    if not isinstance(widget, dict):
        return {}
    return {key: widget.get(key) for key in _ALLOWED_WIDGET_KEYS}


def _clean_formatted_data(formatted: Any) -> List[Dict[str, Any]]:
    """
    Always returns a list of cleaned widget dicts.

    formattedData changed from a single dict to a list in the multi-widget
    refactor. This function handles both shapes for backward compatibility,
    but the canonical output is always a list.
    """
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

    clean_meta = {
        "sessionId": raw_meta.get("sessionId") or payload.get("sessionId") or "admin-default",
        "metrics": {
            "confidence": round(float(raw_meta.get("confidence") or 0.0), 2),
            "queryType": raw_meta.get("queryType") or "",
            "operationCount": int(raw_meta.get("operationCount") or 0),
        },
        "executionTimeMs": round(float(raw_meta.get("executionTimeMs") or 0.0), 2),
    }

    return {
        "status": bool(payload.get("status", True)),
        "message": (payload.get("message") or "").strip(),
        "formattedData": formatted,
        "suggestedQuestions": list(payload.get("suggestedQuestions") or []),
        "dotnetPayload": payload.get("dotnetPayload") or {},
        "metadata": clean_meta,
    }
