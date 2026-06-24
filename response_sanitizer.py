"""Response sanitization helpers for public API outputs."""

from __future__ import annotations

from typing import Any, Dict

_ALLOWED_ROOT_KEYS = (
    "status",
    "sessionId",
    "message",
    "widget",
    "formattedData",
    "suggestedQuestions",
    "dotnetPayload",
    "metadata",
)

_ALLOWED_FORMATTED_KEYS = ("type", "title", "data", "analysis", "prediction", "conclusion")


def _clean_formatted_data(formatted: Any) -> Dict[str, Any]:
    if not isinstance(formatted, dict):
        formatted = {}
    return {key: formatted.get(key) for key in _ALLOWED_FORMATTED_KEYS}


def public_response_view(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        payload = {}

    formatted = _clean_formatted_data(payload.get("formattedData"))
    metadata = payload.get("metadata") if isinstance(payload.get("metadata"), dict) else {}

    return {
        "status": bool(payload.get("status", True)),
        "sessionId": payload.get("sessionId") or metadata.get("sessionId") or "admin-default",
        "message": (payload.get("message") or "").strip(),
        "widget": payload.get("widget") or formatted.get("type") or "TABLE",
        "formattedData": formatted,
        "suggestedQuestions": list(payload.get("suggestedQuestions") or []),
        "dotnetPayload": payload.get("dotnetPayload") or {},
        "metadata": metadata,
    }
