"""Response sanitization helpers for public API outputs."""

from __future__ import annotations

from typing import Any, Dict

_ALLOWED_ROOT_KEYS = (
    "status",
    "message",
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
