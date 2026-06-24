"""
response_builder.py
===================
Thin final response assembly layer.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_response(
    message: str,
    formatted_data: Optional[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]],
    session_id: str,
    suggested_questions: Optional[List[str]] = None,
    dotnet_payload: Optional[Any] = None,
    combined_result: Optional[Any] = None,
) -> Dict[str, Any]:
    formatted = formatted_data if isinstance(formatted_data, dict) else {}
    
    # Ensure strict contract subfields inside formattedData
    for key in ("type", "title"):
        if key not in formatted:
            formatted[key] = ""
    for key in ("data", "analysis", "prediction", "conclusion"):
        if key not in formatted or not isinstance(formatted[key], dict):
            formatted[key] = {}

    meta = dict(metadata) if isinstance(metadata, dict) else {}
    if "sessionId" not in meta:
        meta["sessionId"] = session_id

    return {
        "status": True,
        "message": message or "",
        "formattedData": formatted,
        "suggestedQuestions": suggested_questions or [],
        "dotnetPayload": dotnet_payload or {},
        "metadata": meta,
    }


