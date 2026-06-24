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
) -> Dict[str, Any]:
    formatted = formatted_data if isinstance(formatted_data, dict) else {}
    return {
        "status": True,
        "sessionId": session_id,
        "message": message or "",
        "formattedData": formatted,
        "suggestedQuestions": suggested_questions or [],
        "dotnetPayload": dotnet_payload or {},
        "metadata": metadata or {},
    }
