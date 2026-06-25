"""
response_builder.py
===================
Final response assembly layer.

formattedData is ALWAYS a list of widget dicts.
analysis, prediction, conclusion are injected into every widget in the list
so each widget is fully self-contained.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_response(
    message: str,
    formatted_data: Any,
    metadata: Optional[Dict[str, Any]],
    session_id: str,
    analysis: Optional[Dict[str, Any]] = None,
    prediction: Optional[Dict[str, Any]] = None,
    conclusion: Optional[Dict[str, Any]] = None,
    suggested_questions: Optional[List[str]] = None,
    dotnet_payload: Optional[Any] = None,
    overall_confidence: float = 0.0,
    partial_failure: bool = False,
    failed_sections: Optional[List[Any]] = None,
) -> Dict[str, Any]:
    # formattedData is always a list — backward-compat: wrap single dict
    if isinstance(formatted_data, list):
        fd: List[Any] = formatted_data
    elif isinstance(formatted_data, dict) and formatted_data:
        fd = [formatted_data]
    else:
        fd = []

    # Inject analysis/prediction/conclusion into every widget so each is
    # self-contained and renderable without additional context.
    extras: Dict[str, Any] = {}
    if analysis:
        extras["analysis"] = analysis
    if prediction:
        extras["prediction"] = prediction
    if conclusion:
        extras["conclusion"] = conclusion

    if extras:
        fd = [{**widget, **extras} if isinstance(widget, dict) else widget for widget in fd]

    meta = dict(metadata) if isinstance(metadata, dict) else {}
    if "sessionId" not in meta:
        meta["sessionId"] = session_id

    return {
        "status": True,
        "message": message or "",
        "formattedData": fd,
        "suggestedQuestions": suggested_questions or [],
        "dotnetPayload": dotnet_payload or {},
        "metadata": meta,
        "overallConfidence": round(float(overall_confidence or 0.0), 2),
        "partialFailure": bool(partial_failure),
        "failedSections": failed_sections or [],
    }
