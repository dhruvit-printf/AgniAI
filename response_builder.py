"""
response_builder.py
====================
Assembles the final JSON response payload for the admin pipeline.

Security: dotnetResponse.rawResponse is NOT included in the payload
returned to the frontend. Raw backend data must never leave the server.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def build_combined_message(
    intro_message: str,
    formatted_data: str,
    analysis: Optional[Dict[str, Any]],
    conclusion: Optional[Dict[str, Any]],
) -> str:
    """
    Merge introMessage + formatted_data + analysis + conclusion into one
    string that the frontend reads for the text bubble.
    """
    parts = []

    intro = (intro_message or "").strip()
    if intro:
        parts.append(intro)

    data_text = (formatted_data or "").strip()
    if data_text:
        parts.append(data_text)

    analysis_parts = []
    if analysis:
        summary = (analysis.get("summary") or "").strip()
        if summary:
            analysis_parts.append(summary)

        obs = analysis.get("observations") or []
        clean_obs = [o.strip() for o in obs if o and o.strip()]
        if clean_obs:
            analysis_parts.append(
                "Observations:\n" + "\n".join(f"- {o}" for o in clean_obs)
            )

        insights = analysis.get("insights") or []
        clean_ins = [i.strip() for i in insights if i and i.strip()]
        if clean_ins:
            analysis_parts.append(
                "Insights:\n" + "\n".join(f"- {i}" for i in clean_ins)
            )

        if analysis_parts:
            parts.append("Analysis:\n" + "\n\n".join(analysis_parts))

    if conclusion:
        conclusion_summary = (conclusion.get("summary") or "").strip()
        if conclusion_summary:
            parts.append(f"Conclusion:\n{conclusion_summary}")

    return "\n\n".join(parts)


def build_response(
    query_type: str,
    intro_message: str,
    combined_result: Any,
    analysis: Optional[Dict[str, Any]],
    conclusion: Optional[Dict[str, Any]],
    intent: Dict[str, Any],
    raw_results: List[Any],
    confidence: float,
    operation_count: int,
    formatted_data: str,
    session_id: Optional[str] = None,
    durations: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Assembles the JSON response structure for the frontend.

    Security guarantee: raw .NET API responses are NEVER included in the
    returned payload. The 'dotnetResponse' key is omitted entirely to
    prevent internal data leakage.
    """
    # ── Normalize intent confidence to float ─────────────────────────────────
    intent_conf = intent.get("confidence")
    if isinstance(intent_conf, str):
        mapping = {"high": 0.95, "medium": 0.70, "low": 0.30}
        intent_conf_float = mapping.get(intent_conf.lower(), 0.5)
    elif isinstance(intent_conf, (int, float)):
        intent_conf_float = float(intent_conf)
    else:
        intent_conf_float = 0.95

    # ── Build the message bubble string ──────────────────────────────────────
    combined_message = build_combined_message(
        intro_message, formatted_data, analysis, conclusion
    )

    # ── Standard JSON schema — NO raw backend data ────────────────────────────
    payload: Dict[str, Any] = {
        "status": True,
        "queryType": query_type,
        "introMessage": intro_message,
        "result": {
            "processedData": combined_result if combined_result is not None else {}
        },
        "analysis": (
            {
                "summary": analysis.get("summary", ""),
                "observations": list(analysis.get("observations") or []),
                "insights": list(analysis.get("insights") or []),
            }
            if analysis is not None
            else None
        ),
        "conclusion": (
            {"summary": conclusion.get("summary", "")}
            if conclusion is not None
            else None
        ),
        "intent": {
            "category": intent.get("category", ""),
            "subcategory": intent.get("subcategory", ""),
            "confidence": intent_conf_float,
        },
        # dotnetResponse is intentionally omitted — raw backend data must
        # never be forwarded to the frontend (security requirement).
        "metadata": {
            "confidence": round(float(confidence), 2),
            "queryType": query_type,
            "operationCount": int(operation_count),
        },
        # Backward-compatible fields
        "formattedData": formatted_data,
        "message": combined_message,
    }

    if durations:
        payload["metadata"].update(durations)

    if session_id and session_id != "admin-default":
        payload["sessionId"] = session_id

    return payload
