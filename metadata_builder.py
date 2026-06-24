"""Build response metadata outside of the response assembly layer."""

from __future__ import annotations

from typing import Any, Dict, Optional


def build_metadata(
    *,
    session_id: str,
    confidence: float,
    query_type: str,
    operation_count: int,
    durations: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    durations = durations or {}

    def _pick(*keys: str) -> float:
        for key in keys:
            value = durations.get(key)
            if value is not None:
                try:
                    return float(value)
                except (TypeError, ValueError):
                    continue
        return 0.0

    planner_ms = _pick("plannerMs", "plannerDurationMs", "planningMs", "planning_ms")
    intent_ms = _pick("intentMs", "intentDurationMs", "intent_duration")
    dotnet_ms = _pick("dotnetMs", "dotnetDurationMs", "dotnet_duration")
    combiner_ms = _pick("combinerMs", "combineDurationMs", "combiner_duration")
    report_ms = _pick("reportMs", "reportDurationMs", "report_duration", "analysisDurationMs")
    total_ms = _pick("totalMs", "totalDurationMs", "executionTimeMs", "total_duration")
    execution_ms = _pick("executionTimeMs", "totalMs", "totalDurationMs", "total_duration")

    return {
        "sessionId": session_id,
        "confidence": round(float(confidence), 2),
        "queryType": query_type,
        "operationCount": int(operation_count),
        "timings": {
            "plannerMs": round(planner_ms, 2),
            "intentMs": round(intent_ms, 2),
            "dotnetMs": round(dotnet_ms, 2),
            "combinerMs": round(combiner_ms, 2),
            "reportMs": round(report_ms, 2),
            "totalMs": round(total_ms, 2),
        },
        "executionTimeMs": round(execution_ms or total_ms, 2),
        "planner_duration": round(planner_ms, 2),
        "intent_duration": round(intent_ms, 2),
        "dotnet_duration": round(dotnet_ms, 2),
        "combiner_duration": round(combiner_ms, 2),
        "report_duration": round(report_ms, 2),
        "total_duration": round(total_ms, 2),
    }
