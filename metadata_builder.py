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
    report_ms = _pick(
        "reportMs", "reportDurationMs", "report_duration", "analysisDurationMs"
    )
    total_ms = _pick("totalMs", "totalDurationMs", "executionTimeMs", "total_duration")
    execution_ms = _pick(
        "executionTimeMs", "totalMs", "totalDurationMs", "total_duration"
    )

    analysis_ms = _pick("analysisDurationMs")
    prediction_ms = _pick("predictionDurationMs")
    conclusion_ms = _pick("conclusionDurationMs")
    entity_resolution_ms = _pick("entityResolutionMs", "entity_resolution_ms")
    planning_ms = _pick("planningMs", "planning_ms")
    widget_ms = _pick("widgetMs", "widget_duration", "widget_ms")
    response_assembly_ms = _pick(
        "responseAssemblyMs", "response_assembly_duration", "response_assembly_ms"
    )

    planner_ms = round(planner_ms, 2)
    intent_ms = round(intent_ms, 2)
    dotnet_ms = round(dotnet_ms, 2)
    combiner_ms = round(combiner_ms, 2)
    report_ms = round(report_ms, 2)
    total_ms = round(total_ms, 2)
    execution_ms = round(execution_ms or total_ms, 2)
    analysis_ms = round(analysis_ms, 2)
    prediction_ms = round(prediction_ms, 2)
    conclusion_ms = round(conclusion_ms, 2)
    entity_resolution_ms = round(entity_resolution_ms, 2)
    planning_ms = round(planning_ms, 2)
    widget_ms = round(widget_ms, 2)
    response_assembly_ms = round(response_assembly_ms, 2)

    return {
        "sessionId": session_id,
        "confidence": round(float(confidence), 2),
        "queryType": query_type,
        "operationCount": int(operation_count),
        "timings": {
            "plannerMs": planner_ms,
            "intentMs": intent_ms,
            "dotnetMs": dotnet_ms,
            "combinerMs": combiner_ms,
            "reportMs": report_ms,
            "totalMs": total_ms,
        },
        "executionTimeMs": execution_ms,
        "plannerDurationMs": planner_ms,
        "intentDurationMs": intent_ms,
        "dotnetDurationMs": dotnet_ms,
        "combineDurationMs": combiner_ms,
        "totalDurationMs": total_ms,
        "analysisDurationMs": analysis_ms,
        "predictionDurationMs": prediction_ms,
        "conclusionDurationMs": conclusion_ms,
        "entityResolutionMs": entity_resolution_ms,
        "planningMs": planning_ms,
        "widgetMs": widget_ms,
        "responseAssemblyMs": response_assembly_ms,
        # Legacy snake_case for backward-compat
        "planner_duration": planner_ms,
        "intent_duration": intent_ms,
        "dotnet_duration": dotnet_ms,
        "combiner_duration": combiner_ms,
        "report_duration": report_ms,
        "total_duration": total_ms,
    }
