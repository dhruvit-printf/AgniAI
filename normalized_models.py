"""
normalized_models.py
====================
Canonical normalization helpers for the admin pipeline.

This module keeps the shared JSON/dict-shape normalization logic in one place
so the rest of the pipeline can stay thin and predictable.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from utils import extract_record_id as _extract_record_id
from utils import extract_records as _extract_records
from utils import normalize_confidence as _normalize_confidence


def extract_records(data: Any) -> List[Dict]:
    return _extract_records(data)


def extract_record_id(record: Dict[str, Any]) -> Optional[str]:
    return _extract_record_id(record)


def normalize_dotnet_response(data: Any) -> Dict[str, Any]:
    """
    Attach a canonical `records` list to any backend response shape.
    """
    records = extract_records(data)
    if isinstance(data, dict):
        normalized = dict(data)
        normalized["records"] = records
        return normalized
    return {"records": records}


def calculate_section_confidence(
    section: Dict[str, Any],
    intent: Dict[str, Any],
    api_success: bool = True,
) -> float:
    if not api_success:
        return 0.0

    base_conf = _normalize_confidence(intent.get("confidence", 0.95), fallback=0.85)

    records = section.get("data") or []
    rec_count = len(records)
    record_factor = 0.0
    if rec_count == 0:
        record_factor = -0.05
    elif rec_count > 0:
        missing_fields = 0
        total_fields = 0
        for record in records[:5]:
            if isinstance(record, dict):
                for _, value in record.items():
                    total_fields += 1
                    if value is None or value == "":
                        missing_fields += 1
        if total_fields > 0:
            completeness = (total_fields - missing_fields) / total_fields
            record_factor = 0.05 * completeness
        else:
            record_factor = 0.05

    final_conf = base_conf + record_factor
    if section.get("failedFilters") or section.get("degraded"):
        final_conf -= 0.15

    return round(max(0.0, min(1.0, final_conf)), 2)


def normalize_intent_confidence(
    intent: Dict[str, Any],
    fallback_confidence: float,
) -> Dict[str, Any]:
    normalized = dict(intent or {})
    normalized["confidence"] = _normalize_confidence(
        normalized.get("confidence"), fallback=fallback_confidence
    )
    return normalized


def normalize_prediction(
    prediction: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    if not prediction:
        return {
            "trend": "Stable",
            "projection": "A reliable prediction is not available yet.",
            "heuristicEstimate": "A reliable prediction is not available yet.",
            "shortTerm": "stable",
            "futureTrends": ["A reliable prediction is not available yet."],
        }

    future_trends = prediction.get("futureTrends") or []
    trend_text = future_trends[0] if future_trends else ""
    projection = (
        prediction.get("projection") or prediction.get("forecast") or trend_text
    )
    if not projection:
        projection = "A reliable prediction is not available yet."

    heuristic_estimate = prediction.get("heuristicEstimate") or projection
    trend = prediction.get("trend") or "Stable"

    return {
        "trend": trend,
        "projection": projection,
        "heuristicEstimate": heuristic_estimate,
        "shortTerm": prediction.get("shortTerm") or str(trend).lower(),
        "futureTrends": list(prediction.get("futureTrends") or [heuristic_estimate]),
    }


def build_intro_message(
    title: str, intro_message: str, category: str
) -> Dict[str, Any]:
    return {
        "title": title,
        "description": intro_message
        or f"I have prepared a summary of the matching {category.lower()} records.",
    }


def _infer_simple_section_label(combined_result: Any, intent: Dict[str, Any]) -> str:
    """
    Prefer a real domain label over the generic "Result" bucket for simple
    record queries.
    """
    category = (intent.get("category") or "").strip()
    subcategory = (intent.get("subcategory") or "").strip()

    for key in ("section", "sub_section"):
        value = intent.get(key)
        if value:
            return str(value).strip()

    filters = intent.get("filters") or {}
    if isinstance(filters, dict):
        for key in (
            "section",
            "sub_section",
            "platoon",
            "platoonName",
            "unit",
            "unitName",
            "class",
            "sport",
        ):
            value = filters.get(key)
            if value:
                return str(value).strip()

    records = extract_records(combined_result)
    if records:
        sample = records[0]
        for key in (
            "sectionFilter",
            "sectionName",
            "section",
            "unitName",
            "class",
            "sport",
            "group",
        ):
            value = sample.get(key)
            if value:
                return str(value).strip()

        if category == "Performance" and subcategory in {
            "TopPerformers",
            "LowestPerformers",
        }:
            # Top/bottom performer queries often return an overall ranking across
            # the whole dataset. Do not infer a platoon label from the first row,
            # because that can make an overall ranking look like it belongs to one
            # platoon only.
            platoons = {
                str(record.get("platoonName")).strip()
                for record in records
                if isinstance(record, dict) and record.get("platoonName")
            }
            if len(platoons) == 1:
                return next(iter(platoons))
            return "Overall"

    return "Result"


def _derive_title(query_type: str, intent: Dict[str, Any]) -> str:
    category = intent.get("category") or "Agniveer"
    subcategory = intent.get("subcategory") or ""

    if query_type == "compare":
        return f"{category} Side-by-Side Comparison"
    if query_type == "cross_filter":
        return f"{category} Cross-Filter Analysis"
    if query_type == "multi_independent":
        return "Consolidated Module Statistics"
    if query_type == "trend":
        return f"{category} Timeline Trend Analysis"
    if query_type == "distribution":
        return f"{category} Distribution Breakdown"
    if subcategory:
        import re

        spaced = re.sub(r"([A-Z])", r" \1", subcategory).strip()
        return f"{category} {spaced}"
    return f"{category} Overview"


def assemble_response_metadata(
    *,
    confidence: float,
    query_type: str,
    operation_count: int,
    durations: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    from telemetry import request_id_var, session_id_var, trace_id_var

    metadata: Dict[str, Any] = {
        "requestId": request_id_var.get() or "N/A",
        "traceId": trace_id_var.get() or "N/A",
        "sessionId": session_id_var.get() or "N/A",
        "executionTimeMs": 0,
        "intentDurationMs": 0,
        "dotnetDurationMs": 0,
        "combineDurationMs": 0,
        "analysisDurationMs": 0,
        "predictionDurationMs": 0,
        "conclusionDurationMs": 0,
        "totalDurationMs": 0,
        "confidence": round(float(confidence), 2),
        "queryType": query_type,
        "operationCount": int(operation_count),
    }
    if durations:
        metadata.update(durations)
        metadata.setdefault(
            "entityResolutionMs", durations.get("entityResolutionMs", 0)
        )
        metadata.setdefault("planningMs", durations.get("planningMs", 0))
        metadata.setdefault("widgetMs", durations.get("widgetMs", 0))
        metadata.setdefault(
            "responseAssemblyMs", durations.get("responseAssemblyMs", 0)
        )
    return metadata
