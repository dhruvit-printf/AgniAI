"""
visualization_engine.py
=======================
Compatibility shim for the older widget inference entry point.

The active widget engine now lives in widget_engine.py. This module keeps the
legacy function signature alive for any stray imports while delegating to the
single canonical implementation.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from widget_engine import generate_widgets as _generate_widgets

__all__ = ["collect_all_keys", "generate_widgets"]


def collect_all_keys(data: Any) -> set[str]:
    """Retain the old helper for compatibility with older tests and imports."""
    keys: set[str] = set()
    if isinstance(data, dict):
        for key, value in data.items():
            keys.add(key.lower())
            keys.update(collect_all_keys(value))
    elif isinstance(data, list):
        for item in data:
            keys.update(collect_all_keys(item))
    return keys


def _infer_query_type(query_plan: Any) -> Optional[str]:
    if isinstance(query_plan, str):
        return query_plan
    if isinstance(query_plan, dict):
        return query_plan.get("queryType") or query_plan.get("query_type")
    if hasattr(query_plan, "query_type") and query_plan.query_type is not None:
        return getattr(query_plan.query_type, "value", query_plan.query_type)
    return None


def _coerce_answer(combined_result: Any, query_plan: Any) -> Dict[str, Any]:
    if isinstance(combined_result, dict):
        return combined_result

    qtype = (_infer_query_type(query_plan) or "").lower()
    if qtype == "compare":
        return {
            "left": {},
            "right": {},
            "comparison": {},
            "sections": [],
        }
    return {"sections": [{"label": "Result", "data": combined_result or []}]}


def generate_widgets(
    combined_result: Any,
    query_plan: Any = None,
    analysis: Any = None,
) -> List[Dict[str, str]]:
    answer = _coerce_answer(combined_result, query_plan)
    qtype = (_infer_query_type(query_plan) or "simple").lower()
    intent = analysis if isinstance(analysis, dict) else {}
    return _generate_widgets(answer=answer, query_type=qtype, intent=intent)
