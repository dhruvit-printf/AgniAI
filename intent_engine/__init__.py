"""Intent engine package."""

from __future__ import annotations

from importlib import import_module
from typing import Any

__all__ = [
    "classify_admin_intent",
    "format_admin_intent",
    "format_admin_payload",
    "extract_entities",
    "classify_intent",
    "build_ai_command_request_dto",
    "validate_payload",
    "validate_payload_strict",
    "QueryPlan",
    "QueryType",
    "SubOperation",
    "plan_query",
]


def __getattr__(name: str) -> Any:
    if name in {"classify_admin_intent", "format_admin_intent", "format_admin_payload"}:
        module = import_module(".admin_intent", __name__)
    elif name == "extract_entities":
        module = import_module(".entity_extractor", __name__)
    elif name == "classify_intent":
        module = import_module(".intent_classifier", __name__)
    elif name == "build_ai_command_request_dto":
        module = import_module(".payload_builder", __name__)
    elif name in {"validate_payload", "validate_payload_strict"}:
        module = import_module(".payload_validator", __name__)
    elif name in {"QueryPlan", "QueryType", "SubOperation", "plan_query"}:
        module = import_module(".query_planner", __name__)
    else:
        raise AttributeError(name)
    return getattr(module, name)
