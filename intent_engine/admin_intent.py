"""
admin_intent.py
===============

Coordinator for the intent engine.

Responsibility: orchestrate the pipeline stages and assemble the final intent
dict.  Business intent decisions (category, operation, responseType) are made
ONLY in intent_classifier.py and are never re-derived here.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

from query_normalizer import clean_query as _normalise
from query_understanding_engine import understand_query

from .entity_extractor import assert_canonical_entity_keys, extract_entities
from .intent_classifier import classify_intent
from .intent_schema import (
    CATEGORY_OPERATION_TO_SUBCATEGORY,
    INTENT_TYPE_DEFAULTS,
    OPERATIONS_BY_CATEGORY,
    ISSUED_EQUIPMENT_ITEMS,
    PROCURED_EQUIPMENT_ITEMS,
    get_allowed_entities_for_category,
    SUBCATEGORY_TO_OPERATION,
)
from .payload_builder import build_ai_command_request_dto
from .payload_validator import PayloadValidationError, validate_payload

logger = logging.getLogger(__name__)


def _item_category(item_name: Optional[str]) -> Optional[str]:
    """Return 'IssuedItems' or 'ProcuredItems' for a known equipment item name."""
    if not item_name:
        return None
    if not isinstance(item_name, str):
        return None
    item_name = item_name.strip()
    if item_name in ISSUED_EQUIPMENT_ITEMS:
        return "IssuedItems"
    if item_name in PROCURED_EQUIPMENT_ITEMS:
        return "ProcuredItems"
    return None


def _subcategory_from_table(
    category: Optional[str], operation: Optional[str]
) -> Optional[str]:
    """Pure lookup — no inference.  Derives subcategory from the official table."""
    if category and operation:
        return CATEGORY_OPERATION_TO_SUBCATEGORY.get((category, operation))
    return None


def _legacy_type(
    category: Optional[str], operation: Optional[str], subcategory: Optional[str]
) -> Optional[str]:
    """Pure lookup — no inference.  Returns the deprecated visualization hint."""
    if not category or not subcategory:
        return None
    op_key = operation or SUBCATEGORY_TO_OPERATION.get(subcategory, subcategory)
    return INTENT_TYPE_DEFAULTS.get((category, op_key))


def _comparison_fallback_operation(category: Optional[str]) -> str:
    """Choose a category-safe fallback for planner compare fallthroughs."""
    fallback_by_category = {
        # Category-specific overview/list style defaults that already exist in schema.
        "Performance": "Summary",
        "Leave": "Current",
        "Medical": "Individual",
        "Attendance": "Summary",
        "Verification": "Pending",
        "Equipment": "Stats",
        "Distribution": "Latest",
        "Skills": "BySport",
        "Overall": "OverallPerformance",
        "Schedule": "Date",
        "personaldetail": "info",
        "disqualified": "removed",
    }
    if category in fallback_by_category:
        return fallback_by_category[category]
    if category and "Summary" in OPERATIONS_BY_CATEGORY.get(category, frozenset()):
        return "Summary"
    return next(iter(OPERATIONS_BY_CATEGORY.get(category, ("Summary",))), "Summary")


def _filter_entities_for_category(
    category: Optional[str], entities: Dict[str, Any]
) -> Dict[str, Any]:
    if not category:
        return dict(entities)
    allowed = get_allowed_entities_for_category(category)
    return {
        key: value
        for key, value in entities.items()
        if key in allowed or value is None
    }


def _build_base_intent(
    raw_query: str, resolved_entities: Dict[str, Any]
) -> Dict[str, Any]:
    """Return the full intent dict with all fields set to None / safe defaults."""
    return {
        "category": None,
        "subcategory": None,
        "operation": None,
        "number": None,
        "section": None,
        "sub_section": None,
        "metric": None,
        "sort_by": None,
        "group_by": None,
        "grading": None,
        "leave_type": None,
        "sport": None,
        "class": None,
        "unit_name": None,
        "attempt_no": None,
        "from_attempt": None,
        "to_attempt": None,
        "date": None,
        "item_name": None,
        "item_category": None,
        "company_id": None,
        "platoon_id": None,
        "batch_id": None,
        "from_date": None,
        "to_date": None,
        "agniveer_no": None,
        "bmi_category": None,
        "blood_group": None,
        "type": None,
        "medical_status": None,
        "diagnose": None,
        "responseType": None,
        "raw_query": raw_query,
        "confidence": "low",
        "confidence_score": 0.0,
        "query_type": "simple",
        "filters": {},
    }


def classify_admin_intent(
    query: str,
    resolved_entities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Coordinator — assembles the final intent dict.

    Decision ownership:
      - Category, Operation, ResponseType → intent_classifier.py (classify_intent)
      - Entities                          → entity_extractor.py  (extract_entities)
      - Semantic understanding            → query_understanding_engine.py (understand_query)
      - Subcategory                       → CATEGORY_OPERATION_TO_SUBCATEGORY table (pure lookup)

    No re-inference is performed here.  Each field is set exactly once.
    """
    raw_query = str(query or "").strip()
    resolved_entities = resolved_entities or {}

    # Guard: LLM disclaimer text — not a real query
    lowered_query = raw_query.lower()
    if (
        "may make mistakes" in lowered_query
        and "verify important information" in lowered_query
    ):
        return _build_base_intent(raw_query, resolved_entities)

    # ── Comparison short-circuit ─────────────────────────────────────────────
    # "BPET vs PPT", "BPET versus PPT scores", "compare BPET and PPT"
    _VS_PATTERNS = (" vs ", " versus ", "compare ")
    if any(pat in lowered_query for pat in _VS_PATTERNS):
        entities = extract_entities(raw_query, resolved_entities)
        semantic = understand_query(raw_query)
        intent_result = classify_intent(raw_query, entities, semantic)
        category = intent_result.get("category") or "Performance"
        base: Dict[str, Any] = _build_base_intent(raw_query, resolved_entities)
        base.update(
            {
                "category": category,
                "subcategory": "Comparison",
                "operation": "Compare",
                "query_type": "comparison",
                "confidence": intent_result.get("confidence", "medium"),
                "confidence_score": intent_result.get("confidence_score", 0.7),
                "number": entities.get("n"),
                "section": entities.get("section"),
                "sub_section": entities.get("subSection"),
                "grading": entities.get("grading"),
                "leave_type": entities.get("leaveType"),
                "sport": entities.get("sport"),
                "class": entities.get("class"),
                "unit_name": entities.get("unitName"),
                "attempt_no": entities.get("attemptNo"),
                "from_attempt": entities.get("fromAttempt"),
                "to_attempt": entities.get("toAttempt"),
                "date": entities.get("date"),
                "item_name": entities.get("equipmentName"),
                "item_category": _item_category(entities.get("equipmentName")),
                "company_id": entities.get("companyId"),
                "platoon_id": entities.get("platoonId"),
                "batch_id": entities.get("batchId"),
                "from_date": entities.get("fromDate"),
                "to_date": entities.get("toDate"),
                "agniveer_no": entities.get("agniveerNo"),
                "bmi_category": entities.get("bmiCategory"),
                "blood_group": entities.get("bloodGroup"),
                "medical_status": entities.get("medicalStatus"),
                "diagnose": entities.get("diagnose"),
                "days": entities.get("days"),
            }
        )
        base["filters"] = {
            key: value
            for key, value in (
                ("section", base["section"]),
                ("subSection", base["sub_section"]),
                ("grading", base["grading"]),
                ("leaveType", base["leave_type"]),
                ("sport", base["sport"]),
                ("class", base["class"]),
                ("unitName", base["unit_name"]),
                ("attemptNo", base["attempt_no"]),
                ("fromAttempt", base["from_attempt"]),
                ("toAttempt", base["to_attempt"]),
                ("date", base["date"]),
                ("companyId", base["company_id"]),
                ("platoonId", base["platoon_id"]),
                ("batchId", base["batch_id"]),
                ("agniveerNo", base["agniveer_no"]),
                ("bmiCategory", base["bmi_category"]),
                ("bloodGroup", base["blood_group"]),
                ("equipmentName", base["item_name"]),
                ("medicalStatus", base["medical_status"]),
                ("diagnose", base["diagnose"]),
                ("days", base["days"]),
            )
            if value is not None
        }
        return base

    # ── Stage 1: Extract entities ────────────────────────────────────────────
    entities = extract_entities(raw_query, resolved_entities)
    # FIX 4: Assert canonical entity keys (camelCase)
    assert_canonical_entity_keys(entities)

    # ── Stage 2: Semantic understanding ─────────────────────────────────────
    semantic = understand_query(raw_query)

    # ── Stage 3: Classify intent (single source of truth for category/op) ───
    intent_result = classify_intent(raw_query, entities, semantic)

    category = intent_result.get("category")
    operation: Optional[str] = intent_result.get("operation")

    # ── Stage 4: Subcategory — pure table lookup, no inference ───────────────
    subcategory: Optional[str] = _subcategory_from_table(category, operation)

    # Equipment item-list override: when the user names a specific item and an
    # explicitly-contextual operation keyword is present (overdue, poor condition,
    # holding, stats, etc.), the item's list membership (IssuedItems / ProcuredItems)
    # is authoritative. Otherwise, preserve classifier inference.
    if category == "Equipment" and entities.get("equipmentName"):
        item_cat = _item_category(entities.get("equipmentName"))
        if item_cat:
            _nq = _normalise(raw_query)
            if "issued" in _nq or "issue" in _nq:
                subcategory = "IssuedItems"
                operation = "Issued"
            elif "procured" in _nq or "procure" in _nq:
                subcategory = "ProcuredItems"
                operation = "Procured"
            elif any(
                kw in _nq
                for kw in {
                    "overdue",
                    "poor condition",
                    "returned",
                    "holding",
                    "stats",
                    "summary",
                    "agniveer wise",
                }
            ):
                # An explicit operational keyword is present. Keep the classifier/table lookup.
                pass
            else:
                # Fallback: no explicit keywords, use item category.
                subcategory = item_cat
                operation = SUBCATEGORY_TO_OPERATION.get(subcategory, operation)

    # ── Stage 5: Legacy visualization hint — pure lookup ────────────────────
    legacy_type: Optional[str] = _legacy_type(category, operation, subcategory)
    normalized_query = _normalise(raw_query)
    if "tabular" in normalized_query or "table" in normalized_query:
        legacy_type = "Tabular"

    # ── Stage 6: Confidence — derived from classifier output ─────────────────
    confidence_score = intent_result.get("confidence_score", 0.0)
    confidence = intent_result.get("confidence", "low")

    # ── Stage 7: Assemble result ─────────────────────────────────────────────
    result: Dict[str, Any] = {
        "category": category,
        "subcategory": subcategory,
        "operation": operation,
        "number": entities.get("n"),
        "section": entities.get("section"),
        "sub_section": entities.get("subSection"),
        "metric": None,
        "sort_by": None,
        "group_by": None,
        "grading": entities.get("grading"),
        "leave_type": entities.get("leaveType"),
        "sport": entities.get("sport"),
        "class": entities.get("class"),
        "unit_name": entities.get("unitName"),
        "attempt_no": entities.get("attemptNo"),
        "from_attempt": entities.get("fromAttempt"),
        "to_attempt": entities.get("toAttempt"),
        "date": entities.get("date"),
        "item_name": entities.get("equipmentName"),
        "item_category": _item_category(entities.get("equipmentName")),
        "company_id": entities.get("companyId"),
        "platoon_id": entities.get("platoonId"),
        "batch_id": entities.get("batchId"),
        "from_date": entities.get("fromDate"),
        "to_date": entities.get("toDate"),
        "agniveer_no": entities.get("agniveerNo"),
        "bmi_category": entities.get("bmiCategory"),
        "blood_group": entities.get("bloodGroup"),
        "type": legacy_type,
        "medical_status": entities.get("medicalStatus"),
        "diagnose": entities.get("diagnose"),
        "days": entities.get("days"),
        "responseType": intent_result.get("responseType"),
        "raw_query": raw_query,
        "confidence_score": confidence_score,
        "confidence": confidence,
        "query_type": "simple",
    }

    # Equipment subcategory → operation consistency (IssuedItems ↔ Issued)
    if result["category"] == "Equipment" and result["subcategory"] in {
        "IssuedItems",
        "ProcuredItems",
    }:
        result["operation"] = SUBCATEGORY_TO_OPERATION.get(
            result["subcategory"], result["operation"]
        )

    result["filters"] = {
        key: value
        for key, value in (
            ("section", result["section"]),
            ("subSection", result["sub_section"]),
            ("grading", result["grading"]),
            ("leaveType", result["leave_type"]),
            ("sport", result["sport"]),
            ("class", result["class"]),
            ("unitName", result["unit_name"]),
            ("attemptNo", result["attempt_no"]),
            ("fromAttempt", result["from_attempt"]),
            ("toAttempt", result["to_attempt"]),
            ("date", result["date"]),
            ("companyId", result["company_id"]),
            ("platoonId", result["platoon_id"]),
            ("batchId", result["batch_id"]),
            ("agniveerNo", result["agniveer_no"]),
            ("bmiCategory", result["bmi_category"]),
            ("bloodGroup", result["blood_group"]),
            ("equipmentName", result["item_name"]),
            ("medicalStatus", result["medical_status"]),
            ("diagnose", result["diagnose"]),
        )
        if value is not None
    }

    return result


_INVALID_DOTNET_OPERATIONS: frozenset = frozenset({"Compare"})


def format_admin_payload(intent_result: Dict[str, Any]) -> Dict[str, Any]:
    category = intent_result.get("category")
    operation = intent_result.get("operation")
    subcategory = intent_result.get("subcategory")
    response_type = intent_result.get("responseType")

    # "Compare" is an application-layer aggregation, never a .NET backend operation.
    # The planner decomposes comparison queries into independent retrieval operations
    # before they reach this layer. If "Compare" arrives here, it means the planner
    # fell through (could not extract >= 2 components); remap to "Summary" so the
    # .NET call is valid rather than returning HTTP 400.
    if operation in _INVALID_DOTNET_OPERATIONS:
        logger.warning(
            "format_admin_payload: blocked operation=%r from reaching .NET "
            "(category=%r) — remapping to 'Summary'. "
            "Indicates planner fallthrough on a comparison query.",
            operation,
            category,
        )
        operation = _comparison_fallback_operation(category)
        intent_result = {**intent_result, "operation": operation}

    entities: Dict[str, Any] = {
        "n": intent_result.get("number"),
        "section": intent_result.get("section"),
        "subSection": intent_result.get("sub_section"),
        "grading": intent_result.get("grading"),
        "leaveType": intent_result.get("leave_type"),
        "bmiCategory": intent_result.get("bmi_category"),
        "bloodGroup": intent_result.get("blood_group"),
        "equipmentName": intent_result.get("item_name"),
        "sport": intent_result.get("sport"),
        "class": intent_result.get("class"),
        "unitName": intent_result.get("unit_name"),
        "attemptNo": intent_result.get("attempt_no"),
        "fromAttempt": intent_result.get("from_attempt"),
        "toAttempt": intent_result.get("to_attempt"),
        "date": intent_result.get("date"),
        "fromDate": intent_result.get("from_date"),
        "toDate": intent_result.get("to_date"),
        "companyId": intent_result.get("company_id"),
        "platoonId": intent_result.get("platoon_id"),
        "batchId": intent_result.get("batch_id"),
        "agniveerNo": intent_result.get("agniveer_no"),
        "medicalStatus": intent_result.get("medical_status"),
        "diagnose": intent_result.get("diagnose"),
        "days": intent_result.get("days"),
    }

    # Assumption: after category is finalized, only category-safe entities should
    # reach payload validation and .NET DTO construction.
    entities = _filter_entities_for_category(category, entities)

    is_valid, errors = validate_payload(category, operation, entities)
    if not is_valid:
        raise PayloadValidationError(errors)

    payload = build_ai_command_request_dto(category, operation, response_type, entities)
    if category == "Leave" and intent_result.get("leave_type") is not None:
        payload["leaveType"] = intent_result["leave_type"]

    if subcategory and not operation:
        payload["operation"] = SUBCATEGORY_TO_OPERATION.get(
            subcategory, payload.get("operation")
        )

    return payload


def format_admin_intent(intent_result: Dict[str, Any]) -> Dict[str, Any]:
    payload = format_admin_payload(intent_result)
    payload["type"] = intent_result.get("type") or "Tabular"
    return payload
