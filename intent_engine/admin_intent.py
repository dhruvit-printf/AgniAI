"""
admin_intent.py
===============

Coordinator for the intent engine.

Responsibility: orchestrate the pipeline stages and assemble the final intent
dict.  Business intent decisions (category, operation, responseType) are made
ONLY in intent_classifier.py and are never re-derived here.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .entity_extractor import extract_entities
from .intent_classifier import classify_intent
from .intent_schema import (
    CATEGORY_OPERATION_TO_SUBCATEGORY,
    INTENT_TYPE_DEFAULTS,
    ISSUED_EQUIPMENT_ITEMS,
    PROCURED_EQUIPMENT_ITEMS,
    SUBCATEGORY_TO_OPERATION,
)
from .payload_builder import build_ai_command_request_dto
from .payload_validator import validate_payload
from query_normalizer import clean_query as _normalise
from query_understanding_engine import understand_query


def _item_category(item_name: Optional[str]) -> Optional[str]:
    """Return 'IssuedItems' or 'ProcuredItems' for a known equipment item name."""
    if not item_name:
        return None
    lowered = item_name.lower()
    if any(item.lower() == lowered for item in ISSUED_EQUIPMENT_ITEMS):
        return "IssuedItems"
    if any(item.lower() == lowered for item in PROCURED_EQUIPMENT_ITEMS):
        return "ProcuredItems"
    return None


def _subcategory_from_table(category: Optional[str], operation: Optional[str]) -> Optional[str]:
    """Pure lookup — no inference.  Derives subcategory from the official table."""
    if category and operation:
        return CATEGORY_OPERATION_TO_SUBCATEGORY.get((category, operation))
    return None


def _legacy_type(category: Optional[str], operation: Optional[str], subcategory: Optional[str]) -> Optional[str]:
    """Pure lookup — no inference.  Returns the deprecated visualization hint."""
    if not category or not subcategory:
        return None
    op_key = operation or SUBCATEGORY_TO_OPERATION.get(subcategory, subcategory)
    return INTENT_TYPE_DEFAULTS.get((category, op_key))


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
    if "may make mistakes" in lowered_query and "verify important information" in lowered_query:
        return {
            "category": None, "subcategory": None, "operation": None,
            "number": None, "section": None, "sub_section": None,
            "metric": None, "sort_by": None, "group_by": None,
            "grading": None, "leave_type": None, "sport": None,
            "class": None, "unit_name": None, "attempt_no": None,
            "from_attempt": None, "to_attempt": None, "date": None,
            "item_name": None, "item_category": None,
            "company_id": None, "platoon_id": None, "batch_id": None,
            "from_date": None, "to_date": None, "agniveer_no": None,
            "bmi_category": None, "blood_group": None,
            "type": None, "medical_status": None,
            "responseType": "Summary", "raw_query": raw_query,
            "confidence": "low", "query_type": "simple", "filters": {},
        }

    # ── Stage 1: Extract entities ────────────────────────────────────────────
    entities = extract_entities(raw_query, resolved_entities)

    # ── Stage 2: Semantic understanding ─────────────────────────────────────
    semantic = understand_query(raw_query)

    # ── Stage 3: Classify intent (single source of truth for category/op) ───
    intent_result = classify_intent(raw_query, entities, semantic)

    category: Optional[str] = intent_result.get("category")
    operation: Optional[str] = intent_result.get("operation")

    # ── Stage 4: Subcategory — pure table lookup, no inference ───────────────
    subcategory: Optional[str] = _subcategory_from_table(category, operation)

    # Equipment item-list override: when the user names a specific item and no
    # explicitly-contextual operation keyword is present (overdue, poor condition,
    # holding, stats, etc.), the item's list membership (IssuedItems / ProcuredItems)
    # is authoritative over the classifier's entity-bonus tie-breaker.
    if category == "Equipment" and entities.get("equipmentName"):
        item_cat = _item_category(entities.get("equipmentName"))
        if item_cat:
            _EXPLICIT_EQUIP_OP_KEYWORDS = frozenset({
                "overdue", "poor condition", "returned", "holding",
                "stats", "summary", "agniveer wise",
            })
            _nq = _normalise(raw_query)
            if not any(kw in _nq for kw in _EXPLICIT_EQUIP_OP_KEYWORDS):
                subcategory = item_cat
                operation = SUBCATEGORY_TO_OPERATION.get(subcategory, operation)

    # Backfill: recover category/operation from subcategory when the classifier
    # had insufficient keyword evidence (extremely terse queries).
    if not category and subcategory:
        category = next(
            (cat for (cat, _op), sub in CATEGORY_OPERATION_TO_SUBCATEGORY.items() if sub == subcategory),
            None,
        )
    if not operation and subcategory:
        operation = SUBCATEGORY_TO_OPERATION.get(subcategory)

    # ── Stage 5: Legacy visualization hint — pure lookup ────────────────────
    legacy_type: Optional[str] = _legacy_type(category, operation, subcategory)
    normalized_query = _normalise(raw_query)
    if "tabular" in normalized_query or "table" in normalized_query:
        legacy_type = "Tabular"

    # ── Stage 6: Confidence — derived from classifier output ─────────────────
    confidence = "low"
    if category and subcategory:
        confidence = "high"
    elif category:
        confidence = "medium"
    if subcategory in {"TopPerformers", "LowestPerformers"} and entities.get("n") is None:
        confidence = "medium"

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
        "medical_status": entities.get("medical_status"),
        "responseType": intent_result.get("responseType", "Summary"),
        "raw_query": raw_query,
        "confidence": confidence,
        "query_type": "simple",
    }

    # Equipment subcategory → operation consistency (IssuedItems ↔ Issued)
    if result["category"] == "Equipment" and result["subcategory"] in {"IssuedItems", "ProcuredItems"}:
        result["operation"] = SUBCATEGORY_TO_OPERATION.get(result["subcategory"], result["operation"])

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
        )
        if value is not None
    }

    return result


def format_admin_payload(intent_result: Dict[str, Any]) -> Dict[str, Any]:
    category = intent_result.get("category")
    operation = intent_result.get("operation")
    subcategory = intent_result.get("subcategory")
    response_type = intent_result.get("responseType", "Summary")

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
    }

    validation_result = validate_payload(category, operation, entities)
    if not validation_result["valid"]:
        import logging

        logger = logging.getLogger(__name__)
        for error in validation_result["errors"]:
            logger.warning("Payload validation warning: %s", error)

    payload = build_ai_command_request_dto(category, operation, response_type, entities)
    if category == "Leave" and intent_result.get("leave_type") is not None:
        payload["leaveType"] = intent_result["leave_type"]

    if subcategory and not operation:
        payload["operation"] = SUBCATEGORY_TO_OPERATION.get(subcategory, payload.get("operation"))

    return payload


def format_admin_intent(intent_result: Dict[str, Any]) -> Dict[str, Any]:
    payload = format_admin_payload(intent_result)
    payload["type"] = intent_result.get("type") or "Tabular"
    return payload
