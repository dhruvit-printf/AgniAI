"""
admin_intent.py
===============

Backwards-compatible coordinator for the intent engine.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Tuple

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

ISSUED_ITEMS = tuple(ISSUED_EQUIPMENT_ITEMS)
PROCURED_ITEMS = tuple(PROCURED_EQUIPMENT_ITEMS)
_ITEM_LOOKUP: Dict[str, Tuple[str, str]] = {
    item.lower(): (item, "IssuedItems") for item in ISSUED_ITEMS
}
_ITEM_LOOKUP.update({item.lower(): (item, "ProcuredItems") for item in PROCURED_ITEMS})


def _item_category(item_name: Optional[str]) -> Optional[str]:
    if not item_name:
        return None
    lowered = item_name.lower()
    if any(item.lower() == lowered for item in ISSUED_EQUIPMENT_ITEMS):
        return "IssuedItems"
    if any(item.lower() == lowered for item in PROCURED_EQUIPMENT_ITEMS):
        return "ProcuredItems"
    return None


def _extract_item_query(query: str) -> Tuple[Optional[str], Optional[str]]:
    normalized = _normalise(query)
    for key, (item, category) in _ITEM_LOOKUP.items():
        if key in normalized:
            return item, category
    return None, None


def _infer_category(query: str, intent: Dict[str, Any], entities: Dict[str, Any]) -> Optional[str]:
    normalized = _normalise(query)
    if re.search(r"make mistakes|verify important information", normalized):
        return None
    if entities.get("equipmentName") or any(
        token in normalized
        for token in (
            "issued items",
            "procured items",
            "equipment summary",
            "equipment returned",
            "poor condition",
            "holding equipment",
            "overdue equipment",
        )
    ):
        return "Equipment"
    if entities.get("section"):
        return "Performance"
    if entities.get("bmiCategory") or entities.get("bloodGroup"):
        return "Medical"
    if entities.get("leaveType"):
        return "Leave"
    if entities.get("medical_status"):
        return "Medical"
    if entities.get("sport") or entities.get("class"):
        if "roster" in normalized:
            return "Roster"
        if "attendance" in normalized:
            return "Attendance"
        if "distribution" in normalized:
            return "Distribution"
        return "Skills"
    if "attendance" in normalized:
        return "Attendance"
    if "distribution" in normalized and "unit" in normalized:
        return "Distribution"
    if "schedule" in normalized:
        return "Schedule"
    if "strength" in normalized:
        return "Strength"
    if "overall" in normalized:
        return "Overall"
    return intent.get("category")


def _infer_operation(category: Optional[str], intent: Dict[str, Any], entities: Dict[str, Any], query: str) -> Optional[str]:
    normalized = _normalise(query)
    operation = intent.get("operation")
    if entities.get("grading"):
        operation = operation or "Grading"
    if entities.get("bmiCategory"):
        operation = "BMI"
    if entities.get("bloodGroup"):
        operation = "BloodGroup"
    if entities.get("attemptNo") is not None or entities.get("fromAttempt") is not None or entities.get("toAttempt") is not None:
        operation = "AttemptWise"
    if entities.get("leaveType") == "Current":
        operation = "Current"
    if entities.get("medical_status") == "Active":
        operation = "Active"

    if category == "Performance":
        if "pass percentage" in normalized or "pass rate" in normalized:
            return "PassPercentage"
        if "fail percentage" in normalized or "fail rate" in normalized:
            return "FailPercentage"
        if "grading summary" in normalized or "grade summary" in normalized:
            return "GradingSummary"
        if "grade distribution" in normalized or "grading distribution" in normalized:
            return "Grading"
        if any(token in normalized for token in ("compare", "comparison", " vs ", " versus ")):
            return "Compare"
        if any(token in normalized for token in ("top", "highest", "best")):
            return "Top"
        if any(token in normalized for token in ("bottom", "lowest", "worst")):
            return "Bottom"
        if any(token in normalized for token in ("improvement", "improve", "improved")):
            return "Improvement"
        if any(token in normalized for token in ("drop", "decline", "declined", "worse")):
            return "Drop"
        if entities.get("section") and not operation:
            operation = "Summary"

    if category == "Leave":
        if any(token in normalized for token in ("fewest", "least")):
            return "Least"
        if any(token in normalized for token in ("most", "maximum")):
            return "Most"
        if any(token in normalized for token in ("current", "today", "on leave", "absent today", "currently absent")):
            return "Current"
        if any(token in normalized for token in ("abscond", "missing", "awol")):
            return "Absconded"

    if category == "Attendance":
        month_match = re.search(
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b",
            query,
            re.IGNORECASE,
        )
        if any(token in normalized for token in ("present", "on campus", "who is present")):
            return "Present"
        if any(token in normalized for token in ("weekly", "week")):
            return "Weekly"
        if any(token in normalized for token in ("yearly", "annual")):
            return "Yearly"
        if month_match or entities.get("date"):
            return "Monthly"
        if any(token in normalized for token in ("daily", "today", "day")):
            return "Daily"

    if category == "Distribution":
        if any(token in normalized for token in ("top unit", "highest", "most")) and "unit" in normalized:
            return "TopUnit"
        if "latest" in normalized or "recent" in normalized:
            return "Latest"
        if "unassigned" in normalized:
            return "Unassigned"
        if "unit" in normalized:
            return "ByUnit"

    if category == "Verification":
        if "pending" in normalized:
            return "Pending"
        if "sent" in normalized:
            return "Sent"
        if "not responded" in normalized or "no response" in normalized:
            return "NotResponded"
        if "rejected" in normalized:
            return "Rejected"
        if "verified" in normalized or "completed" in normalized or "done" in normalized:
            return "Verified"

    if category == "Equipment":
        if "summary" in normalized or "stats" in normalized:
            return "Stats"
        if "overdue" in normalized:
            return "Overdue"
        if "poor condition" in normalized or "returned" in normalized:
            return "Returned"
        if "holding" in normalized:
            return "Holding"
        if "issued" in normalized:
            return "Issued"
        if "procured" in normalized:
            return "Procured"

    if category == "Equipment" and entities.get("equipmentName"):
        item_cat = _item_category(entities.get("equipmentName"))
        if item_cat == "IssuedItems" and any(token in normalized for token in ("issued", "holding", "overdue", "poor condition", "returned")):
            return SUBCATEGORY_TO_OPERATION.get("IssuedItems", "Issued")
        if item_cat == "ProcuredItems" and any(token in normalized for token in ("procured", "self purchased")):
            return SUBCATEGORY_TO_OPERATION.get("ProcuredItems", "Procured")
    if category == "Roster" and entities.get("sport"):
        return "BySport"
    if category == "Roster" and entities.get("class"):
        return "ByClass"
    if category == "Strength":
        return "Summary"
    if category == "Overall":
        return "OverallPerformance"
    return operation


def _infer_legacy_subcategory(
    category: Optional[str],
    operation: Optional[str],
    query: str,
    entities: Dict[str, Any],
) -> Optional[str]:
    normalized = _normalise(query)
    if category == "Verification":
        if "completed" in normalized:
            return "CompletedVerification"
        if "sent" in normalized:
            return "SentVerification"
        if "not responded" in normalized or "no response" in normalized:
            return "NotRespondedVerification"
        if "rejected" in normalized:
            return "RejectedVerification"
        if "pending" in normalized:
            return "PendingVerification"
        if "verified" in normalized:
            return "VerifiedVerification"

    if category == "Performance":
        if "pass percentage" in normalized or "pass rate" in normalized:
            return "PassPercentage"
        if "fail percentage" in normalized or "fail rate" in normalized:
            return "FailPercentage"
        if "grading summary" in normalized or "grade summary" in normalized:
            return "GradingSummary"
        if "grade distribution" in normalized or "grading distribution" in normalized:
            return "GradeDistribution"
        if any(token in normalized for token in ("compare", "comparison", " vs ", " versus ")):
            return "Comparison"

    if category and operation and (category, operation) in CATEGORY_OPERATION_TO_SUBCATEGORY:
        return CATEGORY_OPERATION_TO_SUBCATEGORY[(category, operation)]

    if category == "Equipment" and entities.get("equipmentName"):
        item_category = _item_category(entities["equipmentName"])
        if item_category and not any(token in normalized for token in ("poor condition", "returned", "overdue", "holding", "issued", "procured")):
            return item_category
        if "poor condition" in normalized or "returned" in normalized:
            return "PoorConditionEquipment"
        if "overdue" in normalized:
            return "OverdueEquipment"
        if "holding" in normalized:
            return "HoldingEquipment"
    if category == "Performance" and entities.get("section") and not operation:
        return "SectionSummary"
    if category == "Medical" and entities.get("medical_status") == "Active":
        return "ActiveCases"
    if category == "Attendance":
        if any(token in normalized for token in ("present", "on campus", "who is present")):
            return "PresentToday"
        if any(token in normalized for token in ("weekly", "week")):
            return "WeeklyAttendance"
        if any(token in normalized for token in ("yearly", "annual")):
            return "YearlyAttendance"
        if re.search(
            r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b",
            query,
            re.IGNORECASE,
        ):
            return "MonthlyAttendance"
        if entities.get("date") and not operation:
            return "DailyAttendance"
    if category == "Distribution":
        if any(token in normalized for token in ("top unit", "highest", "most")) and "unit" in normalized:
            return "TopUnit"
        if "unassigned" in normalized:
            return "UnassignedItems"
        if "latest" in normalized or "recent" in normalized:
            return "LatestDistribution"
        if "unit" in normalized:
            return "DistributionByUnit"
    if category == "Strength":
        return "StrengthBreakdown"
    if category == "Overall":
        return "OverallPerformance"
    return None


def _legacy_type(category: Optional[str], operation: Optional[str], subcategory: Optional[str]) -> Optional[str]:
    if not category or not subcategory:
        return None
    op_key = operation or SUBCATEGORY_TO_OPERATION.get(subcategory, subcategory)
    return INTENT_TYPE_DEFAULTS.get((category, op_key))


def classify_admin_intent(
    query: str,
    resolved_entities: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    raw_query = str(query or "").strip()
    resolved_entities = resolved_entities or {}

    lowered_query = raw_query.lower()
    if "may make mistakes" in lowered_query and "verify important information" in lowered_query:
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
            "responseType": "Summary",
            "raw_query": raw_query,
            "confidence": "low",
            "query_type": "simple",
            "filters": {},
        }

    intent_result = classify_intent(raw_query)
    entities = extract_entities(raw_query, resolved_entities)

    category = _infer_category(raw_query, intent_result, entities)
    operation = _infer_operation(category, intent_result, entities, raw_query)
    subcategory = _infer_legacy_subcategory(category, operation, raw_query, entities)

    if category == "Performance" and entities.get("section") in {"BPET", "PPT", "Firing", "Drill"} and not subcategory:
        subcategory = "SectionSummary"

    if not category and subcategory:
        category = next((cat for (cat, op), sub in CATEGORY_OPERATION_TO_SUBCATEGORY.items() if sub == subcategory), None)

    if not operation and subcategory:
        operation = SUBCATEGORY_TO_OPERATION.get(subcategory)

    if category == "Equipment" and entities.get("equipmentName") and not subcategory:
        subcategory = _item_category(entities["equipmentName"])

    normalized_query = _normalise(raw_query)

    if category == "Attendance" and "annual" in normalized_query:
        subcategory = "YearlyAttendance"
        operation = "Yearly"
    if category == "Attendance" and re.search(
        r"\b(January|February|March|April|May|June|July|August|September|October|November|December|Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4}\b",
        raw_query,
        re.IGNORECASE,
    ):
        subcategory = "MonthlyAttendance"
        operation = "Monthly"
    if category == "Attendance" and any(token in normalized_query for token in ("present", "on campus", "who is present")):
        subcategory = "PresentToday"
        operation = "Present"
    if category == "Attendance" and any(token in normalized_query for token in ("weekly", "week")):
        subcategory = "WeeklyAttendance"
        operation = "Weekly"
    if category == "Attendance" and any(token in normalized_query for token in ("daily", "today", "day")) and subcategory is None:
        subcategory = "DailyAttendance"
        operation = "Daily"

    if category == "Strength" and not subcategory:
        subcategory = "StrengthBreakdown"
        operation = "Summary"

    if category == "Overall" and not subcategory:
        subcategory = "OverallPerformance"
        operation = "OverallPerformance"

    legacy_type = _legacy_type(category, operation, subcategory)
    if "tabular" in normalized_query or "table" in normalized_query:
        legacy_type = "Tabular"

    confidence = "low"
    if category and subcategory:
        confidence = "high"
    elif category:
        confidence = "medium"
    if subcategory in {"TopPerformers", "LowestPerformers"} and entities.get("n") is None:
        confidence = "medium"

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

    if result["category"] == "Leave" and result["leave_type"] == "Current":
        result["operation"] = "Current"

    if result["category"] == "Medical" and result["bmi_category"]:
        result["operation"] = "BMI"
        result["subcategory"] = result["subcategory"] or "BMIAnalysis"

    if result["category"] == "Medical" and result["blood_group"]:
        result["operation"] = "BloodGroup"
        result["subcategory"] = result["subcategory"] or "BloodGroup"

    if result["category"] == "Verification" and not result["subcategory"] and result["operation"]:
        result["subcategory"] = CATEGORY_OPERATION_TO_SUBCATEGORY.get(("Verification", result["operation"]))

    if result["category"] == "Equipment" and result["subcategory"] in {"IssuedItems", "ProcuredItems"}:
        result["operation"] = SUBCATEGORY_TO_OPERATION.get(result["subcategory"], result["operation"])

    if result["category"] == "Distribution" and not result["subcategory"]:
        if "unit" in _normalise(raw_query):
            result["subcategory"] = "DistributionByUnit"
        elif "unassigned" in _normalise(raw_query):
            result["subcategory"] = "UnassignedItems"
        elif "latest" in _normalise(raw_query):
            result["subcategory"] = "LatestDistribution"

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
