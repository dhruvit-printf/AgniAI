"""
payload_builder.py
==================

Single responsibility: build AiCommandRequestDto.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .intent_schema import OPERATION_TO_PAYLOAD_FIELD


def _get_operation_for_payload(operation: Optional[str]) -> Optional[str]:
    if not operation:
        return None
    return OPERATION_TO_PAYLOAD_FIELD.get(operation, operation)


def build_ai_command_request_dto(
    category: Optional[str],
    operation: Optional[str],
    response_type: str,
    entities: Dict[str, Any],
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"commandId": 0, "responseType": response_type}

    if category:
        payload["category"] = category
    if operation:
        payload["operation"] = _get_operation_for_payload(operation)

    entity_mappings = {
        "n": "n",
        "section": "section",
        "subSection": "subSection",
        "grading": "grading",
        "leaveType": "leaveType",
        "bmiCategory": "bmiCategory",
        "bloodGroup": "bloodGroup",
        "equipmentName": "equipmentName",
        "sport": "sport",
        "class": "class",
        "unitName": "unitName",
        "attemptNo": "attemptNo",
        "fromAttempt": "fromAttempt",
        "toAttempt": "toAttempt",
        "date": "date",
        "fromDate": "fromDate",
        "toDate": "toDate",
        "companyId": "companyId",
        "platoonId": "platoonId",
        "batchId": "batchId",
        "agniveerNo": "agniveerNo",
        "medicalStatus": "medicalStatus",
        "diagnose": "diagnose",
        "isExceptional": "isExceptional",
        "days": "days",
    }

    for source_key, dest_key in entity_mappings.items():
        value = entities.get(source_key)
        if value is not None:
            payload[dest_key] = value

    return payload
