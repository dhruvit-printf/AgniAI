"""
payload_validator.py
====================

Single responsibility: validate entity-category compatibility.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .intent_schema import (
    get_allowed_entities_for_category,
    is_valid_category,
    is_valid_operation,
)


class ValidationError(Exception):
    pass


def _get_non_null_entities(entities: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in entities.items() if v is not None}


def _validate_operation_for_category(category: str, operation: Optional[str]) -> bool:
    return True if not operation else is_valid_operation(category, operation)


def _validate_entities_for_category(category: str, entities: Dict[str, Any]) -> List[str]:
    errors: List[str] = []
    allowed_entities = get_allowed_entities_for_category(category)
    for entity_type, value in _get_non_null_entities(entities).items():
        if entity_type not in allowed_entities:
            errors.append(
                f"Entity '{entity_type}' (value={value}) is not compatible with category '{category}'"
            )
    return errors


def validate_payload(
    category: Optional[str],
    operation: Optional[str],
    entities: Dict[str, Any],
) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []

    if not category:
        warnings.append("No category detected")
        return {"valid": False, "errors": errors, "warnings": warnings}

    if not is_valid_category(category):
        errors.append(f"Category '{category}' is not in official list")
        return {"valid": False, "errors": errors, "warnings": warnings}

    if not _validate_operation_for_category(category, operation):
        errors.append(f"Operation '{operation}' is not valid for category '{category}'")

    errors.extend(_validate_entities_for_category(category, entities))
    return {"valid": len(errors) == 0, "errors": errors, "warnings": warnings}


def validate_payload_strict(
    category: Optional[str],
    operation: Optional[str],
    entities: Dict[str, Any],
) -> None:
    result = validate_payload(category, operation, entities)
    if not result["valid"]:
        raise ValidationError("; ".join(result["errors"]))
