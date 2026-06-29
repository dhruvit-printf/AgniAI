"""
tests/test_equipment_items.py
==============================
Unit tests for IssuedItems / ProcuredItems intent classification.
Run with: pytest tests/test_equipment_items.py -v
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Minimal stubs so admin_intent imports without a real Flask app ──────────
import types

for mod in (
    "flask",
    "flask_cors",
    "flask_limiter",
    "flask_limiter.util",
    "dotenv",
    "requests",
):
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

# stub dotenv.load_dotenv
sys.modules["dotenv"].load_dotenv = lambda *a, **kw: None  # type: ignore

# ── Now we can import ────────────────────────────────────────────────────────
from intent_engine.admin_intent import (
    _item_category,
    classify_admin_intent,
    format_admin_payload,
)
from intent_engine.intent_schema import ISSUED_EQUIPMENT_ITEMS, PROCURED_EQUIPMENT_ITEMS

ISSUED_ITEMS = tuple(ISSUED_EQUIPMENT_ITEMS)
PROCURED_ITEMS = tuple(PROCURED_EQUIPMENT_ITEMS)

# =============================================================================
# Master-list sanity checks
# =============================================================================


def test_issued_items_count():
    assert len(ISSUED_ITEMS) == 46, f"Expected 46 issued items, got {len(ISSUED_ITEMS)}"


def test_procured_items_count():
    assert (
        len(PROCURED_ITEMS) == 53
    ), f"Expected 53 procured items, got {len(PROCURED_ITEMS)}"


def test_item_category_covers_all_issued():
    for item in ISSUED_ITEMS:
        assert (
            _item_category(item) == "IssuedItems"
        ), f"Expected IssuedItems for: {item}"


def test_item_category_covers_all_procured():
    for item in PROCURED_ITEMS:
        assert (
            _item_category(item) == "ProcuredItems"
        ), f"Expected ProcuredItems for: {item}"


# =============================================================================
# classify_admin_intent — end-to-end intent tests
# =============================================================================


def test_intent_issued_items_overview():
    r = classify_admin_intent("Show me all issued items")
    assert r["category"] == "Equipment"
    assert r["subcategory"] == "IssuedItems"


def test_intent_procured_items_overview():
    r = classify_admin_intent("List all procured items")
    assert r["category"] == "Equipment"
    assert r["subcategory"] == "ProcuredItems"


def test_intent_specific_issued_item():
    r = classify_admin_intent("Is the DMS Boot GP issued to all agniveers?")
    assert r["category"] == "Equipment"
    assert r["subcategory"] == "IssuedItems"
    assert r["item_name"] == "DMS Boot GP"
    assert r["item_category"] == "IssuedItems"


def test_intent_specific_procured_item():
    r = classify_admin_intent("Has the Rifle Sling been procured?")
    assert r["category"] == "Equipment"
    assert r["subcategory"] == "ProcuredItems"
    assert r["item_name"] == "Rifle Sling"
    assert r["item_category"] == "ProcuredItems"


def test_intent_jungle_shoes():
    r = classify_admin_intent("Do we have jungle shoes in the procured list?")
    assert r["category"] == "Equipment"
    assert r["subcategory"] == "ProcuredItems"
    assert r["item_name"] == "Jungle Shoes"


def test_intent_mug_steel():
    r = classify_admin_intent("mug steel — is it issued?")
    assert r["category"] == "Equipment"
    assert r["subcategory"] == "IssuedItems"
    assert r["item_name"] == "Mug Steel"


def test_intent_swimming_costumes():
    r = classify_admin_intent("Show swimming costumes availability")
    assert r["category"] == "Equipment"
    assert r["subcategory"] == "ProcuredItems"
    assert r["item_name"] == "Swimming Costumes"


def test_intent_blanket():
    r = classify_admin_intent("Is blanket an issued item?")
    assert r["category"] == "Equipment"
    assert r["subcategory"] == "IssuedItems"
    assert r["item_name"] == "Blanket"


def test_intent_health_card():
    r = classify_admin_intent("health card details")
    assert r["category"] == "Equipment"
    assert r["subcategory"] == "ProcuredItems"
    assert r["item_name"] == "Health Card"


def test_intent_kit_bag():
    r = classify_admin_intent("kit bag issued or not")
    assert r["category"] == "Equipment"
    assert r["subcategory"] == "IssuedItems"
    assert r["item_name"] == "Kit Bag"


# =============================================================================
# format_admin_payload — payload key tests
# =============================================================================


def test_payload_issued_items_keys():
    r = classify_admin_intent("Show all issued items")
    p = format_admin_payload(r)
    assert p.get("category") == "Equipment"
    assert p.get("operation") == "Issued"


def test_payload_procured_items_keys():
    r = classify_admin_intent("List procured items")
    p = format_admin_payload(r)
    assert p.get("category") == "Equipment"
    assert p.get("operation") == "Procured"


def test_payload_specific_item_name():
    r = classify_admin_intent("Show barret cap status")
    p = format_admin_payload(r)
    assert p.get("equipmentName") == "Barret Cap"
    assert "itemName" not in p
    assert "itemCategory" not in p


def test_payload_no_item_name_for_general_query():
    r = classify_admin_intent("Show equipment summary")
    p = format_admin_payload(r)
    assert "equipmentName" not in p


# =============================================================================
# Existing equipment intents still work after the change
# =============================================================================


def test_existing_equipment_summary_unchanged():
    r = classify_admin_intent("Give me an equipment summary")
    assert r["category"] == "Equipment"
    assert r["subcategory"] == "EquipmentSummary"


def test_existing_overdue_unchanged():
    r = classify_admin_intent("What equipment is overdue?")
    assert r["category"] == "Equipment"
    assert r["subcategory"] == "OverdueEquipment"


def test_existing_poor_condition_unchanged():
    r = classify_admin_intent("Show equipment returned in poor condition")
    assert r["category"] == "Equipment"
    assert r["subcategory"] == "PoorConditionEquipment"
