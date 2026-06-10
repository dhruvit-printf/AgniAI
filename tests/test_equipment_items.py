"""
tests/test_equipment_items.py
==============================
Unit tests for IssuedItems / ProcuredItems intent classification.
Run with: pytest tests/test_equipment_items.py -v
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# ── Minimal stubs so admin_intent imports without a real Flask app ──────────
import types
for mod in ("flask", "flask_cors", "flask_limiter", "flask_limiter.util",
            "dotenv", "requests"):
    if mod not in sys.modules:
        sys.modules[mod] = types.ModuleType(mod)

# stub dotenv.load_dotenv
sys.modules["dotenv"].load_dotenv = lambda *a, **kw: None  # type: ignore

# ── Now we can import ────────────────────────────────────────────────────────
from admin_intent import (
    classify_admin_intent,
    format_admin_payload,
    ISSUED_ITEMS,
    PROCURED_ITEMS,
    _extract_item_query,
    _ITEM_LOOKUP,
)


# =============================================================================
# Master-list sanity checks
# =============================================================================

def test_issued_items_count():
    assert len(ISSUED_ITEMS) == 46, f"Expected 46 issued items, got {len(ISSUED_ITEMS)}"


def test_procured_items_count():
    assert len(PROCURED_ITEMS) == 53, f"Expected 53 procured items, got {len(PROCURED_ITEMS)}"


def test_lookup_table_has_both_lists():
    # Every item should appear in the lookup exactly once
    for item in ISSUED_ITEMS:
        key = item.lower()
        assert key in _ITEM_LOOKUP, f"Missing from lookup: {item}"
        assert _ITEM_LOOKUP[key][1] == "IssuedItems"

    for item in PROCURED_ITEMS:
        key = item.lower()
        assert key in _ITEM_LOOKUP, f"Missing from lookup: {item}"
        assert _ITEM_LOOKUP[key][1] == "ProcuredItems"


# =============================================================================
# _extract_item_query — direct unit tests
# =============================================================================

def test_extract_exact_issued_item():
    name, cat = _extract_item_query("mug steel")
    assert cat == "IssuedItems"
    assert name == "Mug Steel"


def test_extract_exact_procured_item():
    name, cat = _extract_item_query("rifle sling")
    assert cat == "ProcuredItems"
    assert name == "Rifle Sling"


def test_extract_partial_issued_item():
    name, cat = _extract_item_query("what is the status of the combat t shirt")
    assert cat == "IssuedItems"
    assert name == "Combat T Shirt"


def test_extract_partial_procured_item():
    name, cat = _extract_item_query("do we have a barret cap in stock")
    assert cat == "ProcuredItems"
    assert name == "Barret Cap"


def test_extract_no_match():
    name, cat = _extract_item_query("what is the weather today")
    assert name is None
    assert cat is None


def test_extract_longer_key_wins():
    # "vest cotton white s4" should win over bare "vest"
    name, cat = _extract_item_query("vest cotton white s4 availability")
    assert name == "Vest Cotton White S4"
    assert cat == "IssuedItems"


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
    assert p.get("operation") == "IssuedItems"


def test_payload_procured_items_keys():
    r = classify_admin_intent("List procured items")
    p = format_admin_payload(r)
    assert p.get("category") == "Equipment"
    assert p.get("operation") == "ProcuredItems"


def test_payload_specific_item_name():
    r = classify_admin_intent("Show barret cap status")
    p = format_admin_payload(r)
    assert p.get("itemName") == "Barret Cap"
    assert p.get("itemCategory") == "ProcuredItems"


def test_payload_no_item_name_for_general_query():
    r = classify_admin_intent("Show equipment summary")
    p = format_admin_payload(r)
    assert "itemName" not in p
    assert "itemCategory" not in p


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