"""
admin_intent.py
===============
Intent classifier for the AgniAI Admin Chatbot.

Given a natural-language question from an admin, this module:
  1. Identifies the top-level module (Performance, Leave, Medical, etc.)
  2. Identifies the specific intent within that module
  3. Extracts any filters (number, section, grading, leave type, operation)
  4. Returns a structured JSON payload ready to POST to the .NET AiCommand API

Flow:
  Admin question → classify_admin_intent() → dict payload → POST to .NET

FIX — camelCase keys in format_admin_payload():
  The .NET JSON deserializer expects camelCase property names by default
  (standard ASP.NET Core behaviour with System.Text.Json).
  Internal Python keys use snake_case for readability; format_admin_payload()
  converts them to camelCase before sending to .NET:
    leave_type → leaveType
  All other keys (category, subcategory, number, operation, section, grading)
  are already single-word or naturally camelCase-compatible.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# INTENT MAPS
# =============================================================================

# Each entry: (intent_name, dotnet_intent_code, keyword_list)
_PERFORMANCE_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("Top Performers",          "TopPerformers",        ("top", "highest", "best", "topper", "rank 1", "first", "leading")),
    ("Lowest Performers",       "LowestPerformers",     ("bottom", "lowest", "worst", "weakest", "poor performer", "last")),
    ("Improvement",             "Improvement",          ("improvement", "improve", "improved", "progress", "better")),
    ("Decline",                 "Decline",              ("drop", "decline", "declined", "dropped", "regression", "fallen")),
    ("Grade Distribution",      "GradeDistribution",    ("grading", "grade", "grade distribution", "grades")),
    ("Grade Summary",           "GradingSummary",       ("gradingsummary", "grade summary", "grading summary", "gradesummary")),
    ("Average Score",           "AverageScore",         ("average", "avg", "mean", "average score", "mean score")),
    ("Attempt Wise Analysis",   "AttemptWise",          ("attemptwise", "by attempt", "attempt wise", "attempt analysis")),
    ("Best Attempt",            "BestAttempt",          ("best attempt", "bestattempt", "best score", "bestscoreattempt")),
    ("Comparison",              "Comparison",           ("compare", "comparison", "vs", "versus", "compared")),
    ("Section Summary",         "SectionSummary",       ("summary", "section summary", "sectionsummary", "section overview")),
    ("Pass Percentage",         "PassPercentage",       ("pass percentage", "passpercentage", "pass rate", "passed")),
    ("Fail Percentage",         "FailPercentage",       ("fail percentage", "failpercentage", "fail rate", "failed", "failure rate")),
    ("Overall Performance",     "OverallPerformance",   ("overall", "composite", "overall performance", "overall score")),
]

_LEAVE_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("Most Leave Taken",        "MostLeaveTaken",       ("most leave", "most leaves", "highest leave", "maximum leave", "most absent")),
    ("Least Leave Taken",       "LeastLeaveTaken",      ("least leave", "fewest leave", "lowest leave", "minimum leave")),
    ("Current Leave Status",    "CurrentLeaveStatus",   ("current leave", "leave today", "leave now", "on leave today", "who is on leave")),
    ("Absconded Personnel",     "AbscondedPersonnel",   ("absconded", "abscond", "awol", "missing")),
]

_MEDICAL_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("Active Cases",            "ActiveCases",          ("active", "admitted", "cases", "in ward", "hospitalised", "hospitalized", "current patients")),
    ("BMI Analysis",            "BMIAnalysis",          ("bmi", "weight", "fitness", "body mass", "overweight", "underweight")),
    ("Disease Statistics",      "DiseaseStatistics",    ("disease", "diagnoses", "top disease", "illness", "ailment", "medical condition")),
]

_ATTENDANCE_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("Monthly Attendance",      "MonthlyAttendance",    ("monthly", "month", "monthly stats", "attendance stats", "month attendance")),
    ("Present Today",           "PresentToday",         ("present", "campus", "today", "here today", "who is present", "on campus")),
    ("Strength Breakdown",      "StrengthBreakdown",    ("strength", "breakdown", "strength breakdown", "total strength")),
]

_VERIFICATION_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("Pending Verification",    "PendingVerification",  ("pending", "sent", "no response", "noresponse", "not verified", "awaiting")),
    ("Completed Verification",  "CompletedVerification",("completed", "verified", "done", "verification done")),
]

_EQUIPMENT_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("Equipment Summary",       "EquipmentSummary",     ("stats", "summary", "overview", "equipment stats", "equipment summary", "equipment overview")),
    ("Overdue Equipment",       "OverdueEquipment",     ("overdue", "pending", "late", "not returned", "overdue equipment")),
    ("Poor Condition Equipment","PoorConditionEquipment",("poor", "condition", "returned poor", "damaged", "bad condition")),
]

_DISTRIBUTION_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("Latest Distribution",     "LatestDistribution",   ("latest", "recent", "last distribution", "newest")),
    ("Distribution by Unit",    "DistributionByUnit",   ("by unit", "byunit", "unit distribution", "in unit", "inunit", "per unit")),
    ("Unassigned Items",        "UnassignedItems",      ("unassigned", "not assigned", "notassigned", "no unit")),
    ("Top Unit",                "TopUnit",              ("top unit", "topunit", "highest unit", "most distributed unit", "highest distribution", "top distribution unit")),
]

_SKILLS_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("By Sport",                "BySport",              ("by sport", "bysport", "sport", "sports", "sports skill", "game")),
    ("By Class",                "ByClass",              ("by class", "byclass", "class", "per class", "class wise")),
]

# Module → (keyword_triggers, intent_list)
_MODULES: Dict[str, Tuple[Tuple[str, ...], List[Tuple[str, str, Tuple[str, ...]]]]] = {
    "Performance": (
        ("performance", "score", "marks", "exam", "test", "result", "grade", "pass", "fail",
         "attempt", "section", "grading", "training test", "written", "cee"),
        _PERFORMANCE_INTENTS,
    ),
    "Leave": (
        ("leave", "absent", "absconded", "awol", "off duty", "sick leave", "annual leave",
         "medical leave", "on leave"),
        _LEAVE_INTENTS,
    ),
    "Medical": (
        ("medical", "bmi", "disease", "health", "hospital", "patient", "diagnosis",
         "fitness", "weight", "body mass"),
        _MEDICAL_INTENTS,
    ),
    "Attendance": (
        ("attendance", "present", "campus", "strength", "headcount", "muster"),
        _ATTENDANCE_INTENTS,
    ),
    "Verification": (
        ("verification", "verify", "verified", "pending verification", "document verification"),
        _VERIFICATION_INTENTS,
    ),
    "Equipment": (
        ("equipment", "gear", "overdue", "weapon", "kit", "issued", "inventory"),
        _EQUIPMENT_INTENTS,
    ),
    "Distribution": (
        ("distribution", "distributed", "issued to", "unit distribution", "item distribution",
         "unassigned", "notassigned", "latest distribution", "recent distribution"),
        _DISTRIBUTION_INTENTS,
    ),
    "Skills": (
        ("skill", "sport", "sports", "roster", "class skill", "sports skill"),
        _SKILLS_INTENTS,
    ),
}

# =============================================================================
# FILTER CONSTANTS
# =============================================================================

_SECTIONS = {"BEPT", "PPT", "FIRING", "DRILL"}

_GRADING_VALUES = {
    "exceptionally well": "ExceptionallyWell",
    "excellent":          "Excellent",
    "good":               "Good",
    "sat":                "SAT",
    "fail":               "Fail",
    "unsa":               "UNSA",
}

_LEAVE_TYPES = {
    "medical":   "Medical",
    "annual":    "Annual",
    "sick":      "Sick",
    "absconded": "Absconded",
}

_OPERATION_KEYWORDS = {
    "all":     "All",
    "top":     "Top",
    "bottom":  "Bottom",
    "first":   "First",
    "last":    "Last",
    "latest":  "Latest",
}


# =============================================================================
# HELPERS
# =============================================================================

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _extract_number(text: str) -> Optional[int]:
    """Extract the first integer from text (e.g. 'top 5 performers' → 5)."""
    match = re.search(r"\b(\d+)\b", text)
    return int(match.group(1)) if match else None


def _extract_section(text_lower: str) -> Optional[str]:
    for section in _SECTIONS:
        if section.lower() in text_lower:
            return section
    return None


def _extract_grading(text_lower: str) -> Optional[str]:
    for phrase, code in _GRADING_VALUES.items():
        if phrase in text_lower:
            return code
    return None


def _extract_leave_type(text_lower: str) -> Optional[str]:
    for phrase, code in _LEAVE_TYPES.items():
        if phrase in text_lower:
            return code
    return None


def _extract_operation(text_lower: str) -> Optional[str]:
    for phrase, code in _OPERATION_KEYWORDS.items():
        pattern = rf"\b{re.escape(phrase)}\b"
        if re.search(pattern, text_lower):
            return code
    return None


def _score_intent(
    query_lower: str,
    keywords: Tuple[str, ...],
) -> int:
    """Score how strongly the query matches a set of intent keywords."""
    score = 0
    for kw in keywords:
        # Multi-word keywords score higher
        if kw in query_lower:
            score += len(kw.split())
    return score


def _match_module(query_lower: str) -> Optional[str]:
    best_module = None
    best_score = 0
    for module, (triggers, _) in _MODULES.items():
        score = _score_intent(query_lower, triggers)
        if score > best_score:
            best_score = score
            best_module = module
    return best_module if best_score > 0 else None


def _match_intent(
    query_lower: str,
    intent_list: List[Tuple[str, str, Tuple[str, ...]]],
) -> Optional[Tuple[str, str]]:
    """Return (intent_name, dotnet_code) with the best keyword match."""
    best_name = None
    best_code = None
    best_score = 0
    for name, code, keywords in intent_list:
        score = _score_intent(query_lower, keywords)
        if score > best_score:
            best_score = score
            best_name = name
            best_code = code
    return (best_name, best_code) if best_score > 0 else None


# =============================================================================
# PUBLIC CLASSIFIER
# =============================================================================

def classify_admin_intent(query: str) -> Dict[str, Any]:
    """
    Analyse an admin's natural-language question and return a structured
    JSON payload for the .NET AiCommand/execute endpoint.

    Returns:
    {
        "category":    "Performance",           # top-level module
        "subcategory": "TopPerformers",         # .NET intent code
        "number":      5,                       # extracted count (or null)
        "operation":   "Top",                   # extracted operation (or null)
        "section":     "BEPT",                  # section filter (or null)
        "grading":     "Excellent",             # grading filter (or null)
        "leave_type":  null,                    # leave type (or null) — snake_case internally
        "raw_query":   "Who are the top 5 ...", # original question
        "confidence":  "high" | "medium" | "low"
    }

    Note: leave_type is stored in snake_case here for readability.
    format_admin_payload() converts it to leaveType (camelCase) when
    building the JSON payload sent to .NET.
    """
    raw_query = (query or "").strip()
    q = _normalise(raw_query)

    result: Dict[str, Any] = {
        "category":    None,
        "subcategory": None,
        "number":      None,
        "operation":   None,
        "section":     None,
        "grading":     None,
        "leave_type":  None,   # internal snake_case; converted to leaveType for .NET
        "raw_query":   raw_query,
        "confidence":  "low",
    }

    # ── Module detection ───────────────────────────────────────────────────
    module = _match_module(q)
    if module is None:
        # Try broader fallback: check all intent keywords across all modules
        best_module = None
        best_score = 0
        for mod, (_, intents) in _MODULES.items():
            for _, _, kws in intents:
                sc = _score_intent(q, kws)
                if sc > best_score:
                    best_score = sc
                    best_module = mod
        module = best_module

    if module is None:
        return result  # Cannot classify

    result["category"] = module
    _, intent_list = _MODULES[module]

    # ── Intent detection ───────────────────────────────────────────────────
    intent_match = _match_intent(q, intent_list)
    if intent_match:
        intent_name, intent_code = intent_match
        result["subcategory"] = intent_code
        result["confidence"] = "high"
    else:
        # Module matched but intent is ambiguous — use first intent as default
        if intent_list:
            result["subcategory"] = intent_list[0][1]
        result["confidence"] = "medium"

    # ── Filters ────────────────────────────────────────────────────────────
    result["number"]     = _extract_number(q)
    result["operation"]  = _extract_operation(q)
    result["section"]    = _extract_section(q)
    result["grading"]    = _extract_grading(q)
    result["leave_type"] = _extract_leave_type(q)

    # Downgrade confidence if no number was extracted for "top/bottom" intents
    if result["confidence"] == "high":
        if result["subcategory"] in ("TopPerformers", "LowestPerformers") and result["number"] is None:
            result["confidence"] = "medium"

    return result


def format_admin_payload(intent_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the exact payload that gets sent to the .NET AiCommand/execute endpoint.

    KEY FIX: .NET's default JSON deserializer (System.Text.Json) expects
    camelCase property names. This function maps Python snake_case keys to
    their camelCase equivalents for the outgoing request:

        leave_type  →  leaveType

    All other keys (category, subcategory, number, operation, section, grading)
    are single-word and already camelCase-compatible.

    Strips None values and internal-only fields (raw_query, confidence).
    """
    # Mapping: Python internal key → .NET JSON key
    _KEY_MAP = {
        "category":    "category",
        "subcategory": "subcategory",
        "number":      "number",
        "operation":   "operation",
        "section":     "section",
        "grading":     "grading",
        "leave_type":  "leaveType",   # FIX: snake_case → camelCase for .NET
    }

    payload: Dict[str, Any] = {}
    for python_key, dotnet_key in _KEY_MAP.items():
        value = intent_result.get(python_key)
        if value is not None:
            payload[dotnet_key] = value

    return payload