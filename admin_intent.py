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

PAYLOAD FORMAT (matches .NET AiCommand API spec exactly):
  The .NET endpoint expects these camelCase keys:
    category   → "Performance", "Leave", "Medical", etc.
    operation  → "Top", "Bottom", "Improvement", etc.  (alias from API docs)
    n          → top N records (integer)
    section    → "BEPT", "PPT", "FIRING", "DRILL"
    grading    → "SAT", "Excellent", "Good", "Fail"
    leaveType  → "Medical", "Annual", "Sick", "Absconded"
    unitName   → unit name for Distribution queries
    sport      → sport name for Skills queries
    class      → class name for Skills queries
    fromAttempt / toAttempt / attemptNo → attempt filters

  Example payloads (from API documentation):
    {"category":"Performance","operation":"Top","section":"BEPT","n":10}
    {"category":"Performance","operation":"Improvement","fromAttempt":1,"toAttempt":4,"n":10}
    {"category":"Leave","operation":"Most","leaveType":"Medical","n":10}
    {"category":"Medical","operation":"Active"}
    {"category":"Distribution","operation":"ByUnit","unitName":"Alpha Unit"}
    {"category":"Skills","operation":"BySport","sport":"Cricket"}
    {"category":"Skills","operation":"ByClass","class":"Sikh"}

Internal snake_case keys (Python side only, never sent to .NET):
  subcategory → internal code like "TopPerformers"; converted to operation via _SUBCATEGORY_TO_OPERATION
  number      → converted to "n" in the payload
  leave_type  → converted to "leaveType" in the payload
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# SUBCATEGORY → OPERATION MAPPING
# Maps internal Python subcategory codes to the .NET API operation strings
# exactly as documented in the AiCommand API reference.
# =============================================================================

_SUBCATEGORY_TO_OPERATION: Dict[str, str] = {
    # Performance
    "TopPerformers":          "Top",
    "LowestPerformers":       "Bottom",
    "Improvement":            "Improvement",
    "Decline":                "Drop",
    "GradeDistribution":      "Grading",
    "GradingSummary":         "GradingSummary",
    "AverageScore":           "Average",
    "AttemptWise":            "AttemptWise",
    "BestAttempt":            "BestAttempt",
    "Comparison":             "Compare",
    "SectionSummary":         "Summary",
    "PassPercentage":         "PassPercentage",
    "FailPercentage":         "FailPercentage",
    "OverallPerformance":     "Overall",
    # Leave
    "MostLeaveTaken":         "Most",
    "LeastLeaveTaken":        "Least",
    "CurrentLeaveStatus":     "Current",
    "AbscondedPersonnel":     "Absconded",
    # Medical
    "ActiveCases":            "Active",
    "BMIAnalysis":            "BMI",
    "DiseaseStatistics":      "Disease",
    # Attendance
    "MonthlyAttendance":      "Monthly",
    "PresentToday":           "Present",
    "StrengthBreakdown":      "Strength",
    # Verification
    "PendingVerification":    "Pending",
    "CompletedVerification":  "Completed",
    # Equipment
    "EquipmentSummary":       "Stats",
    "OverdueEquipment":       "Overdue",
    "PoorConditionEquipment": "Returned",
    # Distribution
    "LatestDistribution":     "Latest",
    "DistributionByUnit":     "ByUnit",
    "UnassignedItems":        "Unassigned",
    "TopUnit":                "TopUnit",
    # Skills / Roster
    "BySport":                "BySport",
    "ByClass":                "ByClass",
    "BloodGroup":             "BloodGroup",
}


# =============================================================================
# INTENT MAPS
# Each entry: (intent_name, internal_subcategory_code, keyword_list)
# =============================================================================

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
    ("Blood Group",             "BloodGroup",           ("blood group", "bloodgroup", "blood", "blood type")),
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
        ("skill", "sport", "sports", "roster", "class skill", "sports skill", "blood group", "blood"),
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

# Sport names recognised for Skills/BySport queries
_SPORT_NAMES = {
    "cricket":    "Cricket",
    "football":   "Football",
    "running":    "Running",
    "basketball": "Basketball",
    "volleyball": "Volleyball",
    "kabaddi":    "Kabaddi",
    "hockey":     "Hockey",
}

# Class names recognised for Skills/ByClass queries
_CLASS_NAMES = {
    "sikh":    "Sikh",
    "oic":     "OIC",
    "gurkha":  "Gurkha",
    "gorkha":  "Gurkha",
    "dogra":   "Dogra",
    "jat":     "Jat",
    "rajput":  "Rajput",
    "punjabi": "Punjabi",
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


def _extract_leave_type(text_lower: str, category: Optional[str] = None) -> Optional[str]:
    """
    Extract leave type filter. Only applies when the category is Leave.
    Avoids false-positives like 'medical cases' triggering leaveType=Medical
    in the Medical category.
    """
    # Only extract leaveType when we are in the Leave module
    if category and category != "Leave":
        return None
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


def _extract_sport(text_lower: str) -> Optional[str]:
    for phrase, code in _SPORT_NAMES.items():
        if phrase in text_lower:
            return code
    return None


def _extract_class(text_lower: str) -> Optional[str]:
    for phrase, code in _CLASS_NAMES.items():
        if phrase in text_lower:
            return code
    return None


def _extract_unit_name(text: str) -> Optional[str]:
    """
    Extract a unit name from text.
    Handles patterns like:
      "Alpha Unit"                    → "Alpha Unit"
      "by unit Alpha Unit"            → "Alpha Unit"
      "in unit Bravo"                 → "Bravo Unit"
      "Distribution by unit Alpha Unit" → "Alpha Unit"
    """
    # Generic words that should NOT be treated as unit names
    _GENERIC = {
        "by", "in", "for", "per", "top", "distribution", "show",
        "of", "the", "a", "an", "latest", "recent", "last",
    }

    # Priority 1: "by unit <Name>" / "in unit <Name>" etc. — strip the keyword first
    match_kw = re.search(
        r"\b(?:in unit|by unit|for unit|unit)\s+([A-Za-z][A-Za-z0-9]*)(\s+[Uu]nit)?\b",
        text,
        re.IGNORECASE,
    )
    if match_kw:
        name_part = match_kw.group(1).strip()
        if name_part.lower() not in _GENERIC:
            unit_suffix = match_kw.group(2)  # " Unit" if present
            if unit_suffix:
                return f"{name_part.title()} Unit"
            return f"{name_part.title()} Unit"

    # Priority 2: "<Name> Unit" anywhere in the text
    for m in re.finditer(
        r"\b([A-Za-z][A-Za-z0-9]*)\s+[Uu]nit\b",
        text,
    ):
        candidate = m.group(1).strip()
        if candidate.lower() not in _GENERIC:
            return f"{candidate.title()} Unit"

    return None


def _extract_attempt_no(text_lower: str) -> Optional[int]:
    match = re.search(r"\battempt\s*(?:no\.?|number)?\s*(\d+)\b", text_lower)
    return int(match.group(1)) if match else None


def _extract_from_attempt(text_lower: str) -> Optional[int]:
    match = re.search(r"\bfrom\s*attempt\s*(\d+)\b", text_lower)
    return int(match.group(1)) if match else None


def _extract_to_attempt(text_lower: str) -> Optional[int]:
    match = re.search(r"\bto\s*attempt\s*(\d+)\b", text_lower)
    return int(match.group(1)) if match else None


def _score_intent(
    query_lower: str,
    keywords: Tuple[str, ...],
) -> int:
    """Score how strongly the query matches a set of intent keywords."""
    score = 0
    for kw in keywords:
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
    """Return (intent_name, subcategory_code) with the best keyword match."""
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
    result dict for the .NET AiCommand/execute endpoint.

    Returns (Python internal format — snake_case):
    {
        "category":    "Performance",
        "subcategory": "TopPerformers",     ← internal code, NOT sent to .NET
        "number":      5,                   ← converted to "n" for .NET
        "operation":   "Top",
        "section":     "BEPT",
        "grading":     null,
        "leave_type":  null,                ← converted to "leaveType" for .NET
        "sport":       null,
        "class":       null,
        "unit_name":   null,                ← converted to "unitName" for .NET
        "attempt_no":  null,
        "from_attempt":null,
        "to_attempt":  null,
        "raw_query":   "...",
        "confidence":  "high" | "medium" | "low"
    }
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
        "leave_type":  None,
        "sport":       None,
        "class":       None,
        "unit_name":   None,
        "attempt_no":  None,
        "from_attempt":None,
        "to_attempt":  None,
        "raw_query":   raw_query,
        "confidence":  "low",
    }

    # ── Module detection ───────────────────────────────────────────────────
    module = _match_module(q)
    if module is None:
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
        return result

    result["category"] = module
    _, intent_list = _MODULES[module]

    # ── Intent detection ───────────────────────────────────────────────────
    intent_match = _match_intent(q, intent_list)
    if intent_match:
        intent_name, intent_code = intent_match
        result["subcategory"] = intent_code
        result["confidence"] = "high"
    else:
        if intent_list:
            result["subcategory"] = intent_list[0][1]
        result["confidence"] = "medium"

    # ── Filters ────────────────────────────────────────────────────────────
    result["number"]      = _extract_number(q)
    result["operation"]   = _extract_operation(q)
    result["section"]     = _extract_section(q)
    result["grading"]     = _extract_grading(q)
    result["leave_type"]  = _extract_leave_type(q, category=module)
    result["sport"]       = _extract_sport(q)
    result["class"]       = _extract_class(q)
    result["unit_name"]   = _extract_unit_name(raw_query)
    result["attempt_no"]  = _extract_attempt_no(q)
    result["from_attempt"]= _extract_from_attempt(q)
    result["to_attempt"]  = _extract_to_attempt(q)

    # Downgrade confidence if no number for top/bottom intents
    if result["confidence"] == "high":
        if result["subcategory"] in ("TopPerformers", "LowestPerformers") and result["number"] is None:
            result["confidence"] = "medium"

    return result


def format_admin_payload(intent_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Build the exact JSON payload to POST to .NET /api/AiCommand/execute.

    Maps internal Python keys → .NET camelCase keys per the API documentation:
        subcategory  → operation  (via _SUBCATEGORY_TO_OPERATION)
        number       → n
        leave_type   → leaveType
        unit_name    → unitName
        from_attempt → fromAttempt
        to_attempt   → toAttempt
        attempt_no   → attemptNo

    Strips None values and internal-only fields (raw_query, confidence).

    Example outputs matching the API docs:
        {"category":"Performance","operation":"Top","section":"BEPT","n":10}
        {"category":"Leave","operation":"Most","leaveType":"Medical","n":10}
        {"category":"Distribution","operation":"ByUnit","unitName":"Alpha Unit"}
        {"category":"Skills","operation":"BySport","sport":"Cricket"}
    """
    payload: Dict[str, Any] = {}

    # category — passed through directly
    if intent_result.get("category"):
        payload["category"] = intent_result["category"]

    # subcategory → operation (using the alias table from API docs)
    subcategory = intent_result.get("subcategory")
    if subcategory:
        payload["operation"] = _SUBCATEGORY_TO_OPERATION.get(subcategory, subcategory)

    # number → n
    if intent_result.get("number") is not None:
        payload["n"] = intent_result["number"]

    # section
    if intent_result.get("section"):
        payload["section"] = intent_result["section"]

    # grading
    if intent_result.get("grading"):
        payload["grading"] = intent_result["grading"]

    # leave_type → leaveType
    if intent_result.get("leave_type"):
        payload["leaveType"] = intent_result["leave_type"]

    # sport
    if intent_result.get("sport"):
        payload["sport"] = intent_result["sport"]

    # class
    if intent_result.get("class"):
        payload["class"] = intent_result["class"]

    # unit_name → unitName
    if intent_result.get("unit_name"):
        payload["unitName"] = intent_result["unit_name"]

    # attempt_no → attemptNo
    if intent_result.get("attempt_no") is not None:
        payload["attemptNo"] = intent_result["attempt_no"]

    # from_attempt → fromAttempt
    if intent_result.get("from_attempt") is not None:
        payload["fromAttempt"] = intent_result["from_attempt"]

    # to_attempt → toAttempt
    if intent_result.get("to_attempt") is not None:
        payload["toAttempt"] = intent_result["to_attempt"]

    return payload