"""
admin_intent.py
===============
Intent classifier for the AgniAI Admin Chatbot.

Built directly from the official Agniveer AI Command API documentation.

CATEGORIES & OPERATIONS (exact strings sent to .NET):
  Performance  → Top, Bottom, Improvement, Drop, Grading, GradingSummary,
                 Average, AttemptWise, BestAttempt, Compare, Summary,
                 PassPercentage, FailPercentage, Overall
  Leave        → Most, Least, Current, Absconded
  Medical      → Active, BMI, Disease
  Attendance   → Monthly, Present, Strength
  Verification → Pending, Completed
  Equipment    → Stats, Overdue, Returned, IssuedItems, ProcuredItems
  Distribution → Latest, ByUnit, Unassigned, TopUnit
  Skills       → BySport, ByClass, BloodGroup
  Overall      → Overall   (separate top-level category — composite ranking)

FILTERS (exact camelCase keys sent to .NET):
  Performance: section, subSection, grading, attemptNo, fromAttempt, toAttempt, class, n
  Leave:       leaveType, n
  Attendance:  date
  Distribution: unitName
  Skills:      sport, class
  Equipment:   itemName, itemCategory

LEAVE TYPE VALUES (leaveType filter sent to .NET):
  Annual | Medical | Sick | Absconded | ATTNC | ExPPG

DESIGN NOTES:
  - "Overall" is a SEPARATE category (not Performance/Overall).
    Use it when the user asks for composite/overall ranking without a specific
    Performance sub-filter.  "Show overall attendance" → Attendance/Monthly.
  - Module trigger keywords must NOT include bare ambiguous words like "overall",
    "stats", "monthly" — those belong only in intent-level keywords so the
    module tiebreaker can resolve correctly.
  - Attendance queries containing "overall", "statistics", "stats" are
    routed to Attendance, not Performance.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# SUBCATEGORY → EXACT .NET OPERATION STRING
# =============================================================================

_SUBCATEGORY_TO_OPERATION: Dict[str, str] = {
    # Performance
    "TopPerformers":      "Top",
    "LowestPerformers":   "Bottom",
    "Improvement":        "Improvement",
    "Drop":               "Drop",
    "GradeDistribution":  "Grading",
    "GradingSummary":     "GradingSummary",
    "AverageScore":       "Average",
    "AttemptWise":        "AttemptWise",
    "BestAttempt":        "BestAttempt",
    "Comparison":         "Compare",
    "SectionSummary":     "Summary",
    "PassPercentage":     "PassPercentage",
    "FailPercentage":     "FailPercentage",
    "OverallPerformance": "Overall",
    # Leave
    "MostLeaveTaken":        "Most",
    "LeastLeaveTaken":       "Least",
    "CurrentLeaveStatus":    "Current",
    "AbscondedPerson":       "Absconded",
    # Medical
    "ActiveCases":       "Active",
    "BMIAnalysis":       "BMI",
    "DiseaseStatistics": "Disease",
    # Attendance
    "MonthlyAttendance":  "Monthly",
    "PresentToday":       "Present",
    "StrengthBreakdown":  "Strength",
    # Verification
    "PendingVerification":   "Pending",
    "CompletedVerification": "Completed",
    # Equipment
    "EquipmentSummary":       "Stats",
    "OverdueEquipment":       "Overdue",
    "PoorConditionEquipment": "Returned",
    "IssuedItems":            "IssuedItems",
    "ProcuredItems":          "ProcuredItems",
    # Distribution
    "LatestDistribution": "Latest",
    "DistributionByUnit": "ByUnit",
    "UnassignedItems":    "Unassigned",
    "TopUnit":            "TopUnit",
    # Skills
    "BySport":   "BySport",
    "ByClass":   "ByClass",
    "BloodGroup":"BloodGroup",
    # Overall (separate category)
    "OverallRanking": "Overall",
}


# =============================================================================
# INTENT LISTS
# =============================================================================

# ── Performance ──────────────────────────────────────────────────────────────
_PERFORMANCE_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("Top Performers", "TopPerformers", (
        "top performer", "highest performer", "best performer",
        "top scorer", "highest scorer", "best scorer",
        "top scoring", "highest scoring", "best scoring",
        "rank 1", "first rank", "topper", "leading performer",
        "top agniveer", "top agniveers", "highest agniveer",
        "best agniveer", "agniveers in performance",
        "who scored highest", "who is the best",
        "top", "highest", "best",
    )),
    ("Lowest Performers", "LowestPerformers", (
        "bottom performer", "lowest performer", "worst performer",
        "bottom scorer", "lowest scorer", "worst scorer",
        "bottom scoring", "lowest scoring", "worst scoring",
        "last rank", "poor performer", "weakest performer",
        "bottom agniveer", "lowest agniveer", "worst agniveer",
        "who scored lowest", "who is the worst",
        "bottom", "lowest", "worst",
    )),
    ("Improvement", "Improvement", (
        "improvement between attempts", "score improvement",
        "improved between", "improvement from attempt",
        "improvement", "improved", "improve",
        "progress", "getting better", "score increased",
    )),
    ("Drop", "Drop", (
        "score drop", "biggest drop", "score decline",
        "dropped between", "decline between",
        "drop", "decline", "declined", "dropped",
        "regression", "fallen", "getting worse", "score decreased",
    )),
    ("Grade Distribution", "GradeDistribution", (
        "filter by grading", "grade distribution",
        "grading distribution", "grades breakdown",
        "grading filter", "by grading",
        "grading", "grade",
    )),
    ("Grading Summary", "GradingSummary", (
        "grading summary", "grade summary",
        "gradingsummary", "gradesummary",
        "distribution of grades", "summary of grading",
    )),
    ("Average Score", "AverageScore", (
        "average score", "average marks", "mean score",
        "avg score", "average section score",
        "average subsection score", "average analysis",
        "average", "avg", "mean",
    )),
    ("Attempt Wise", "AttemptWise", (
        "attempt wise analysis", "attempt wise",
        "attemptwise", "by attempt",
        "attempt analysis", "per attempt",
        "attempts breakdown",
    )),
    ("Best Attempt", "BestAttempt", (
        "best attempt analysis", "best attempt",
        "bestattempt", "best score attempt",
        "bestscoreattempt", "highest attempt",
        "top attempt",
    )),
    ("Comparison", "Comparison", (
        "compare sections", "section comparison",
        "compare", "comparison", " vs ",
        "versus", "compared to", "difference between",
    )),
    ("Section Summary", "SectionSummary", (
        "section summary", "section overview",
        "sectionsummary", "summary by section",
        "section wise summary", "sectionwise",
        "summary",
    )),
    ("Pass Percentage", "PassPercentage", (
        "pass percentage", "passing percentage",
        "pass rate", "passing rate",
        "passpercentage", "passrate",
        "how many passed", "who passed",
        "passed",
    )),
    ("Fail Percentage", "FailPercentage", (
        "fail percentage", "failure percentage",
        "fail rate", "failure rate",
        "failpercentage", "failrate",
        "how many failed", "who failed",
        "failed", "failure",
    )),
    ("Overall Performance", "OverallPerformance", (
        "overall performance", "overall performance report",
        "overall score", "overall scoring",
        "composite performance", "all criteria",
        "overall performer", "overall performers",
        "overall ranking performance", "overall result",
        "overall marks",
    )),
]

# ── Leave ─────────────────────────────────────────────────────────────────────
_LEAVE_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("Most Leave Taken", "MostLeaveTaken", (
        "most leave taken", "most leaves taken",
        "highest leave taken", "maximum leave taken",
        "most absent", "taken most leave",
        "most leave", "most leaves",
        "highest leave", "maximum leave",
        "most",           # bare fallback — single-word signal
        "maximum",
    )),
    ("Least Leave Taken", "LeastLeaveTaken", (
        "least leave taken", "fewest leave taken",
        "lowest leave taken", "minimum leave taken",
        "taken least leave",
        "least leave", "fewest leave",
        "lowest leave", "minimum leave",
        "least",          # bare fallback — single-word signal
        "fewest",
        "minimum",
    )),
    ("Current Leave Status", "CurrentLeaveStatus", (
        "currently on leave", "who is on leave",
        "on leave today", "current leave status",
        "leave today", "leave now",
        "current leave", "leave status",
        "on leave now",
    )),
    ("Absconded Person", "AbscondedPerson", (
        "absconded leave records", "absconded person",
        "gone missing", "missing person",
        "absconded", "abscond", "awol",
    )),
]

# ── Medical ───────────────────────────────────────────────────────────────────
_MEDICAL_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("Active Cases", "ActiveCases", (
        "active medical cases", "active cases",
        "current patients", "admitted patients",
        "in ward", "hospitalised", "hospitalized",
        "medical case", "active", "cases", "admitted",
    )),
    ("BMI Analysis", "BMIAnalysis", (
        "bmi outliers", "bmi analysis",
        "body mass index", "weight analysis",
        "fitness analysis", "overweight", "underweight",
        "bmi", "weight", "fitness",
    )),
    ("Disease Statistics", "DiseaseStatistics", (
        "top diagnoses", "disease statistics",
        "top disease", "common disease",
        "disease analysis", "illness statistics",
        "most common disease", "frequent disease",
        "diagnoses", "diagnosis",
        "disease", "ailment",
    )),
]

# ── Attendance ────────────────────────────────────────────────────────────────
_ATTENDANCE_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("Monthly Attendance", "MonthlyAttendance", (
        "monthly attendance statistics",
        "attendance statistics for the current month",
        "attendance statistics for this month",
        "attendance statistics for the month",
        "overall attendance statistics",
        "overall attendance for the month",
        "overall attendance this month",
        "overall attendance",
        "attendance this month",
        "month wise attendance",
        "attendance by month",
        "attendance for the month",
        "attendance for this month",
        "current month attendance",
        "attendance stats",
        "monthly attendance",
        "attendance report",
        "monthly report",
        "monthly stats",
        "monthly",
        "month",
    )),
    ("Present Today", "PresentToday", (
        "present on campus", "present today",
        "who is present", "how many present",
        "attendance today", "today attendance",
        "on campus today", "on campus",
        "present", "campus",
    )),
    ("Strength Breakdown", "StrengthBreakdown", (
        "strength breakdown", "total strength",
        "strength report", "headcount breakdown",
        "strength", "breakdown",
    )),
]

# ── Verification ──────────────────────────────────────────────────────────────
_VERIFICATION_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("Pending Verification", "PendingVerification", (
        "pending verifications", "sent but no response",
        "no response verification", "awaiting verification",
        "verification pending", "not verified",
        "pending", "sent", "noresponse",
    )),
    ("Completed Verification", "CompletedVerification", (
        "completed verifications", "verification completed",
        "verification done", "verified documents",
        "completed", "verified", "done",
    )),
]

# ── Equipment ─────────────────────────────────────────────────────────────────
_EQUIPMENT_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("Equipment Summary", "EquipmentSummary", (
        "equipment stats", "equipment summary",
        "equipment overview", "equipment report",
        "gear summary", "kit summary", "inventory summary",
        "stats", "summary", "overview",
    )),
    ("Overdue Equipment", "OverdueEquipment", (
        "overdue equipment", "equipment overdue",
        "overdue returns", "not returned equipment",
        "late equipment", "equipment not returned",
        "overdue gear",
        "overdue", "late",
    )),
    ("Poor Condition Equipment", "PoorConditionEquipment", (
        "returned poor condition", "poor condition equipment",
        "equipment returned poor", "damaged equipment",
        "bad condition equipment", "equipment damaged",
        "returned", "poor condition", "poor", "damaged",
    )),
    ("Issued Items", "IssuedItems", (
        # List / overview queries
        "issued items", "items issued", "all issued items",
        "list of issued items", "issued equipment list",
        "kit issued", "gear issued", "uniform issued",
        "what was issued", "what is issued to agniveer",
        "issued clothing", "issued kit list",
        # Specific item name fragments
        "dms boot", "gp boot", "pt shoes brown",
        "cap fs", "mug steel", "blanket",
        "terry towel", "under pant woollen",
        "vest woollen og", "vest cotton h/s",
        "ground shed", "kit bag", "combat t shirt",
        "net mosquito", "pagari 5.5m",
        "combat coat", "belt ick", "combat dress",
        "line bedding", "ffd",
        "cover water proof og", "cover water pro sikh",
        "haver shack", "boot high ankle dvs",
        "spoon desert", "frog bayonet ick",
        "net camouflage h/d", "lases nylon",
        "pouches amn ick", "pack with allmn frame",
        "cord disk identity", "disk identity oval",
        "disk identity round", "water bottle plastic",
        "vest cotton white s4", "drawers cotton white",
        "jersey v neck", "short kd",
        "knee elbow", "socks woollen og",
        "trouser drill khaki", "shirt man khaki",
        "short kd light green", "jersey man woollen",
        "shirt angola drive", "trouser bd serge",
        "issued",
    )),
    ("Procured Items", "ProcuredItems", (
        # List / overview queries
        "procured items", "items procured", "all procured items",
        "list of procured items", "procured equipment list",
        "kit procured", "gear procured", "self purchased",
        "what was procured", "what is procured by agniveer",
        "bought items", "purchase list", "items to buy",
        "procured clothing", "procured kit list",
        # Specific item name fragments
        "rect bag khaki", "mufti shoes",
        "black socks", "black pagri with fifty",
        "drill shoes", "pt dress complete",
        "games dress", "mufti dress white",
        "regt tie with clip", "mufti blazer",
        "khaki half sleeves", "khaki full sleeves",
        "socks og", "green pagri with fifty",
        "og dress", "combat cap",
        "leather belt with crest", "bed sheet",
        "rect shoulder", "name plate black",
        "name plate khaki", "belt black",
        "water bottle 02 ltr", "mug plastic",
        "box steel with paint", "hanger",
        "health card", "bed card", "locker cloth",
        "firing data card",
        "track suit with name", "clip board",
        "256 pages copy", "white hanky",
        "steel thali", "glass steel", "spoon steel",
        "soap case", "indls photograph",
        "small bucket", "progress card", "mattress",
        "angola shirt with og pant", "big bucket",
        "pt vest with chest no", "underwear",
        "swimming costumes", "swimming cover",
        "jungle shoes", "barret cap", "rifle sling",
        "procured",
    )),
]

# ── Distribution ──────────────────────────────────────────────────────────────
_DISTRIBUTION_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("Latest Distribution", "LatestDistribution", (
        "latest distribution", "recent distribution",
        "last distribution", "newest distribution",
        "latest", "recent",
    )),
    ("Distribution By Unit", "DistributionByUnit", (
        "agniveers in unit", "distribution by unit",
        "how many agniveers in", "agniveers in the",
        "by unit distribution", "unit wise distribution",
        "per unit distribution", "in unit distribution",
        "byunit", "inunit",
        "by unit", "in unit",
    )),
    ("Unassigned Items", "UnassignedItems", (
        "not assigned to unit", "unassigned agniveers",
        "no unit assigned", "items without unit",
        "unassigned", "notassigned", "no unit",
    )),
    ("Top Unit", "TopUnit", (
        "unit with most agniveers", "top unit",
        "highest unit", "most agniveers in unit",
        "unit with highest", "highest distribution unit",
        "which unit has the highest", "which unit has most",
        "topunit", "highestunit",
    )),
]

# ── Skills ────────────────────────────────────────────────────────────────────
_SKILLS_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("By Sport", "BySport", (
        "best performers in sport", "best in sport",
        "by sport", "sport wise",
        "roster by sport", "skills by sport",
        "bysport", "sport", "sports",
    )),
    ("By Class", "ByClass", (
        "agniveers by class", "by class",
        "class wise", "roster by class",
        "skills by class", "byclass",
        "class",
    )),
    ("Blood Group", "BloodGroup", (
        "blood group statistics", "blood group distribution",
        "blood group", "blood type",
        "bloodgroup", "blood",
    )),
]

# ── Overall (separate top-level category) ─────────────────────────────────────
_OVERALL_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    ("Overall Ranking", "OverallRanking", (
        "overall top performers", "top overall performers",
        "overall ranking", "overall rank",
        "overall top 10", "top 10 overall",
        "top agniveers overall", "best agniveers overall",
        "overall composite", "composite ranking",
        "overall top agniveer",
        "overall",
    )),
]


# =============================================================================
# MODULE REGISTRY
# =============================================================================

_MODULES: Dict[str, Tuple[Tuple[str, ...], List[Tuple[str, str, Tuple[str, ...]]]]] = {

    # ── Attendance (BEFORE Performance) ───────────────────────────────────────
    "Attendance": (
        (
            "attendance statistics",
            "attendance stats",
            "overall attendance",
            "monthly attendance",
            "attendance this month",
            "attendance for the month",
            "attendance for this month",
            "current month attendance",
            "present on campus",
            "on campus today",
            "strength breakdown",
            "attendance",
            "present",
            "campus",
            "strength",
            "headcount",
            "muster",
        ),
        _ATTENDANCE_INTENTS,
    ),

    # ── Overall (BEFORE Performance) ──────────────────────────────────────────
    "Overall": (
        (
            "overall top performers",
            "top overall performers",
            "overall ranking",
            "overall rank",
            "top 10 overall",
            "top agniveers overall",
            "best agniveers overall",
            "overall composite",
        ),
        _OVERALL_INTENTS,
    ),

    # ── Performance ───────────────────────────────────────────────────────────
    "Performance": (
        (
            "top performer", "bottom performer", "worst performer",
            "best performer", "highest performer",
            "top scorer", "worst scorer", "lowest scorer",
            "score improvement", "score drop", "score decline",
            "pass percentage", "fail percentage",
            "grade distribution", "grading summary",
            "average score", "average marks",
            "attempt wise", "best attempt",
            "section summary", "section comparison",
            "who scored", "who passed", "who failed",
            "highest score", "lowest score", "best score", "worst score",
            "performance", "score", "marks", "exam", "grading",
            "bpet", "ppt", "firing", "drill",
            "performer", "performers",
            "attempt", "section",
        ),
        _PERFORMANCE_INTENTS,
    ),

    # ── Leave ─────────────────────────────────────────────────────────────────
    "Leave": (
        (
            "most leave taken", "least leave taken",
            "currently on leave", "who is on leave",
            "on leave today", "absconded leave",
            "leave taken", "leave status",
            "leave", "absent", "absconded", "awol",
            "sick leave", "annual leave", "medical leave",
            # new leave types
            "attnc", "attenc", "att nc", "attn c",
            "ex ppg", "exppg", "ex-ppg",
        ),
        _LEAVE_INTENTS,
    ),

    # ── Medical ───────────────────────────────────────────────────────────────
    "Medical": (
        (
            "active medical cases", "medical cases",
            "bmi analysis", "bmi outliers",
            "disease statistics", "top diagnoses",
            "body mass index", "fitness analysis",
            "medical", "bmi", "disease", "health",
            "hospital", "patient", "diagnosis", "fitness",
            "ward", "admitted", "ailment", "illness",
        ),
        _MEDICAL_INTENTS,
    ),

    # ── Verification ──────────────────────────────────────────────────────────
    "Verification": (
        (
            "pending verifications", "completed verifications",
            "verification pending", "awaiting verification",
            "verification", "verify", "verified",
            "document verification", "not verified",
        ),
        _VERIFICATION_INTENTS,
    ),

    # ── Equipment ─────────────────────────────────────────────────────────────
    "Equipment": (
        (
            # Existing triggers
            "equipment stats", "equipment summary",
            "overdue equipment", "overdue returns",
            "poor condition equipment", "returned poor condition",
            "equipment", "gear", "overdue", "weapon",
            "kit", "issued", "inventory", "damaged",
            # New: category-level
            "issued items", "procured items",
            "items issued", "items procured",
            "issued equipment", "procured equipment",
            "issued list", "procured list",
            "kit issued", "kit procured",
            "issued gear", "procured gear",
            "issued uniform", "issued clothing",
            "self purchased", "bought items",
            # New: high-signal issued item fragments
            "dms boot", "cap fs", "mug steel",
            "terry towel", "combat coat", "belt ick",
            "haver shack", "dvs boot", "net camouflage",
            "pagari 5.5m", "lases nylon",
            # New: high-signal procured item fragments
            "rect bag", "mufti shoes", "black pagri",
            "drill shoes", "games dress", "mufti dress",
            "regt tie", "mufti blazer", "socks og",
            "green pagri", "og dress", "combat cap",
            "leather belt with crest", "rifle sling",
            "barret cap", "jungle shoes",
        ),
        _EQUIPMENT_INTENTS,
    ),

    # ── Distribution ──────────────────────────────────────────────────────────
    "Distribution": (
        (
            "distribution by unit", "agniveers in unit",
            "how many agniveers in", "agniveers in the unit",
            "latest distribution", "unassigned agniveers",
            "unit with most agniveers",
            "distribution", "distributed", "issued to",
            "unit distribution", "unassigned",
        ),
        _DISTRIBUTION_INTENTS,
    ),

    # ── Skills ────────────────────────────────────────────────────────────────
    "Skills": (
        (
            "by sport", "best in sport", "roster by sport",
            "by class", "agniveers by class",
            "blood group distribution", "blood group statistics",
            "skill", "sport", "sports", "roster",
            "blood group", "blood",
        ),
        _SKILLS_INTENTS,
    ),
}


# =============================================================================
# ADMIN FUZZY VOCABULARY
# =============================================================================

ADMIN_FUZZY_VOCAB: Dict[str, str] = {
    # Module names
    "performace":    "performance",
    "performence":   "performance",
    "prefomance":    "performance",
    "preformance":   "performance",
    "performnce":    "performance",
    "attendence":    "attendance",
    "attendnce":     "attendance",
    "atendance":     "attendance",
    "attandance":    "attendance",
    "verfication":   "verification",
    "verifcation":   "verification",
    "verificaton":   "verification",
    "varification":  "verification",
    "distribtion":   "distribution",
    "distributon":   "distribution",
    "distibution":   "distribution",
    "equipement":    "equipment",
    "equiptment":    "equipment",
    "equipmnt":      "equipment",
    "equpment":      "equipment",
    "meical":        "medical",
    "medicl":        "medical",
    "medcal":        "medical",
    # Section names
    "bpet":          "bpet",
    "betp":          "bpet",
    "pptt":          "ppt",
    "fiiring":       "firing",
    "firng":         "firing",
    "fring":         "firing",
    "fireing":       "firing",
    "drll":          "drill",
    "dril":          "drill",
    # Operation words
    "performrs":     "performers",
    "preformers":    "performers",
    "perfomers":     "performers",
    "bottm":         "bottom",
    "lowst":         "lowest",
    "loest":         "lowest",
    "hihest":        "highest",
    "higest":        "highest",
    "avrage":        "average",
    "averge":        "average",
    "avrg":          "average",
    "improvment":    "improvement",
    "improvemnt":    "improvement",
    "percentge":     "percentage",
    "percntage":     "percentage",
    "gradig":        "grading",
    "gradng":        "grading",
    "gradeing":      "grading",
    "sumary":        "summary",
    "summry":        "summary",
    "comparson":     "comparison",
    "comparsion":    "comparison",
    "attmpt":        "attempt",
    "attepmpt":      "attempt",
    "atempt":        "attempt",
    # Leave
    "leeve":         "leave",
    "leve":          "leave",
    "abscnded":      "absconded",
    "absconed":      "absconded",
    # ── New leave type misspellings ────────────────────────────────────────
    # ATTNC variants
    "attnc":         "attnc",
    "attenc":        "attnc",
    "attn c":        "attnc",
    "att nc":        "attnc",
    "attnce":        "attnc",
    "attncc":        "attnc",
    "atnc":          "attnc",
    "atncl":         "attnc",
    # ExPPG variants
    "exppg":         "exppg",
    "ex ppg":        "exppg",
    "ex-ppg":        "exppg",
    "expg":          "exppg",
    "exppq":         "exppg",
    "ex pgg":        "exppg",
    "exppgg":        "exppg",
    # Attendance
    "presnt":        "present",
    "preent":        "present",
    "campas":        "campus",
    "strenght":      "strength",
    "strengh":       "strength",
    "montly":        "monthly",
    "monthyl":       "monthly",
    # Equipment
    "overdeu":       "overdue",
    "overdu":        "overdue",
    "condtion":      "condition",
    "conditon":      "condition",
    # Skills
    "blod":          "blood",
    "bloog":         "blood",
    "sportt":        "sport",
    "sprot":         "sport",
    "classs":        "class",
    "claas":         "class",
    # Common words
    "todya":         "today",
    "todday":        "today",
    "persnnel":      "person",
    "personel":      "person",
    "agniverr":      "agniveer",
    "agniver":       "agniveer",
    "excelent":      "excellent",
    "excellnt":      "excellent",
    "satifactory":   "satisfactory",
    "unassignd":     "unassigned",
    "unasigned":     "unassigned",
}

_ADMIN_CANONICAL_CASE: Dict[str, str] = {
    "bpet":   "BPET",
    "ppt":    "PPT",
    "firing": "Firing",
    "drill":  "Drill",
}


def admin_normalize_query(query: str) -> str:
    """Normalise an admin query: fix misspellings and restore canonical casing."""
    if not query:
        return query
    words = query.split()
    out: List[str] = []
    for word in words:
        suffix = ""
        core = word
        while core and core[-1] in "?.,!:;":
            suffix = core[-1] + suffix
            core = core[:-1]
        if not core:
            out.append(word)
            continue
        lower_core = core.lower()
        fixed = ADMIN_FUZZY_VOCAB.get(lower_core)
        if fixed is not None:
            core = fixed
            lower_core = fixed
        canonical = _ADMIN_CANONICAL_CASE.get(lower_core)
        if canonical is not None:
            core = canonical
        out.append(core + suffix)
    return " ".join(out)


# =============================================================================
# FILTER VALUE MAPS
# =============================================================================

_SECTION_MAP = {
    "bpet":   "BPET",
    "ppt":    "PPT",
    "firing": "Firing",
    "drill":  "Drill",
}

_SUBSECTION_PATTERNS = {
    "5km":       "5km",
    "5 km":      "5km",
    "chin up":   "Chin Ups",
    "chin-up":   "Chin Ups",
    "chinup":    "Chin Ups",
    "h rope":    "H Rope",
    "h-rope":    "H Rope",
    "hrope":     "H Rope",
}

_GRADING_MAP = {
    "exceptionally well": "ExceptionallyWell",
    "excellent":          "Excellent",
    "good":               "Good",
    "sat":                "SAT",
    "fail":               "Fail",
    "unsa":               "UNSA",
}

# ── Leave type filter map ──────────────────────────────────────────────────────
# Values are the exact strings sent to the .NET leaveType filter field.
_LEAVE_TYPE_MAP = {
    # Existing types
    "annual":    "Annual",
    "medical":   "Medical",
    "sick":      "Sick",
    "absconded": "Absconded",
    # ── New types ────────────────────────────────────────────────────────────
    # ATTNC  (Attendance Not Counted / Attendance Cause leave)
    "attnc":     "ATTNC",
    "attenc":    "ATTNC",
    "attn c":    "ATTNC",
    "att nc":    "ATTNC",
    # ExPPG  (Ex-PPG / Posting Pay Group leave)
    "ex ppg":    "ExPPG",
    "exppg":     "ExPPG",
    "ex-ppg":    "ExPPG",
}

_SPORT_MAP = {
    "cricket":    "Cricket",
    "football":   "Football",
    "running":    "Running",
    "basketball": "Basketball",
    "volleyball": "Volleyball",
    "kabaddi":    "Kabaddi",
    "hockey":     "Hockey",
}

_CLASS_MAP = {
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
# EQUIPMENT ITEM MASTER LISTS
# =============================================================================

ISSUED_ITEMS: List[str] = [
    "Pt Shoes Brown",
    "Cap FS",
    "Mug Steel",
    "Blanket",
    "DMS Boot GP",
    "Terry Towel Light Blue",
    "Under Pant Woollen",
    "Vest Woollen OG FS",
    "Vest Cotton H/S RN",
    "Pt Dress",
    "Ground Shed",
    "Kit Bag",
    "Combat T Shirt",
    "Net Mosquito",
    "Pagari 5.5m",
    "Combat Coat",
    "Belt ICK",
    "Combat Dress",
    "Line Bedding",
    "FFD",
    "Cover Water Proof OG",
    "Cover Water Pro Sikh",
    "Haver Shack All Rank",
    "Boot High Ankle DVS",
    "Spoon Desert",
    "Frog Bayonet ICK",
    "Net Camouflage H/D",
    "Lases Nylon Black 100cm for Footwear",
    "Pouches Amn ICK",
    "Pack with Allmn Frame",
    "Cord Disk identity",
    "Disk identity Oval",
    "Disk identity Round",
    "Water Bottle Plastic with C",
    "Vest Cotton White S4",
    "Drawers Cotton White",
    "Jersey V Neck",
    "Short KD",
    "Knee & Elbow",
    "Socks Woollen OG",
    "Trouser Drill Khaki Poly",
    "Shirt Man Khaki Poly",
    "Short KD Light Green",
    "Jersey Man Woollen OG",
    "Shirt Angola Drive",
    "Trouser Bd Serge",
]

PROCURED_ITEMS: List[str] = [
    "Rect Bag (Khaki)",
    "Pt Shoes",
    "Mufti Shoes",
    "Black Socks",
    "Black Pagri with Fifty",
    "Drill Shoes",
    "PT Dress Complete",
    "Games Dress",
    "Mufti Dress (White)",
    "Regt Tie with Clip",
    "Mufti Dress (Winter) / Mufti Blazer",
    "Khaki Half Sleeves (Drill)",
    "Khaki Full Sleeves (WT & Drill)",
    "Socks OG",
    "Green Pagri with Fifty",
    "OG Dress",
    "Combat Cap",
    "Leather Belt with Crest",
    "Bed Sheet",
    "Rect Shoulder",
    "Name Plate (Black)",
    "Name Plate (Khaki)",
    "Belt (Black)",
    "Water Bottle 02 Ltr",
    "Mug Plastic",
    "Box Steel with Paint",
    "Hanger",
    "Health Card",
    "Bed Card",
    "Locker Cloth",
    "Firing Data Card",
    "Vest",
    "Track Suit with Name",
    "Clip Board",
    "256 Pages Copy",
    "White Hanky",
    "Steel Thali",
    "Glass (Steel)",
    "Spoon (Steel)",
    "Soap Case",
    "Indls Photograph",
    "Small Bucket",
    "Progress Card",
    "Mattress",
    "Angola Shirt with OG Pant",
    "Big Bucket",
    "PT Vest with Chest No",
    "Underwear",
    "Swimming Costumes",
    "Swimming Cover",
    "Jungle Shoes",
    "Barret Cap",
    "Rifle Sling",
]

# Flat lookup: normalised item name → (canonical_name, subcategory_code)
_ITEM_LOOKUP: Dict[str, Tuple[str, str]] = {}
for _item in ISSUED_ITEMS:
    _ITEM_LOOKUP[_item.lower()] = (_item, "IssuedItems")
for _item in PROCURED_ITEMS:
    _ITEM_LOOKUP[_item.lower()] = (_item, "ProcuredItems")

_GENERIC_WORDS = {
    "by", "in", "for", "per", "top", "distribution", "show",
    "of", "the", "a", "an", "latest", "recent", "last",
    "get", "give", "show", "list",
}


# =============================================================================
# FILTER EXTRACTORS
# =============================================================================

def _normalise(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def _extract_number(text: str) -> Optional[int]:
    explicit = re.search(
        r"\b(?:top|bottom|show|best|worst|lowest|highest|last|first)\s+(\d+)\b",
        text,
        re.IGNORECASE,
    )
    if explicit:
        return int(explicit.group(1))
    stripped = re.sub(
        r"\b(?:attempt\s*(?:no\.?\s*)?|from\s*attempt\s*|to\s*attempt\s*)\d+\b",
        "",
        text,
        flags=re.IGNORECASE,
    )
    match = re.search(r"\b(\d+)\b", stripped)
    return int(match.group(1)) if match else None


def _extract_section(text_lower: str) -> Optional[str]:
    for key, val in _SECTION_MAP.items():
        if key in text_lower:
            return val
    return None


def _extract_subsection(text_lower: str) -> Optional[str]:
    for key, val in _SUBSECTION_PATTERNS.items():
        if key in text_lower:
            return val
    return None


def _extract_grading(text_lower: str) -> Optional[str]:
    for phrase, code in _GRADING_MAP.items():
        if phrase in text_lower:
            return code
    return None


def _extract_leave_type(text_lower: str, category: Optional[str] = None) -> Optional[str]:
    """Extract leaveType filter value from query text.

    Searches for all known leave type phrases (including ATTNC and ExPPG)
    using longest-match priority so "ex ppg" wins over bare "ex".
    The category guard is intentionally removed for the new types because
    ATTNC/ExPPG are always leave-domain terms.
    """
    if category and category not in ("Leave", None):
        # For the legacy types, only extract when already in Leave context.
        # For the new types (attnc / exppg) we allow extraction regardless.
        new_type_phrases = {"attnc", "attenc", "attn c", "att nc",
                            "ex ppg", "exppg", "ex-ppg"}
        found = None
        best_len = 0
        for phrase, code in _LEAVE_TYPE_MAP.items():
            if phrase in text_lower and len(phrase) > best_len:
                if phrase in new_type_phrases or category == "Leave":
                    found = code
                    best_len = len(phrase)
        return found

    # category is Leave (or None) — search all phrases, longest match wins
    found = None
    best_len = 0
    for phrase, code in _LEAVE_TYPE_MAP.items():
        if phrase in text_lower and len(phrase) > best_len:
            found = code
            best_len = len(phrase)
    return found


def _extract_sport(text_lower: str) -> Optional[str]:
    for phrase, code in _SPORT_MAP.items():
        if phrase in text_lower:
            return code
    return None


def _extract_class(text_lower: str) -> Optional[str]:
    for phrase, code in _CLASS_MAP.items():
        if phrase in text_lower:
            return code
    return None


def _extract_item_query(text_lower: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Identify a specific issued or procured item mentioned in the query.

    Returns (canonical_item_name, subcategory_code) where subcategory_code
    is "IssuedItems" or "ProcuredItems", or (None, None) if no match.

    Priority:
      1. Longest substring match against _ITEM_LOOKUP keys (prevents "vest"
         from shadowing "vest cotton white s4").
      2. Token-overlap fallback for abbreviated / paraphrased queries
         (requires >=2 shared content tokens).
    """
    # 1. Longest-match substring scan
    best_key: Optional[str] = None
    best_len = 0
    for key in _ITEM_LOOKUP:
        if key in text_lower and len(key) > best_len:
            best_key = key
            best_len = len(key)
    if best_key:
        name, cat = _ITEM_LOOKUP[best_key]
        return name, cat

    # 2. Token-overlap fallback
    _STOP = {
        "a", "an", "the", "of", "for", "in", "with", "and", "or",
        "is", "are", "was", "were", "show", "list", "get", "give",
        "tell", "me", "its", "item", "items", "all", "any", "which",
        "what", "who", "how", "where", "do", "does",
    }
    query_tokens = set(re.findall(r"[a-z]+", text_lower)) - _STOP
    if len(query_tokens) < 2:
        return None, None

    best_overlap = 0
    best_match: Optional[Tuple[str, str]] = None
    for key, (name, cat) in _ITEM_LOOKUP.items():
        item_tokens = set(re.findall(r"[a-z]+", key)) - _STOP
        if not item_tokens:
            continue
        overlap = len(query_tokens & item_tokens)
        if overlap >= 2 and overlap > best_overlap:
            best_overlap = overlap
            best_match = (name, cat)

    if best_match:
        return best_match
    return None, None


def _extract_unit_name(text: str) -> Optional[str]:
    match = re.search(
        r"\b(?:in unit|by unit|for unit|unit)\s+([A-Za-z][A-Za-z0-9]*)(\s+[Uu]nit)?\b",
        text,
        re.IGNORECASE,
    )
    if match:
        candidate = match.group(1).strip()
        if candidate.lower() not in _GENERIC_WORDS:
            return f"{candidate.title()} Unit"
    for m in re.finditer(r"\b([A-Za-z][A-Za-z0-9]*)\s+[Uu]nit\b", text):
        candidate = m.group(1).strip()
        if candidate.lower() not in _GENERIC_WORDS:
            return f"{candidate.title()} Unit"
    return None


def _extract_attempt_no(text_lower: str) -> Optional[int]:
    match = re.search(r"\battempt\s*(?:no\.?|number)?\s*(\d+)\b", text_lower)
    return int(match.group(1)) if match else None


def _extract_from_attempt(text_lower: str) -> Optional[int]:
    match = re.search(r"\bfrom\s*attempt\s*(\d+)\b", text_lower)
    return int(match.group(1)) if match else None


def _extract_to_attempt(text_lower: str) -> Optional[int]:
    match = (
        re.search(r"\bto\s*attempt\s*(\d+)\b", text_lower) or
        re.search(r"\battempt\s*\d+\s+to\s+(\d+)\b", text_lower)
    )
    return int(match.group(1)) if match else None


def _extract_date(text: str) -> Optional[str]:
    patterns = [
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{2}/\d{2}/\d{4})\b",
        r"\b((?:January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4})\b",
        r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


# =============================================================================
# SCORING & MATCHING
# =============================================================================

def _score_intent(query_lower: str, keywords: Tuple[str, ...]) -> int:
    score = 0
    for kw in keywords:
        if kw in query_lower:
            score += len(kw.split())
    return score


def _match_module(query_lower: str) -> Optional[str]:
    scores: Dict[str, int] = {}
    for module, (triggers, _) in _MODULES.items():
        scores[module] = _score_intent(query_lower, triggers)

    best_score = max(scores.values(), default=0)
    if best_score == 0:
        return None

    tied = [m for m, s in scores.items() if s == best_score]
    if len(tied) == 1:
        return tied[0]

    best_module = None
    best_intent_score = -1
    for module in tied:
        _, intent_list = _MODULES[module]
        intent_score = 0
        for _, _, kws in intent_list:
            intent_score += _score_intent(query_lower, kws)
        if intent_score > best_intent_score:
            best_intent_score = intent_score
            best_module = module

    if best_module is None:
        module_order = list(_MODULES.keys())
        for module in module_order:
            if module in tied:
                return module

    return best_module


def _match_intent(
    query_lower: str,
    intent_list: List[Tuple[str, str, Tuple[str, ...]]],
) -> Optional[Tuple[str, str]]:
    best_name  = None
    best_code  = None
    best_score = 0
    for name, code, keywords in intent_list:
        score = _score_intent(query_lower, keywords)
        if score > best_score:
            best_score = score
            best_name  = name
            best_code  = code
    return (best_name, best_code) if best_score > 0 else None


# =============================================================================
# PUBLIC API
# =============================================================================

def classify_admin_intent(query: str) -> Dict[str, Any]:
    """
    Classify an admin natural-language question.

    Returns internal Python dict (snake_case).
    Call format_admin_payload() to convert to .NET camelCase payload.
    """
    raw_query = (query or "").strip()
    q = _normalise(raw_query)

    result: Dict[str, Any] = {
        "category":     None,
        "subcategory":  None,
        "number":       None,
        "section":      None,
        "sub_section":  None,
        "grading":      None,
        "leave_type":   None,
        "sport":        None,
        "class":        None,
        "unit_name":    None,
        "attempt_no":   None,
        "from_attempt": None,
        "to_attempt":   None,
        "date":         None,
        "item_name":    None,
        "item_category": None,
        "raw_query":    raw_query,
        "confidence":   "low",
    }

    # ── Module detection ───────────────────────────────────────────────────────
    module = _match_module(q)

    if module is None:
        best_module = None
        best_score  = 0
        for mod, (_, intents) in _MODULES.items():
            for _, _, kws in intents:
                sc = _score_intent(q, kws)
                if sc > best_score:
                    best_score  = sc
                    best_module = mod
        module = best_module

    if module is None:
        return result

    result["category"] = module
    _, intent_list = _MODULES[module]

    # ── Intent detection ───────────────────────────────────────────────────────
    intent_match = _match_intent(q, intent_list)
    if intent_match:
        _, intent_code       = intent_match
        result["subcategory"] = intent_code
        result["confidence"]  = "high"
    else:
        result["subcategory"] = intent_list[0][1]
        result["confidence"]  = "medium"

    # ── Filter extraction ──────────────────────────────────────────────────────
    result["number"]       = _extract_number(q)
    result["section"]      = _extract_section(q)
    result["sub_section"]  = _extract_subsection(q)
    result["grading"]      = _extract_grading(q)
    result["leave_type"]   = _extract_leave_type(q, category=module)
    result["sport"]        = _extract_sport(q)
    result["class"]        = _extract_class(q)
    result["unit_name"]    = _extract_unit_name(raw_query)
    result["attempt_no"]   = _extract_attempt_no(q)
    result["from_attempt"] = _extract_from_attempt(q)
    result["to_attempt"]   = _extract_to_attempt(q)
    result["date"]         = _extract_date(raw_query)

    # ── Item lookup (Equipment only) ───────────────────────────────────────────
    item_name, item_cat = _extract_item_query(q)
    result["item_name"]     = item_name
    result["item_category"] = item_cat

    # If a specific item was found but subcategory resolved to a generic
    # Equipment intent, promote it to the item-specific subcategory.
    if item_cat and result.get("subcategory") in (
        "EquipmentSummary", "OverdueEquipment", "PoorConditionEquipment", None
    ):
        result["subcategory"] = item_cat
        if result.get("confidence") != "high":
            result["confidence"] = "medium"

    # ── Confidence downgrade ───────────────────────────────────────────────────
    if result["confidence"] == "high":
        if result["subcategory"] in ("TopPerformers", "LowestPerformers"):
            if result["number"] is None:
                result["confidence"] = "medium"

    return result


def format_admin_payload(intent_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert internal intent dict to .NET AiCommand payload (camelCase).

    Python key      → .NET key
    ─────────────────────────────────────
    subcategory     → operation  (via _SUBCATEGORY_TO_OPERATION)
    number          → n
    sub_section     → subSection
    leave_type      → leaveType
    unit_name       → unitName
    attempt_no      → attemptNo
    from_attempt    → fromAttempt
    to_attempt      → toAttempt
    date            → date
    item_name       → itemName
    item_category   → itemCategory

    None values and internal-only fields are stripped.
    """
    payload: Dict[str, Any] = {}

    if intent_result.get("category"):
        payload["category"] = intent_result["category"]

    subcategory = intent_result.get("subcategory")
    if subcategory:
        payload["operation"] = _SUBCATEGORY_TO_OPERATION.get(subcategory, subcategory)

    if intent_result.get("number") is not None:
        payload["n"] = intent_result["number"]

    if intent_result.get("section"):
        payload["section"] = intent_result["section"]

    if intent_result.get("sub_section"):
        payload["subSection"] = intent_result["sub_section"]

    if intent_result.get("grading"):
        payload["grading"] = intent_result["grading"]

    if intent_result.get("leave_type"):
        payload["leaveType"] = intent_result["leave_type"]

    if intent_result.get("sport"):
        payload["sport"] = intent_result["sport"]

    if intent_result.get("class"):
        payload["class"] = intent_result["class"]

    if intent_result.get("unit_name"):
        payload["unitName"] = intent_result["unit_name"]

    if intent_result.get("attempt_no") is not None:
        payload["attemptNo"] = intent_result["attempt_no"]

    if intent_result.get("from_attempt") is not None:
        payload["fromAttempt"] = intent_result["from_attempt"]

    if intent_result.get("to_attempt") is not None:
        payload["toAttempt"] = intent_result["to_attempt"]

    if intent_result.get("date"):
        payload["date"] = intent_result["date"]

    if intent_result.get("item_name"):
        payload["itemName"] = intent_result["item_name"]

    if intent_result.get("item_category"):
        payload["itemCategory"] = intent_result["item_category"]

    return payload