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
  Equipment    → Stats, Overdue, Returned
  Distribution → Latest, ByUnit, Unassigned, TopUnit
  Skills       → BySport, ByClass, BloodGroup

FILTERS (exact camelCase keys sent to .NET):
  section      → BEPT, PPT, Firing, Drill
  subSection   → 5km, Chin Ups, H Rope, etc.
  grading      → SAT, Excellent, Good, Fail
  attemptNo    → specific attempt number
  fromAttempt  → improvement start
  toAttempt    → improvement end
  leaveType    → Annual, Medical, Sick, Absconded
  unitName     → unit/team name
  sport        → Cricket, Football, Running, etc.
  class        → Sikh, OIC, Gurkha, etc.
  n            → top N records
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple


# =============================================================================
# SUBCATEGORY → EXACT .NET OPERATION STRING
# Taken verbatim from the API documentation.
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
    "MostLeaveTaken":     "Most",
    "LeastLeaveTaken":    "Least",
    "CurrentLeaveStatus": "Current",
    "AbscondedPersonnel": "Absconded",
    # Medical
    "ActiveCases":        "Active",
    "BMIAnalysis":        "BMI",
    "DiseaseStatistics":  "Disease",
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
    # Distribution
    "LatestDistribution": "Latest",
    "DistributionByUnit": "ByUnit",
    "UnassignedItems":    "Unassigned",
    "TopUnit":            "TopUnit",
    # Skills
    "BySport":    "BySport",
    "ByClass":    "ByClass",
    "BloodGroup": "BloodGroup",
}


# =============================================================================
# INTENT MAPS
# Keywords sourced from official aliases + natural language variations.
# Longer / more specific phrases are listed first so they score higher.
# Format: (display_name, subcategory_code, keyword_tuple)
# =============================================================================

_PERFORMANCE_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    # Aliases: Top, Highest, Best
    ("Top Performers", "TopPerformers", (
        "top performer", "highest performer", "best performer",
        "top scorer", "highest scorer", "best scorer",
        "top scoring", "highest scoring", "best scoring",
        "rank 1", "first rank", "topper", "leading performer",
        "top agniveer", "top agniveers", "highest agniveer",
        "best agniveer", "agniveers in performance",
        "top", "highest", "best",
    )),
    # Aliases: Bottom, Lowest, Worst
    ("Lowest Performers", "LowestPerformers", (
        "bottom performer", "lowest performer", "worst performer",
        "bottom scorer", "lowest scorer", "worst scorer",
        "bottom scoring", "lowest scoring", "worst scoring",
        "last rank", "poor performer", "weakest performer",
        "bottom agniveer", "lowest agniveer", "worst agniveer",
        "bottom", "lowest", "worst",
    )),
    # Aliases: Improvement, Improve, Improved
    ("Improvement", "Improvement", (
        "improvement between attempts", "score improvement",
        "improved between", "improvement from attempt",
        "improvement", "improved", "improve",
        "progress", "getting better", "score increased",
    )),
    # Aliases: Drop, Decline, Dropped
    ("Drop", "Drop", (
        "score drop", "biggest drop", "score decline",
        "dropped between", "decline between",
        "drop", "decline", "declined", "dropped",
        "regression", "fallen", "getting worse", "score decreased",
    )),
    # Aliases: Grading, Grade
    ("Grade Distribution", "GradeDistribution", (
        "filter by grading", "grade distribution",
        "grading distribution", "grades breakdown",
        "grading filter", "by grading",
        "grading", "grade",
    )),
    # Aliases: GradingSummary, GradeSummary
    ("Grading Summary", "GradingSummary", (
        "grading summary", "grade summary",
        "gradingsummary", "gradesummary",
        "distribution of grades", "summary of grading",
    )),
    # Aliases: Average, Avg, Mean
    ("Average Score", "AverageScore", (
        "average score", "average marks", "mean score",
        "avg score", "average section score",
        "average subsection score", "average analysis",
        "average", "avg", "mean",
    )),
    # Aliases: AttemptWise, ByAttempt, Attempts
    ("Attempt Wise", "AttemptWise", (
        "attempt wise analysis", "attempt wise",
        "attemptwise", "by attempt",
        "attempt analysis", "per attempt",
        "attempts breakdown",
    )),
    # Aliases: BestAttempt, BestScoreAttempt
    ("Best Attempt", "BestAttempt", (
        "best attempt analysis", "best attempt",
        "bestattempt", "best score attempt",
        "bestscoreattempt", "highest attempt",
        "top attempt",
    )),
    # Aliases: Compare, Comparison, VS
    ("Comparison", "Comparison", (
        "compare sections", "section comparison",
        "compare", "comparison", " vs ",
        "versus", "compared to", "difference between",
    )),
    # Aliases: Summary, SectionSummary
    ("Section Summary", "SectionSummary", (
        "section summary", "section overview",
        "sectionsummary", "summary by section",
        "section wise summary", "sectionwise",
        "summary",
    )),
    # Aliases: PassPercentage, PassRate
    ("Pass Percentage", "PassPercentage", (
        "pass percentage", "passing percentage",
        "pass rate", "passing rate",
        "passpercentage", "passrate",
        "how many passed", "who passed",
        "passed",
    )),
    # Aliases: FailPercentage, FailRate
    ("Fail Percentage", "FailPercentage", (
        "fail percentage", "failure percentage",
        "fail rate", "failure rate",
        "failpercentage", "failrate",
        "how many failed", "who failed",
        "failed", "failure",
    )),
    # Aliases: Overall, Composite, AllCriteria
    ("Overall Performance", "OverallPerformance", (
        "overall performance", "overall report",
        "overall score", "composite performance",
        "all criteria", "allcriteria",
        "overall", "composite",
    )),
]

_LEAVE_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    # Aliases: Most, Highest, Maximum
    ("Most Leave Taken", "MostLeaveTaken", (
        "most leave taken", "most leaves taken",
        "highest leave taken", "maximum leave taken",
        "most absent", "taken most leave",
        "most leave", "most leaves",
        "highest leave", "maximum leave",
    )),
    # Aliases: Least, Lowest, Minimum
    ("Least Leave Taken", "LeastLeaveTaken", (
        "least leave taken", "fewest leave taken",
        "lowest leave taken", "minimum leave taken",
        "taken least leave",
        "least leave", "fewest leave",
        "lowest leave", "minimum leave",
    )),
    # Aliases: Current, Today, Now
    ("Current Leave Status", "CurrentLeaveStatus", (
        "currently on leave", "who is on leave",
        "on leave today", "current leave status",
        "leave today", "leave now",
        "current leave", "leave status",
        "on leave now",
    )),
    # Aliases: Absconded, Abscond
    ("Absconded Personnel", "AbscondedPersonnel", (
        "absconded leave records", "absconded personnel",
        "gone missing", "missing personnel",
        "absconded", "abscond", "awol",
    )),
]

_MEDICAL_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    # Aliases: Active, Cases, Admitted
    ("Active Cases", "ActiveCases", (
        "active medical cases", "active cases",
        "current patients", "admitted patients",
        "in ward", "hospitalised", "hospitalized",
        "medical case", "active", "cases", "admitted",
    )),
    # Aliases: BMI, Weight, Fitness
    ("BMI Analysis", "BMIAnalysis", (
        "bmi outliers", "bmi analysis",
        "body mass index", "weight analysis",
        "fitness analysis", "overweight", "underweight",
        "bmi", "weight", "fitness",
    )),
    # Aliases: Disease, Diagnoses, Diagnosis, Top
    ("Disease Statistics", "DiseaseStatistics", (
        "top diagnoses", "disease statistics",
        "top disease", "common disease",
        "disease analysis", "illness statistics",
        "most common disease", "frequent disease",
        "diagnoses", "diagnosis",
        "disease", "ailment",
    )),
]

_ATTENDANCE_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    # Aliases: Monthly, Month, Stats
    ("Monthly Attendance", "MonthlyAttendance", (
        "monthly attendance statistics", "monthly attendance",
        "attendance this month", "month wise attendance",
        "attendance stats", "attendance by month",
        "monthly stats",
        "monthly", "month",
    )),
    # Aliases: Present, Campus, Today
    ("Present Today", "PresentToday", (
        "present on campus", "present today",
        "who is present", "how many present",
        "attendance today", "today attendance",
        "on campus today", "on campus",
        "present", "campus",
    )),
    # Aliases: Strength, Breakdown
    ("Strength Breakdown", "StrengthBreakdown", (
        "strength breakdown", "total strength",
        "strength report", "headcount breakdown",
        "strength", "breakdown",
    )),
]

_VERIFICATION_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    # Aliases: Pending, Sent, NoResponse
    ("Pending Verification", "PendingVerification", (
        "pending verifications", "sent but no response",
        "no response verification", "awaiting verification",
        "verification pending", "not verified",
        "pending", "sent", "noresponse",
    )),
    # Aliases: Completed, Verified, Done
    ("Completed Verification", "CompletedVerification", (
        "completed verifications", "verification completed",
        "verification done", "verified documents",
        "completed", "verified", "done",
    )),
]

_EQUIPMENT_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    # Aliases: Stats, Summary, Overview
    ("Equipment Summary", "EquipmentSummary", (
        "equipment stats", "equipment summary",
        "equipment overview", "equipment report",
        "gear summary", "kit summary", "inventory summary",
        "stats", "summary", "overview",
    )),
    # Aliases: Overdue, Pending, Late
    ("Overdue Equipment", "OverdueEquipment", (
        "overdue equipment", "equipment overdue",
        "overdue returns", "not returned equipment",
        "late equipment", "equipment not returned",
        "overdue gear",
        "overdue", "late",
    )),
    # Aliases: Returned, Poor, Condition
    ("Poor Condition Equipment", "PoorConditionEquipment", (
        "returned poor condition", "poor condition equipment",
        "equipment returned poor", "damaged equipment",
        "bad condition equipment", "equipment damaged",
        "returned", "poor condition", "poor", "damaged",
    )),
]

_DISTRIBUTION_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    # Aliases: Latest, Recent, Last
    ("Latest Distribution", "LatestDistribution", (
        "latest distribution", "recent distribution",
        "last distribution", "newest distribution",
        "latest", "recent",
    )),
    # Aliases: ByUnit, Unit, InUnit
    ("Distribution By Unit", "DistributionByUnit", (
        "agniveers in unit", "distribution by unit",
        "by unit distribution", "unit wise distribution",
        "per unit distribution", "in unit distribution",
        "byunit", "inunit",
        "by unit", "in unit",
    )),
    # Aliases: Unassigned, NotAssigned, NoUnit
    ("Unassigned Items", "UnassignedItems", (
        "not assigned to unit", "unassigned agniveers",
        "no unit assigned", "items without unit",
        "unassigned", "notassigned", "no unit",
    )),
    # Aliases: TopUnit, HighestUnit
    ("Top Unit", "TopUnit", (
        "unit with most agniveers", "top unit",
        "highest unit", "most agniveers in unit",
        "unit with highest", "highest distribution unit",
        "which unit has the highest", "which unit has most",
        "topunit", "highestunit",
    )),
]

_SKILLS_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    # Aliases: BySport, Sport, Sports
    ("By Sport", "BySport", (
        "best performers in sport", "best in sport",
        "by sport", "sport wise",
        "roster by sport", "skills by sport",
        "bysport", "sport", "sports",
    )),
    # Aliases: ByClass, Class
    ("By Class", "ByClass", (
        "agniveers by class", "by class",
        "class wise", "roster by class",
        "skills by class", "byclass",
        "class",
    )),
    # Aliases: BloodGroup, Blood
    ("Blood Group", "BloodGroup", (
        "blood group statistics", "blood group distribution",
        "blood group", "blood type",
        "bloodgroup", "blood",
    )),
]


# =============================================================================
# MODULE REGISTRY
# Maps category → (trigger_keywords, intent_list)
# =============================================================================

_MODULES: Dict[str, Tuple[Tuple[str, ...], List[Tuple[str, str, Tuple[str, ...]]]]] = {
    "Performance": (
        (
            "performance", "score", "marks", "exam", "test", "result",
            "grade", "pass", "fail", "attempt", "section", "grading",
            "bept", "ppt", "firing", "drill",
            "top", "bottom", "average", "improvement", "decline", "drop",
            "overall", "compare", "comparison",
            "agniveer", "agniveers", "performer", "performers",
            "who scored", "who passed", "who failed", "highest score",
            "lowest score", "best score", "worst score",
        ),
        _PERFORMANCE_INTENTS,
    ),
    "Leave": (
        (
            "leave", "absent", "absconded", "awol", "off duty",
            "sick leave", "annual leave", "medical leave", "on leave",
            "leave taken", "leave status", "who is on leave",
            "currently on leave", "on leave today",
            "agniveer", "agniveers",
        ),
        _LEAVE_INTENTS,
    ),
    "Medical": (
        (
            "medical", "bmi", "disease", "health", "hospital",
            "patient", "diagnosis", "diagnoses", "fitness", "weight",
            "body mass", "ward", "admitted", "ailment", "illness",
            "agniveer", "agniveers",
        ),
        _MEDICAL_INTENTS,
    ),
    "Attendance": (
        (
            "attendance", "present", "campus", "strength",
            "headcount", "muster", "monthly attendance",
            "on campus", "agniveer", "agniveers",
        ),
        _ATTENDANCE_INTENTS,
    ),
    "Verification": (
        (
            "verification", "verify", "verified",
            "pending verification", "document verification",
            "not verified", "awaiting verification",
            "no response",
        ),
        _VERIFICATION_INTENTS,
    ),
    "Equipment": (
        (
            "equipment", "gear", "overdue", "weapon", "kit",
            "issued", "inventory", "poor condition", "damaged",
            "returned equipment",
        ),
        _EQUIPMENT_INTENTS,
    ),
    "Distribution": (
        (
            "distribution", "distributed", "issued to",
            "unit distribution", "item distribution",
            "unassigned", "latest distribution",
            "agniveers in unit", "agniveer", "agniveers",
        ),
        _DISTRIBUTION_INTENTS,
    ),
    "Skills": (
        (
            "skill", "sport", "sports", "roster",
            "class skill", "sports skill",
            "blood group", "blood",
            "by sport", "by class",
            "agniveer", "agniveers",
        ),
        _SKILLS_INTENTS,
    ),
}


# =============================================================================
# ADMIN-SPECIFIC QUERY NORMALISATION
# =============================================================================

# Fuzzy vocabulary for the admin domain — fixes common misspellings and
# case variants before the classifier sees the query.
# Keys are always lowercase.  Values are the canonical lowercase forms
# used in the keyword lists above.  Section tokens are later restored to
# their exact casing by _ADMIN_CANONICAL_CASE.
ADMIN_FUZZY_VOCAB: Dict[str, str] = {
    # ── Module names ───────────────────────────────────────────────────
    "performace":    "performance",
    "performence":   "performance",
    "prefomance":    "performance",
    "preformance":   "performance",
    "performnce":    "performance",
    "performanc":    "performance",
    "attendence":    "attendance",
    "attendnce":     "attendance",
    "attendanc":     "attendance",
    "atendance":     "attendance",
    "attandance":    "attendance",
    "verfication":   "verification",
    "verifcation":   "verification",
    "verificaton":   "verification",
    "varification":  "verification",
    "veriification": "verification",
    "distribtion":   "distribution",
    "distributon":   "distribution",
    "distibution":   "distribution",
    "distribusion":  "distribution",
    "equipement":    "equipment",
    "equiptment":    "equipment",
    "equipmnt":      "equipment",
    "equpment":      "equipment",
    "meical":        "medical",
    "medicl":        "medical",
    "medcal":        "medical",
    "meddical":      "medical",
    "medicall":      "medical",
    # ── Section names ─────────────────────────────────────────────────
    # These are corrected to lowercase here; _ADMIN_CANONICAL_CASE then
    # restores the exact casing expected by _extract_section().
    "beptt":         "bept",
    "bpet":          "bept",
    "betp":          "bept",
    "pptt":          "ppt",
    "fiiring":       "firing",
    "firng":         "firing",
    "fring":         "firing",
    "fireing":       "firing",
    "drll":          "drill",
    "dril":          "drill",
    "drile":         "drill",
    # ── Operation / result words ───────────────────────────────────────
    "performrs":     "performers",
    "preformers":    "performers",
    "perfomers":     "performers",
    "performas":     "performers",
    "bottm":         "bottom",
    "lowst":         "lowest",
    "loest":         "lowest",
    "hihest":        "highest",
    "higest":        "highest",
    "higgest":       "highest",
    "avrage":        "average",
    "averge":        "average",
    "averag":        "average",
    "avrg":          "average",
    "improvment":    "improvement",
    "improvemnt":    "improvement",
    "imporvement":   "improvement",
    "percentge":     "percentage",
    "percntage":     "percentage",
    "percenage":     "percentage",
    "gradig":        "grading",
    "gradng":        "grading",
    "gradeing":      "grading",
    "sumary":        "summary",
    "summry":        "summary",
    "sumarry":       "summary",
    "comparson":     "comparison",
    "comparsion":    "comparison",
    "comparision":   "comparison",
    "attmpt":        "attempt",
    "attepmpt":      "attempt",
    "attemp":        "attempt",
    "atempt":        "attempt",
    # ── Leave ─────────────────────────────────────────────────────────
    "leeve":         "leave",
    "leve":          "leave",
    "leav":          "leave",
    "abscnded":      "absconded",
    "absconed":      "absconded",
    "absconede":     "absconded",
    "abscondd":      "absconded",
    # ── Attendance ────────────────────────────────────────────────────
    "presnt":        "present",
    "preent":        "present",
    "presentt":      "present",
    "campas":        "campus",
    "campuss":       "campus",
    "strenght":      "strength",
    "strengh":       "strength",
    "stength":       "strength",
    "montly":        "monthly",
    "monthyl":       "monthly",
    "monthl":        "monthly",
    # ── Equipment ─────────────────────────────────────────────────────
    "overdeu":       "overdue",
    "overdu":        "overdue",
    "overdued":      "overdue",
    "condtion":      "condition",
    "condiiton":     "condition",
    "conditon":      "condition",
    # ── Skills / roster ───────────────────────────────────────────────
    "blod":          "blood",
    "bloog":         "blood",
    "sportt":        "sport",
    "sprot":         "sport",
    "classs":        "class",
    "claas":         "class",
    "rosster":       "roster",
    "rostr":         "roster",
    # ── Common query words ────────────────────────────────────────────
    "shoow":         "show",
    "shw":           "show",
    "lst":           "list",
    "listt":         "list",
    "gve":           "give",
    "givee":         "give",
    "whch":          "which",
    "whcih":         "which",
    "waht":          "what",
    "hwo":           "how",
    "mny":           "many",
    "mant":          "many",
    "mannay":        "many",
    "todya":         "today",
    "todday":        "today",
    "toady":         "today",
    "persnnel":      "personnel",
    "personel":      "personnel",
    "personnnel":    "personnel",
    "persnonel":     "personnel",
    "traineee":      "trainee",
    "tainee":        "trainee",
    "agniverr":      "agniveer",
    "agniver":       "agniveer",
    # ── Grading ───────────────────────────────────────────────────────
    "excelent":      "excellent",
    "excellnt":      "excellent",
    "excellentt":    "excellent",
    "exeptional":    "exceptional",
    "exceptonal":    "exceptional",
    "satifactory":   "satisfactory",
    # ── Unit / distribution ───────────────────────────────────────────
    "unassignd":     "unassigned",
    "unasigned":     "unassigned",
    "unassiged":     "unassigned",
    "distribted":    "distributed",
    "distrubted":    "distributed",
}

# Tokens whose casing must be restored after fuzzy correction so that
# _extract_section() (exact-string lookup in _SECTION_MAP) works correctly
# regardless of how the user typed the section name.
_ADMIN_CANONICAL_CASE: Dict[str, str] = {
    "bept":   "BEPT",
    "ppt":    "PPT",
    "firing": "Firing",
    "drill":  "Drill",
}


def admin_normalize_query(query: str) -> str:
    """
    Normalise an admin chat query before intent classification.

    Steps
    -----
    1. Strip leading/trailing whitespace (internal spacing is preserved
       so the raw_query stored in intent results remains human-readable).
    2. Word-by-word fuzzy correction via ADMIN_FUZZY_VOCAB so that
       misspellings ("performace", "bpet", "avrage", "attendence", …)
       and case variants ("BEPT", "Bept", "bEPT") are mapped to forms
       the keyword matchers recognise.
    3. Canonical-casing restoration for section tokens (BEPT, PPT,
       Firing, Drill) so _extract_section() matches them exactly.

    This function is intentionally separate from
    config._fuzzy_normalize_query() which targets Agniveer recruitment
    vocabulary (salary, eligibility, seva nidhi, …).  Admin vocabulary
    is completely different and the two vocabs must not interfere.

    The function does NOT lowercase the whole query — the classifier's
    own _normalise() handles that internally.  We only fix individual
    tokens so names, numbers, and unit strings arrive un-mangled.
    """
    if not query:
        return query

    words = query.split()
    out: List[str] = []

    for word in words:
        # Peel trailing punctuation so it doesn't break dict lookup,
        # then reattach it after correction.
        suffix = ""
        core = word
        while core and core[-1] in "?.,!:;":
            suffix = core[-1] + suffix
            core = core[:-1]

        if not core:
            out.append(word)
            continue

        lower_core = core.lower()

        # Step 1 — fuzzy spelling correction (keyed by lowercase)
        fixed = ADMIN_FUZZY_VOCAB.get(lower_core)
        if fixed is not None:
            core = fixed
            lower_core = fixed

        # Step 2 — canonical casing for section / special tokens
        canonical_cased = _ADMIN_CANONICAL_CASE.get(lower_core)
        if canonical_cased is not None:
            core = canonical_cased

        out.append(core + suffix)

    return " ".join(out)


# =============================================================================
# FILTER VALUE MAPS
# Exact values accepted by .NET per the API documentation.
# =============================================================================

# Section (Performance filter)
_SECTIONS = {"BEPT", "PPT", "Firing", "Drill"}
_SECTION_MAP = {
    "bept":   "BEPT",
    "ppt":    "PPT",
    "firing": "Firing",
    "drill":  "Drill",
}

# SubSection (Performance filter) — common values
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

# Grading (Performance filter)
_GRADING_MAP = {
    "exceptionally well": "ExceptionallyWell",
    "excellent":          "Excellent",
    "good":               "Good",
    "sat":                "SAT",
    "fail":               "Fail",
    "unsa":               "UNSA",
}

# LeaveType (Leave filter)
_LEAVE_TYPE_MAP = {
    "annual":    "Annual",
    "medical":   "Medical",
    "sick":      "Sick",
    "absconded": "Absconded",
}

# Sport (Skills filter)
_SPORT_MAP = {
    "cricket":    "Cricket",
    "football":   "Football",
    "running":    "Running",
    "basketball": "Basketball",
    "volleyball": "Volleyball",
    "kabaddi":    "Kabaddi",
    "hockey":     "Hockey",
}

# Class (Performance + Skills filter)
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
    match = re.search(r"\b(\d+)\b", text)
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
    # Only extract leaveType for the Leave module to avoid false positives
    # e.g. "medical cases" should NOT set leaveType=Medical in Medical module
    if category and category != "Leave":
        return None
    for phrase, code in _LEAVE_TYPE_MAP.items():
        if phrase in text_lower:
            return code
    return None


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


def _extract_unit_name(text: str) -> Optional[str]:
    # "by unit <Name>" / "in unit <Name>"
    match = re.search(
        r"\b(?:in unit|by unit|for unit|unit)\s+([A-Za-z][A-Za-z0-9]*)(\s+[Uu]nit)?\b",
        text,
        re.IGNORECASE,
    )
    if match:
        candidate = match.group(1).strip()
        if candidate.lower() not in _GENERIC_WORDS:
            return f"{candidate.title()} Unit"

    # "<Name> Unit"
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
    # Matches: "to attempt 4" OR "attempt 1 to 4" (bare number after 'to')
    match = (
        re.search(r"\bto\s*attempt\s*(\d+)\b", text_lower) or
        re.search(r"\battempt\s*\d+\s+to\s+(\d+)\b", text_lower)
    )
    return int(match.group(1)) if match else None


# =============================================================================
# SCORING & MATCHING
# =============================================================================

def _score_intent(query_lower: str, keywords: Tuple[str, ...]) -> int:
    """
    Score query against keywords.
    Longer keyword matches score higher (more specific = better match).
    """
    score = 0
    for kw in keywords:
        if kw in query_lower:
            score += len(kw.split())
    return score


def _match_module(query_lower: str) -> Optional[str]:
    """
    Score each module by its trigger keywords.
    When two modules tie (e.g. 'top' matches both Performance and Medical
    intent keywords), break the tie by also scoring the intent-level keywords
    within each module — the module whose intents match more specifically wins.
    """
    scores: Dict[str, int] = {}
    for module, (triggers, _) in _MODULES.items():
        scores[module] = _score_intent(query_lower, triggers)

    best_score = max(scores.values(), default=0)
    if best_score == 0:
        return None

    # All modules that tied at the top score
    tied = [m for m, s in scores.items() if s == best_score]
    if len(tied) == 1:
        return tied[0]

    # Tiebreak: score each tied module's intent keywords directly
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
    Classify an admin's natural-language question.

    Returns internal Python dict (snake_case).
    Call format_admin_payload() to convert to .NET camelCase payload.
    """
    raw_query = (query or "").strip()
    q         = _normalise(raw_query)

    result: Dict[str, Any] = {
        "category":    None,
        "subcategory": None,
        "number":      None,
        "section":     None,
        "sub_section": None,
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

    # Fallback: scan all intent keywords across all modules
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
    _, intent_list     = _MODULES[module]

    # ── Intent detection ───────────────────────────────────────────────────
    intent_match = _match_intent(q, intent_list)
    if intent_match:
        _, intent_code        = intent_match
        result["subcategory"] = intent_code
        result["confidence"]  = "high"
    else:
        result["subcategory"] = intent_list[0][1]   # default to first
        result["confidence"]  = "medium"

    # ── Filter extraction ──────────────────────────────────────────────────
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

    # Downgrade confidence if top/bottom but no count given
    if result["confidence"] == "high":
        if result["subcategory"] in ("TopPerformers", "LowestPerformers"):
            if result["number"] is None:
                result["confidence"] = "medium"

    return result


def format_admin_payload(intent_result: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert the internal intent dict to the exact .NET AiCommand payload.

    Python key      → .NET key
    ─────────────────────────────
    subcategory     → operation   (via _SUBCATEGORY_TO_OPERATION)
    number          → n
    sub_section     → subSection
    leave_type      → leaveType
    unit_name       → unitName
    attempt_no      → attemptNo
    from_attempt    → fromAttempt
    to_attempt      → toAttempt

    None values and internal-only fields are stripped.
    """
    payload: Dict[str, Any] = {}

    # category — exact string
    if intent_result.get("category"):
        payload["category"] = intent_result["category"]

    # subcategory → operation
    subcategory = intent_result.get("subcategory")
    if subcategory:
        payload["operation"] = _SUBCATEGORY_TO_OPERATION.get(subcategory, subcategory)

    # n
    if intent_result.get("number") is not None:
        payload["n"] = intent_result["number"]

    # section
    if intent_result.get("section"):
        payload["section"] = intent_result["section"]

    # subSection
    if intent_result.get("sub_section"):
        payload["subSection"] = intent_result["sub_section"]

    # grading
    if intent_result.get("grading"):
        payload["grading"] = intent_result["grading"]

    # leaveType
    if intent_result.get("leave_type"):
        payload["leaveType"] = intent_result["leave_type"]

    # sport
    if intent_result.get("sport"):
        payload["sport"] = intent_result["sport"]

    # class
    if intent_result.get("class"):
        payload["class"] = intent_result["class"]

    # unitName
    if intent_result.get("unit_name"):
        payload["unitName"] = intent_result["unit_name"]

    # attemptNo
    if intent_result.get("attempt_no") is not None:
        payload["attemptNo"] = intent_result["attempt_no"]

    # fromAttempt
    if intent_result.get("from_attempt") is not None:
        payload["fromAttempt"] = intent_result["from_attempt"]

    # toAttempt
    if intent_result.get("to_attempt") is not None:
        payload["toAttempt"] = intent_result["to_attempt"]

    return payload