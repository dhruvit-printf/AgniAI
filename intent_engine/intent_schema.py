"""
intent_schema.py
================

SINGLE SOURCE OF TRUTH for all AgniAI intent formation business rules.

This module contains all domain-specific data structures, enumerations, and
mappings that were previously hardcoded across admin_intent.py. By centralizing
this configuration, we ensure:

1. Deterministic intent formation
2. Easy maintenance and updates
3. Consistency across the pipeline
4. Version-controlled schema changes

All modules that need business data MUST import from here, NOT hardcode lists.
"""

from __future__ import annotations

from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

# =============================================================================
# CATEGORIES (OFFICIAL DOMAINS)
# =============================================================================

OFFICIAL_CATEGORIES: FrozenSet[str] = frozenset(
    [
        "Performance",
        "Leave",
        "Medical",
        "Attendance",
        "Verification",
        "Equipment",
        "Distribution",
        "Skills",
        "Roster",
        "Strength",
        "Overall",
        "Schedule",
    ]
)

# =============================================================================
# OPERATIONS (BY CATEGORY)
# =============================================================================

OPERATIONS_BY_CATEGORY: Dict[str, FrozenSet[str]] = {
    "Performance": frozenset(
        [
            "Top",
            "Bottom",
            "Improvement",
            "Drop",
            "Grading",
            "GradingSummary",
            "Average",
            "AttemptWise",
            "BestAttempt",
            "Summary",
            "PassPercentage",
            "FailPercentage",
            "Overall",
        ]
    ),
    "Leave": frozenset(
        [
            "Most",
            "Least",
            "Current",
            "Absconded",
        ]
    ),
    "Medical": frozenset(
        [
            "Active",
            "BMI",
            "BloodGroup",
            "Disease",
            "Individual",
        ]
    ),
    "Attendance": frozenset(
        [
            "Monthly",
            "Weekly",
            "Daily",
            "Yearly",
            "Present",
            "Summary",
        ]
    ),
    "Strength": frozenset(
        [
            "Summary",
        ]
    ),
    "Verification": frozenset(
        [
            "Pending",
            "Sent",
            "NotResponded",
            "Verified",
            "Completed",
            "Rejected",
        ]
    ),
    "Equipment": frozenset(
        [
            "Stats",
            "Issued",
            "Procured",
            "Overdue",
            "Returned",
            "Holding",
            "AgniveerWise",
        ]
    ),
    "Distribution": frozenset(
        [
            "Latest",
            "ByUnit",
            "Unassigned",
            "TopUnit",
        ]
    ),
    "Skills": frozenset(
        [
            "BySport",
            "ByClass",
        ]
    ),
    "Roster": frozenset(
        [
            "BySport",
            "ByClass",
        ]
    ),
    "Overall": frozenset(
        [
            "OverallPerformance",
        ]
    ),
    "Schedule": frozenset(
        [
            "Company",
            "Agniveer",
            "Date",
        ]
    ),
}

# =============================================================================
# RESPONSE TYPES
# =============================================================================

RESPONSE_TYPES: FrozenSet[str] = frozenset(
    [
        "Summary",
        "Detailed",
    ]
)

RESPONSE_TYPE_DEFAULT: str = "Summary"

# Keywords that trigger Detailed response type
DETAILED_KEYWORDS: FrozenSet[str] = frozenset(
    [
        "detail",
        "detailed",
        "complete",
        "full report",
        "complete report",
        "comprehensive",
        "everything",
        "elaborate",
        "deep analysis",
    ]
)

# =============================================================================
# CATEGORY / OPERATION KEYWORDS
# =============================================================================

CATEGORY_KEYWORDS: Dict[str, Tuple[str, ...]] = {
    "Performance": (
        "performance",
        "top performer",
        "top performers",
        "highest performer",
        "best performer",
        "bottom performer",
        "lowest performer",
        "grading",
        "average score",
        "attempt",
        "improvement",
        "drop",
        "comparison",
        "section summary",
        "overall",
        "pass rate",
        "fail rate",
        "pass percentage",
        "fail percentage",
    ),
    "Leave": (
        "leave",
        "absent",
        "absconded",
        "on leave",
        "days off",
        "off days",
        "leave taken",
        "leave status",
    ),
    "Medical": (
        "medical",
        "health",
        "bmi",
        "blood",
        "weight",
        "fitness",
        "disease",
        "active cases",
        "admitted",
        "hospital",
        "sick",
        "fever",
        "injury",
    ),
    "Attendance": (
        "attendance",
        "present",
        "campus",
        "muster",
        "strength",
        "headcount",
        "monthly",
        "weekly",
        "daily",
        "yearly",
    ),
    "Verification": (
        "verification",
        "verify",
        "verified",
        "pending",
        "rejected",
        "approved",
        "cleared",
    ),
    "Equipment": (
        "equipment",
        "issued",
        "procured",
        "overdue",
        "holding",
        "condition",
        "gear",
        "kit",
        "inventory",
    ),
    "Distribution": (
        "distribution",
        "unit",
        "assigned",
        "agniveer assignment",
    ),
    "Roster": (
        "roster",
        "rosters",
        "sport roster",
        "class roster",
        "sports roster",
    ),
    "Strength": (
        "strength",
        "headcount",
        "breakdown",
        "strength breakdown",
    ),
    "Skills": (
        "skills",
        "sport",
        "cricket",
        "football",
        "class",
        "community",
    ),
    "Schedule": (
        "schedule",
        "training",
        "agenda",
    ),
    "Overall": (
        "overall",
        "composite",
        "all criteria",
    ),
}

OPERATION_SYNONYMS: Dict[str, Dict[str, Tuple[str, ...]]] = {
    "Performance": {
        "Top": (
            "top",
            "highest",
            "best",
            "top performer",
            "top performers",
            "highest performer",
            "best scorer",
            "top scorer",
            "topper",
            "toppers",
            "topped",
            "who topped",
            "who scored highest",
            "leading performer",
            "outstanding performer",
            "ace performer",
            "stellar performer",
            "high achiever",
        ),
        "Bottom": (
            "bottom",
            "lowest",
            "worst",
            "bottom performer",
            "lowest performer",
            "worst performer",
            "poor performer",
            "weakest performer",
            "trailing performer",
            "struggling performer",
            "underperformer",
            "low scorer",
            "who scored lowest",
            "who performed worst",
        ),
        "Improvement": (
            "improvement",
            "improve",
            "improved",
            "score went up",
            "positive trend",
            "biggest improvement",
            "most improved",
            "rising score",
            "upward trend",
            "who improved",
            "who got better",
            "score gain",
            "bounced back",
            "getting better",
            "progressing",
        ),
        "Drop": (
            "drop",
            "decline",
            "declined",
            "score drop",
            "downward trend",
            "negative trend",
            "who dropped",
            "who declined",
            "who got worse",
            "fell",
            "slipped",
            "performance degraded",
            "regression",
            "regressed",
            "deterioration",
            "who is struggling more",
        ),
        "Grading": (
            "grading",
            "grade",
            "by grade",
            "grade wise",
            "who got excellent",
            "who got good",
            "who got sat",
            "categorised by grade",
        ),
        "GradingSummary": (
            "grading summary",
            "grade summary",
            "distribution of grades",
            "how many in each grade",
            "grade count",
            "grade statistics",
        ),
        "Average": (
            "average",
            "avg",
            "mean",
            "average score",
            "average marks",
            "what is the average",
            "overall average",
        ),
        "AttemptWise": (
            "attemptwise",
            "attempt wise",
            "by attempt",
            "attempt breakdown",
            "score per attempt",
            "each attempt",
            "all attempts",
            "multiple attempts",
        ),
        "BestAttempt": (
            "best attempt",
            "best score attempt",
            "highest attempt",
            "peak attempt",
            "personal best",
        ),
        "Summary": (
            "summary",
            "overview",
            "section summary",
            "section overview",
            "snapshot",
            "brief",
            "report",
        ),
        "PassPercentage": (
            "pass percentage",
            "pass rate",
            "passing percentage",
            "how many passed",
            "who passed",
            "clearance rate",
        ),
        "FailPercentage": (
            "fail percentage",
            "fail rate",
            "failure rate",
            "how many failed",
            "who failed",
        ),
        "Overall": (
            "overall",
            "composite",
            "all criteria",
            "all section",
            "combined performance",
            "aggregate",
        ),
    },
    "Leave": {
        "Most": (
            "most",
            "highest",
            "maximum",
            "most leave taken",
            "who has the most leave",
            "who took maximum leave",
            "most absent",
            "maximum absentee",
            "leave takers",
            "top leave takers",
            "most leaves",
            "maximum leaves",
        ),
        "Least": (
            "least",
            "lowest",
            "minimum",
            "least leave taken",
            "who has the least leave",
            "fewest days absent",
            "fewest leaves",
            "most regular",
            "most punctual",
        ),
        "Current": (
            "current",
            "today",
            "now",
            "currently on leave",
            "who is on leave",
            "leave status",
            "leave today",
            "absent today",
            "currently absent",
        ),
        "Absconded": (
            "absconded",
            "abscond",
            "gone missing",
            "missing person",
            "awol",
            "whereabouts unknown",
        ),
    },
    "Medical": {
        "Active": (
            "active",
            "cases",
            "admitted",
            "active medical cases",
            "current patients",
            "in hospital",
            "under treatment",
            "how many sick",
            "who is admitted",
        ),
        "BMI": (
            "bmi",
            "weight",
            "fitness",
            "body mass index",
            "who is obese",
            "overweight",
            "underweight",
        ),
        "BloodGroup": (
            "blood",
            "blood group",
            "blood type",
            "blood group distribution",
            "who has which blood group",
        ),
        "Disease": (
            "disease",
            "diseases",
            "diagnoses",
            "top diseases",
            "common disease",
            "most common disease",
        ),
        "Individual": (
            "individual medical",
            "medical status of",
            "medical details of",
            "particular medical",
            "specific medical",
        ),
    },
    "Attendance": {
        "Monthly": (
            "monthly",
            "month",
            "monthly attendance",
            "attendance this month",
            "month wise",
            "attendance by month",
        ),
        "Weekly": (
            "weekly",
            "week",
            "weekly attendance",
            "this week",
            "week wise",
            "attendance by week",
        ),
        "Daily": (
            "daily",
            "day",
            "daily attendance",
            "day wise",
        ),
        "Yearly": (
            "yearly",
            "year",
            "yearly attendance",
            "annual",
        ),
        "Present": (
            "present",
            "present today",
            "on campus",
            "who is present",
            "how many present",
            "attendance today",
            "today attendance",
        ),
        "Summary": (
            "attendance summary",
            "attendance overview",
            "summary of attendance",
        ),
    },
    "Equipment": {
        "Stats": (
            "stats",
            "summary",
            "overview",
            "equipment stats",
            "equipment summary",
            "equipment inventory",
        ),
        "Issued": (
            "issued",
            "issue",
            "issued items",
            "items issued",
            "kit issued",
            "what was issued",
        ),
        "Procured": (
            "procured",
            "procure",
            "procured items",
            "items procured",
            "self purchased",
            "purchased",
        ),
        "Overdue": (
            "overdue",
            "late",
            "not returned",
            "equipment not returned",
            "pending equipment return",
        ),
        "Returned": (
            "returned",
            "return",
            "poor condition",
            "damaged",
            "bad condition",
            "broken",
        ),
        "Holding": (
            "holding",
            "holds",
            "who is holding",
            "equipment holding",
        ),
        "AgniveerWise": (
            "agniveer wise",
            "by agniveer",
            "equipment by agniveer",
        ),
    },
    "Distribution": {
        "Latest": (
            "latest",
            "recent",
            "last",
            "newest",
            "most recent",
            "current",
            "fresh",
        ),
        "ByUnit": (
            "by unit",
            "unit wise",
            "in unit",
            "agniveers in unit",
            "unit distribution",
        ),
        "Unassigned": (
            "unassigned",
            "not assigned",
            "no unit",
            "without unit",
        ),
        "TopUnit": (
            "top unit",
            "highest unit",
            "unit with most",
            "biggest unit",
            "unit with the most agniveers",
            "received most agniveers",
            "most agniveers",
            "unit received most",
            "largest unit",
            "highest distribution",
            "most distribution",
        ),
    },
    "Roster": {
        "BySport": (
            "by sport",
            "sport wise",
            "cricket players",
            "football players",
            "who plays",
            "sports roster",
        ),
        "ByClass": (
            "by class",
            "class wise",
            "sikh class",
            "dogra class",
            "class wise roster",
        ),
    },
    "Skills": {
        "BySport": (
            "by sport",
            "sport wise",
            "cricket players",
            "football players",
            "who plays",
            "sports roster",
        ),
        "ByClass": (
            "by class",
            "class wise",
            "sikh class",
            "dogra class",
            "class wise roster",
        ),
    },
    "Verification": {
        "Pending": (
            "pending",
            "awaiting",
            "not verified",
            "verification pending",
        ),
        "Sent": (
            "sent",
            "sent verification",
            "verification sent",
        ),
        "NotResponded": (
            "not responded",
            "no response",
            "awaiting response",
        ),
        "Completed": (
            "completed",
            "done",
            "verification completed",
            "all clear",
        ),
        "Verified": (
            "verified",
            "all verified",
        ),
        "Rejected": (
            "rejected",
            "not approved",
            "verification failed",
        ),
    },
    "Strength": {
        "Summary": (
            "strength",
            "strength breakdown",
            "headcount",
            "headcount breakdown",
            "strength summary",
        ),
    },
    "Schedule": {
        "Company": (
            "company schedule",
            "schedule by company",
            "company wise schedule",
        ),
        "Agniveer": (
            "agniveer schedule",
            "schedule by agniveer",
            "agniveer wise schedule",
        ),
        "Date": (
            "date schedule",
            "schedule by date",
            "date wise schedule",
        ),
    },
    "Overall": {
        "OverallPerformance": (
            "overall performance",
            "overall report",
            "composite report",
            "all criteria",
            "all criteria performance",
        ),
    },
}

# =============================================================================
# SECTIONS (PERFORMANCE DOMAIN ONLY)
# =============================================================================

SECTION: Dict[str, Dict[str, Any]] = {
    "BPET": {
        "name": "BPET",
        "aliases": ("bpet", "betp", "bept"),
        "isExceptional": False,
    },
    "PPT": {
        "name": "PPT",
        "aliases": ("ppt", "pptt"),
        "isExceptional": False,
    },
    "Firing": {
        "name": "Firing",
        "aliases": ("firing", "fiiring", "firng", "fring", "fireing"),
        "isExceptional": False,
    },
    "Drill": {
        "name": "Drill",
        "aliases": ("drill", "drll", "dril"),
        "isExceptional": False,
    },
    "Map Reading": {
        "name": "Map Reading",
        "aliases": ("map reading", "mapreading"),
        "isExceptional": True,
    },
    "Tactics": {
        "name": "Tactics",
        "aliases": ("tactics",),
        "isExceptional": True,
    },
    "Exceptional Sport": {
        "name": "Exceptional Sport",
        "aliases": ("exceptional sport", "exceptionalSport", "exceptional_sport"),
        "isExceptional": True,
    },
    "BFC": {
        "name": "BFC",
        "aliases": ("bfc",),
        "isExceptional": True,
    },
    "PDP": {
        "name": "PDP",
        "aliases": ("pdp",),
        "isExceptional": True,
    },
    "MR": {
        "name": "MR",
        "aliases": ("mr",),
        "isExceptional": True,
    },
    "IT": {
        "name": "IT",
        "aliases": ("it",),
        "isExceptional": True,
    },
    "FC/BC": {
        "name": "FC/BC",
        "aliases": ("fc/bc", "fcbc", "fc bc"),
        "isExceptional": True,
    },
    "CBRN": {
        "name": "CBRN",
        "aliases": ("cbrn",),
        "isExceptional": True,
    },
    "Test Written (AMT)": {
        "name": "Test Written (AMT)",
        "aliases": ("test written", "test written amt"),
        "isExceptional": True,
    },
    "Test Prac (AMT)": {
        "name": "Test Prac (AMT)",
        "aliases": ("test prac", "test prac amt"),
        "isExceptional": True,
    },
}

# =============================================================================
# SUBSECTIONS (BY SECTION)
# =============================================================================

SUBSECTIONS_BY_SECTION: Dict[str, List[str]] = {
    "BPET": ["5km", "60m Sprint", "9' Ditch", "H Rope", "V Rope"],
    "PPT": ["2.4km", "100m Sprint", "Chin Ups", "Sit Ups", "Toe Touch", "5m Shuttle"],
    "Firing": [
        "300m 5 FTS",
        "200m 5 FTS",
        "100m LU (R)",
        "100m LU (5/5)",
        "50m SU TRU",
        "100m NI SFTS SS",
        "15m (Bal) BC",
    ],
    "Drill": [
        "Turn Out and Bearing",
        "Word of Command",
        "Marching",
        "Slow March",
        "Saluting Salami Shastra",
        "Turning",
    ],
}

# =============================================================================
# GRADING CATEGORIES
# =============================================================================

GRADING_CATEGORIES: Dict[str, str] = {
    "exceptionally well": "ExceptionallyWell",
    "excellent": "Excellent",
    "good": "Good",
    "sat": "SAT",
    "satisfactory": "SAT",
    "fail": "Fail",
    "unsa": "UNSA",
    "unsatisfactory": "UNSA",
}

# =============================================================================
# LEAVE TYPES
# =============================================================================

LEAVE_TYPES: Dict[str, str] = {
    "annual": "Annual",
    "medical": "Medical",
    "sick": "Sick",
    "absconded": "Absconded",
    "attnc": "ATTNC",
    "attenc": "ATTNC",
    "attn c": "ATTNC",
    "att nc": "ATTNC",
    "ex ppg": "ExPPG",
    "exppg": "ExPPG",
    "ex-ppg": "ExPPG",
    "on leave": "Current",
    "currently on leave": "Current",
    "leave today": "Current",
    "current leave": "Current",
    "absent today": "Current",
    "currently absent": "Current",
    "leave status": "Current",
}

# =============================================================================
# BMI CATEGORIES
# =============================================================================

BMI_CATEGORIES: Dict[str, str] = {
    "underweight": "Underweight",
    "normal": "Normal",
    "overweight": "Overweight",
    "obese": "Obese",
}

# =============================================================================
# BLOOD GROUPS
# =============================================================================

BLOOD_GROUPS: FrozenSet[str] = frozenset(
    [
        "A+",
        "A-",
        "B+",
        "B-",
        "O+",
        "O-",
        "AB+",
        "AB-",
    ]
)

# =============================================================================
# SPORTS
# =============================================================================

SPORTS: Dict[str, str] = {
    "cricket": "Cricket",
    "football": "Football",
    "running": "Running",
    "basketball": "Basketball",
    "volleyball": "Volleyball",
    "kabaddi": "Kabaddi",
    "hockey": "Hockey",
}

# =============================================================================
# CLASSES
# =============================================================================

CLASSES: Dict[str, str] = {
    "sikh": "Sikh",
    "oic": "OIC",
    "gurkha": "Gurkha",
    "gorkha": "Gurkha",
    "dogra": "Dogra",
    "jat": "Jat",
    "rajput": "Rajput",
    "punjabi": "Punjabi",
}

# =============================================================================
# EQUIPMENT ITEMS
# =============================================================================

ISSUED_EQUIPMENT_ITEMS: List[str] = [
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

PROCURED_EQUIPMENT_ITEMS: List[str] = [
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

# =============================================================================
# VERIFICATION STATUSES
# =============================================================================

VERIFICATION_STATUSES: Dict[str, str] = {
    "pending": "Pending",
    "sent": "Sent",
    "not responded": "NotResponded",
    "verified": "Verified",
    "rejected": "Rejected",
}

# =============================================================================
# CATEGORY-ENTITY COMPATIBILITY
# =============================================================================

# Which entity types are ALLOWED for each category
CATEGORY_ENTITY_COMPATIBILITY: Dict[str, Set[str]] = {
    "Performance": {
        "section",
        "subSection",
        "grading",
        "attemptNo",
        "fromAttempt",
        "toAttempt",
        "sport",
        "class",
    },
    "Leave": {"leaveType", "sport", "class"},
    "Medical": {"bmiCategory", "bloodGroup", "medicalStatus", "sport", "class"},
    "Attendance": {"date", "fromDate", "toDate", "sport", "class"},
    "Verification": {"sport", "class"},
    "Equipment": {"equipmentName", "sport", "class"},
    "Distribution": {"sport", "class"},
    "Skills": {"sport", "class", "section", "subSection"},
    "Roster": {"sport", "class", "section", "subSection"},
    "Strength": {"sport", "class"},
    "Overall": {"sport", "class"},
    "Schedule": {"date", "sport", "class"},
}

# Common entities allowed in ALL categories
COMMON_ENTITY_TYPES: Set[str] = {
    "companyId",
    "platoonId",
    "batchId",
    "agniveerNo",
    "unitName",
    "n",
    "date",
    "fromDate",
    "toDate",
}

CATEGORY_ENTITY_HINTS: Dict[str, Tuple[str, ...]] = {
    "Performance": ("section", "grading", "attempt", "score", "marks"),
    "Leave": ("leave", "absent", "abscond", "leaveType"),
    "Medical": ("medical", "hospital", "bmi", "blood"),
    "Attendance": ("attendance", "present", "campus", "date"),
    "Verification": ("verification", "verified", "pending"),
    "Equipment": ("equipment", "gear", "inventory", "overdue"),
    "Distribution": ("distribution", "unit", "unassigned"),
    "Roster": ("roster", "sport", "class", "who plays"),
    "Skills": ("skills", "sport", "class"),
    "Strength": ("strength", "headcount", "breakdown"),
    "Overall": ("overall", "composite", "all criteria"),
    "Schedule": ("schedule", "company", "agniveer", "date"),
}

# Named unit aliases that should normalize to canonical unit labels.
UNIT_ALIASES: Dict[str, str] = {
    "alpha": "Alpha Unit",
    "bravo": "Bravo Unit",
    "charlie": "Charlie Unit",
    "delta": "Delta Unit",
    "echo": "Echo Unit",
    "foxtrot": "Foxtrot Unit",
    "golf": "Golf Unit",
    "hotel": "Hotel Unit",
    "india": "India Unit",
    "juliet": "Juliet Unit",
    "kilo": "Kilo Unit",
    "lima": "Lima Unit",
    "mike": "Mike Unit",
    "november": "November Unit",
    "oscar": "Oscar Unit",
    "papa": "Papa Unit",
    "quebec": "Quebec Unit",
    "romeo": "Romeo Unit",
    "sierra": "Sierra Unit",
    "tango": "Tango Unit",
    "uniform": "Uniform Unit",
    "victor": "Victor Unit",
    "whiskey": "Whiskey Unit",
    "xray": "Xray Unit",
    "yankee": "Yankee Unit",
    "zulu": "Zulu Unit",
}

# Relative date phrases returned by the extractor in canonical form.
RELATIVE_DATE_PHRASES: Dict[str, str] = {
    "today": "today",
    "yesterday": "yesterday",
    "tomorrow": "tomorrow",
    "this week": "this week",
    "last week": "last week",
    "this month": "current month",
    "current month": "current month",
    "last month": "last month",
    "this year": "this year",
    "current year": "this year",
}

# Ranking cue words used to distinguish counts from IDs.
RANKING_CONTEXT_PHRASES: Tuple[str, ...] = (
    "top",
    "bottom",
    "highest",
    "lowest",
    "best",
    "worst",
    "rank",
    "topped",
    "scored highest",
    "performed worst",
    "most",
    "least",
)

NEGATIVE_REPORT_PHRASES: List[str] = [
    "no matching data",
    "empty dataset",
    "no records available",
    "zero active",
    "no future trend projection",
    "insufficient data",
    "could not be completed",
    "returned zero",
    "returned 0 records",
]


# =============================================================================
# FUZZY VOCABULARY (MISSPELLINGS)
# =============================================================================

FUZZY_VOCAB: Dict[str, str] = {
    "performace": "performance",
    "performence": "performance",
    "prefomance": "performance",
    "preformance": "performance",
    "performnce": "performance",
    "attendence": "attendance",
    "attendnce": "attendance",
    "atendance": "attendance",
    "attandance": "attendance",
    "verfication": "verification",
    "verifcation": "verification",
    "verificaton": "verification",
    "varification": "verification",
    "distribtion": "distribution",
    "distributon": "distribution",
    "distibution": "distribution",
    "equipement": "equipment",
    "equiptment": "equipment",
    "equipmnt": "equipment",
    "equpment": "equipment",
    "meical": "medical",
    "medicl": "medical",
    "medcal": "medical",
    "bpet": "bpet",
    "bept": "bpet",
    "betp": "bpet",
    "pptt": "ppt",
    "fiiring": "firing",
    "firng": "firing",
    "fring": "firing",
    "fireing": "firing",
    "drll": "drill",
    "dril": "drill",
    "performrs": "performers",
    "preformers": "performers",
    "perfomers": "performers",
    "bottm": "bottom",
    "lowst": "lowest",
    "loest": "lowest",
    "hihest": "highest",
    "higest": "highest",
    "avrage": "average",
    "averge": "average",
    "avrg": "average",
    "improvment": "improvement",
    "improvemnt": "improvement",
    "improv": "improvement",
    "percentge": "percentage",
    "percntage": "percentage",
    "gradig": "grading",
    "gradng": "grading",
    "gradeing": "grading",
    "sumary": "summary",
    "summry": "summary",
    "comparson": "comparison",
    "comparsion": "comparison",
    "attmpt": "attempt",
    "attepmpt": "attempt",
    "atempt": "attempt",
    "attemptwise": "attempt wise",
    "bestattempt": "best attempt",
    "sectionsummary": "section summary",
    "gradingsummary": "grading summary",
    "gradesummary": "grading summary",
    "passrate": "pass rate",
    "failrate": "fail rate",
    "passpercentage": "pass percentage",
    "failpercentage": "fail percentage",
    "leeve": "leave",
    "leve": "leave",
    "abscnded": "absconded",
    "absconed": "absconded",
    "presnt": "present",
    "preent": "present",
    "campas": "campus",
    "strenght": "strength",
    "strengh": "strength",
    "montly": "monthly",
    "monthyl": "monthly",
    "overdeu": "overdue",
    "overdu": "overdue",
    "condtion": "condition",
    "conditon": "condition",
    "blod": "blood",
    "bloog": "blood",
    "sportt": "sport",
    "sprot": "sport",
    "classs": "class",
    "claas": "class",
    "todya": "today",
    "todday": "today",
    "persnnel": "person",
    "personel": "person",
    "agniverr": "agniveer",
    "agniver": "agniveer",
    "excelent": "excellent",
    "excellnt": "excellent",
    "satifactory": "satisfactory",
    "unassignd": "unassigned",
    "unasigned": "unassigned",
}

# =============================================================================
# INTENT TYPE DEFAULTS (VISUALIZATION SUGGESTION - DEPRECATED)
# =============================================================================

# NOTE: These are VISUALIZATION DEFAULTS, which are DEPRECATED per the spec.
# Visualization should be determined AFTER receiving .NET response, not before.
# This is kept for backward compatibility only.

INTENT_TYPE_DEFAULTS: Dict[Tuple[str, str], str] = {
    ("Performance", "Top"): "Tabular",
    ("Performance", "Bottom"): "Tabular",
    ("Performance", "Improvement"): "Trend Chart",
    ("Performance", "Drop"): "Trend Chart",
    ("Performance", "Grading"): "Tabular",
    ("Performance", "GradingSummary"): "Bar Chart",
    ("Performance", "Average"): "Tabular",
    ("Performance", "AttemptWise"): "Tabular",
    ("Performance", "BestAttempt"): "Tabular",
    ("Performance", "Compare"): "Area Chart",
    ("Performance", "Summary"): "Tabular",
    ("Performance", "PassPercentage"): "Tabular",
    ("Performance", "FailPercentage"): "Tabular",
    ("Performance", "Overall"): "Tabular",
    ("Leave", "Most"): "Tabular",
    ("Leave", "Least"): "Tabular",
    ("Leave", "Current"): "Tabular",
    ("Leave", "Absconded"): "Tabular",
    ("Medical", "Active"): "Tabular",
    ("Medical", "BMI"): "Donut Chart",
    ("Medical", "BloodGroup"): "Tabular",
    ("Medical", "Disease"): "Tabular",
    ("Medical", "Individual"): "Tabular",
    ("Attendance", "Monthly"): "Bar Chart",
    ("Attendance", "Weekly"): "Bar Chart",
    ("Attendance", "Daily"): "Calendar UI",
    ("Attendance", "Present"): "Pie Chart",
    ("Attendance", "Summary"): "Tabular",
    ("Attendance", "Yearly"): "Bar Chart",
    ("Verification", "Pending"): "Tabular",
    ("Verification", "Sent"): "Tabular",
    ("Verification", "NotResponded"): "Tabular",
    ("Verification", "Completed"): "Tabular",
    ("Verification", "Verified"): "Tabular",
    ("Verification", "Rejected"): "Tabular",
    ("Equipment", "Stats"): "Card",
    ("Equipment", "Issued"): "Tabular",
    ("Equipment", "Procured"): "Tabular",
    ("Equipment", "Overdue"): "Tabular",
    ("Equipment", "Returned"): "Tabular",
    ("Equipment", "Holding"): "Tabular",
    ("Equipment", "AgniveerWise"): "Tabular",
    ("Distribution", "Latest"): "Tabular",
    ("Distribution", "ByUnit"): "Tabular",
    ("Distribution", "Unassigned"): "Tabular",
    ("Distribution", "TopUnit"): "Tabular",
    ("Skills", "BySport"): "Tabular",
    ("Skills", "ByClass"): "Tabular",
    ("Roster", "BySport"): "Tabular",
    ("Roster", "ByClass"): "Tabular",
    ("Strength", "Summary"): "Radial Chart",
    ("Overall", "OverallPerformance"): "Tabular",
    ("Schedule", "Company"): "Tabular",
    ("Schedule", "Agniveer"): "Tabular",
    ("Schedule", "Date"): "Tabular",
}

SUBCATEGORY_TO_OPERATION: Dict[str, str] = {
    "TopPerformers": "Top",
    "LowestPerformers": "Bottom",
    "Improvement": "Improvement",
    "Drop": "Drop",
    "GradeDistribution": "Grading",
    "GradingSummary": "GradingSummary",
    "AverageScore": "Average",
    "AttemptWise": "AttemptWise",
    "BestAttempt": "BestAttempt",
    "Comparison": "Compare",
    "SectionSummary": "Summary",
    "PassPercentage": "PassPercentage",
    "FailPercentage": "FailPercentage",
    "OverallPerformance": "Overall",
    "MostLeaveTaken": "Most",
    "LeastLeaveTaken": "Least",
    "CurrentLeaveStatus": "Current",
    "AbscondedPerson": "Absconded",
    "LeaveType": "LeaveType",
    "ActiveCases": "Active",
    "BMIAnalysis": "BMI",
    "DiseaseStatistics": "Disease",
    "BloodGroup": "BloodGroup",
    "IndividualMedical": "Individual",
    "MonthlyAttendance": "Monthly",
    "WeeklyAttendance": "Weekly",
    "DailyAttendance": "Daily",
    "PresentToday": "Present",
    "StrengthBreakdown": "Summary",
    "YearlyAttendance": "Yearly",
    "AttendanceSummary": "Summary",
    "PendingVerification": "Pending",
    "CompletedVerification": "Completed",
    "NotRespondedVerification": "NotResponded",
    "VerifiedVerification": "Verified",
    "RejectedVerification": "Rejected",
    "SentVerification": "Sent",
    "EquipmentSummary": "Stats",
    "OverdueEquipment": "Overdue",
    "PoorConditionEquipment": "Returned",
    "IssuedItems": "Issued",
    "ProcuredItems": "Procured",
    "HoldingEquipment": "Holding",
    "AgniveerWiseEquipment": "AgniveerWise",
    "LatestDistribution": "Latest",
    "DistributionByUnit": "ByUnit",
    "UnassignedItems": "Unassigned",
    "TopUnit": "TopUnit",
    "BySport": "BySport",
    "ByClass": "ByClass",
    "CompanySchedule": "Company",
    "AgniveerSchedule": "Agniveer",
    "DateSchedule": "Date",
}

OPERATION_TO_PAYLOAD_FIELD: Dict[str, str] = {
    "Top": "Top",
    "Bottom": "Bottom",
    "Improvement": "Improvement",
    "Drop": "Drop",
    "Grading": "Grading",
    "GradingSummary": "GradingSummary",
    "Average": "Average",
    "AttemptWise": "AttemptWise",
    "BestAttempt": "BestAttempt",
    "Compare": "Compare",
    "Summary": "Summary",
    "PassPercentage": "PassPercentage",
    "FailPercentage": "FailPercentage",
    "Overall": "Overall",
    "Most": "Most",
    "Least": "Least",
    "Current": "Current",
    "Absconded": "Absconded",
    "Active": "Active",
    "BMI": "BMI",
    "BloodGroup": "BloodGroup",
    "Disease": "Disease",
    "Individual": "Individual",
    "Monthly": "Monthly",
    "Weekly": "Weekly",
    "Daily": "Daily",
    "Yearly": "Yearly",
    "Present": "Present",
    "Stats": "Stats",
    "Issued": "Issued",
    "Procured": "Procured",
    "Overdue": "Overdue",
    "Returned": "Returned",
    "Holding": "Holding",
    "AgniveerWise": "AgniveerWise",
    "Latest": "Latest",
    "ByUnit": "ByUnit",
    "Unassigned": "Unassigned",
    "TopUnit": "TopUnit",
    "BySport": "BySport",
    "ByClass": "ByClass",
    "Pending": "Pending",
    "Sent": "Sent",
    "NotResponded": "NotResponded",
    "Verified": "Verified",
    "Rejected": "Rejected",
    "Company": "Company",
    "Agniveer": "Agniveer",
    "Date": "Date",
}

CATEGORY_OPERATION_TO_SUBCATEGORY: Dict[Tuple[str, str], str] = {
    ("Performance", "Top"): "TopPerformers",
    ("Performance", "Bottom"): "LowestPerformers",
    ("Performance", "Improvement"): "Improvement",
    ("Performance", "Drop"): "Drop",
    ("Performance", "Grading"): "GradeDistribution",
    ("Performance", "GradingSummary"): "GradingSummary",
    ("Performance", "Average"): "AverageScore",
    ("Performance", "AttemptWise"): "AttemptWise",
    ("Performance", "BestAttempt"): "BestAttempt",
    ("Performance", "Compare"): "Comparison",
    ("Performance", "Summary"): "SectionSummary",
    ("Performance", "PassPercentage"): "PassPercentage",
    ("Performance", "FailPercentage"): "FailPercentage",
    ("Performance", "Overall"): "OverallPerformance",
    ("Leave", "Most"): "MostLeaveTaken",
    ("Leave", "Least"): "LeastLeaveTaken",
    ("Leave", "Current"): "CurrentLeaveStatus",
    ("Leave", "Absconded"): "AbscondedPerson",
    ("Leave", "LeaveType"): "LeaveType",
    ("Medical", "Active"): "ActiveCases",
    ("Medical", "BMI"): "BMIAnalysis",
    ("Medical", "Disease"): "DiseaseStatistics",
    ("Medical", "BloodGroup"): "BloodGroup",
    ("Medical", "Individual"): "IndividualMedical",
    ("Attendance", "Monthly"): "MonthlyAttendance",
    ("Attendance", "Weekly"): "WeeklyAttendance",
    ("Attendance", "Daily"): "DailyAttendance",
    ("Attendance", "Present"): "PresentToday",
    ("Attendance", "Summary"): "AttendanceSummary",
    ("Attendance", "Yearly"): "YearlyAttendance",
    ("Strength", "Summary"): "StrengthBreakdown",
    ("Verification", "Pending"): "PendingVerification",
    ("Verification", "Sent"): "SentVerification",
    ("Verification", "NotResponded"): "NotRespondedVerification",
    ("Verification", "Completed"): "CompletedVerification",
    ("Verification", "Verified"): "VerifiedVerification",
    ("Verification", "Rejected"): "RejectedVerification",
    ("Equipment", "Stats"): "EquipmentSummary",
    ("Equipment", "Issued"): "IssuedItems",
    ("Equipment", "Procured"): "ProcuredItems",
    ("Equipment", "Overdue"): "OverdueEquipment",
    ("Equipment", "Returned"): "PoorConditionEquipment",
    ("Equipment", "Holding"): "HoldingEquipment",
    ("Equipment", "AgniveerWise"): "AgniveerWiseEquipment",
    ("Distribution", "Latest"): "LatestDistribution",
    ("Distribution", "ByUnit"): "DistributionByUnit",
    ("Distribution", "Unassigned"): "UnassignedItems",
    ("Distribution", "TopUnit"): "TopUnit",
    ("Roster", "BySport"): "BySport",
    ("Roster", "ByClass"): "ByClass",

    ("Skills", "BySport"): "BySport",
    ("Skills", "ByClass"): "ByClass",
    ("Overall", "OverallPerformance"): "OverallPerformance",
    ("Schedule", "Company"): "CompanySchedule",
    ("Schedule", "Agniveer"): "AgniveerSchedule",
    ("Schedule", "Date"): "DateSchedule",
}

METADATA: Dict[str, Any] = {
    "responseTypes": RESPONSE_TYPES,
    "responseTypeDefault": RESPONSE_TYPE_DEFAULT,
    "detailedKeywords": DETAILED_KEYWORDS,
    "categoryKeywords": CATEGORY_KEYWORDS,
    "operationSynonyms": OPERATION_SYNONYMS,
    "subcategoryToOperation": SUBCATEGORY_TO_OPERATION,
    "categoryOperationToSubcategory": CATEGORY_OPERATION_TO_SUBCATEGORY,
    "intentTypeDefaults": INTENT_TYPE_DEFAULTS,
    "operationToPayloadField": OPERATION_TO_PAYLOAD_FIELD,
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def get_section_by_alias(alias: str) -> Optional[str]:
    """Look up section name by alias."""
    alias_lower = str(alias).lower().strip()
    for section_name, section_data in SECTION.items():
        if alias_lower in section_data["aliases"]:
            return section_name
    return None


def is_valid_category(category: str) -> bool:
    """Check if category is in the official list."""
    return category in OFFICIAL_CATEGORIES


def is_valid_operation(category: str, operation: str) -> bool:
    """Check if operation is valid for the given category."""
    if category not in OPERATIONS_BY_CATEGORY:
        return False
    return operation in OPERATIONS_BY_CATEGORY[category]


def get_allowed_entities_for_category(category: str) -> Set[str]:
    """Get all entity types allowed for a category."""
    allowed = COMMON_ENTITY_TYPES.copy()
    if category in CATEGORY_ENTITY_COMPATIBILITY:
        allowed.update(CATEGORY_ENTITY_COMPATIBILITY[category])
    return allowed
