"""
admin_intent.py
===============
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

# =============================================================================
# SUBCATEGORY → EXACT .NET OPERATION STRING
# =============================================================================

_SUBCATEGORY_TO_OPERATION: Dict[str, str] = {
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
    "ActiveCases": "Active",
    "BMIAnalysis": "BMI",
    "DiseaseStatistics": "Disease",
    "MonthlyAttendance": "Monthly",
    "PresentToday": "Present",
    "StrengthBreakdown": "Strength",
    "PendingVerification": "Pending",
    "CompletedVerification": "Completed",
    "EquipmentSummary": "Stats",
    "OverdueEquipment": "Overdue",
    "PoorConditionEquipment": "Returned",
    "LatestDistribution": "Latest",
    "DistributionByUnit": "ByUnit",
    "UnassignedItems": "Unassigned",
    "TopUnit": "TopUnit",
    "BySport": "BySport",
    "ByClass": "ByClass",
    "BloodGroup": "BloodGroup",
    "IssuedItems": "IssuedItems",
    "ProcuredItems": "ProcuredItems",
}


_PERFORMANCE_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    (
        "Top Performers",
        "TopPerformers",
        (
            "top",
            "highest",
            "best",
            "top performer",
            "top performers",
            "highest performer",
            "highest performers",
            "best performer",
            "best performers",
            "top scorer",
            "top scorers",
            "highest scorer",
            "highest scorers",
            "best scorer",
            "best scorers",
            "top scoring",
            "highest scoring",
            "best scoring",
            "highest marks",
            "best marks",
            "maximum marks",
            "highest score",
            "maximum score",
            "best score",
            "highest marks scored",
            "most marks",
            "rank 1",
            "first rank",
            "first position",
            "first place",
            "topper",
            "toppers",
            "leading performer",
            "leading performers",
            "leading scorer",
            "leading scorers",
            "outstanding performer",
            "outstanding performers",
            "ace performer",
            "ace performers",
            "stellar performer",
            "stellar performers",
            "high achiever",
            "high achievers",
            "high scorer",
            "high scorers",
            "who topped",
            "who scored highest",
            "who scored the most",
            "who scored maximum",
            "who is the best",
            "who is number one",
            "who performed best",
            "who did best",
            "top agniveer",
            "top agniveers",
            "best agniveer",
            "best agniveers",
            "highest ranked",
            "top ranked",
            "maximum scorer",
            "strong performer",
            "strong performers",
        ),
    ),
    (
        "Lowest Performers",
        "LowestPerformers",
        (
            "bottom",
            "lowest",
            "worst",
            "bottom performer",
            "bottom performers",
            "lowest performer",
            "lowest performers",
            "worst performer",
            "worst performers",
            "bottom scorer",
            "bottom scorers",
            "lowest scorer",
            "lowest scorers",
            "worst scorer",
            "worst scorers",
            "lowest marks",
            "worst marks",
            "minimum marks",
            "lowest score",
            "minimum score",
            "worst score",
            "last rank",
            "last position",
            "last place",
            "poor performer",
            "poor performers",
            "weakest performer",
            "weakest performers",
            "weakest student",
            "weakest students",
            "trailing performer",
            "trailing performers",
            "struggling performer",
            "struggling performers",
            "underperformer",
            "underperformers",
            "low scorer",
            "low scorers",
            "who scored lowest",
            "who scored the least",
            "who scored minimum",
            "who is the worst",
            "who did worst",
            "who performed worst",
            "who scored last",
            "who needs help",
            "who needs improvement",
            "bottom agniveer",
            "bottom agniveers",
            "lowest agniveer",
            "worst agniveer",
            "minimum scorer",
            "lowest achieving",
        ),
    ),
    (
        "Improvement",
        "Improvement",
        (
            "improvement",
            "improve",
            "improved",
            "score improvement",
            "score improved",
            "improvement between attempts",
            "improved between attempts",
            "improvement from attempt",
            "improvement between",
            "score went up",
            "marks went up",
            "positive progress",
            "positive trend",
            "biggest improvement",
            "biggest gain",
            "most improved",
            "rising score",
            "rising scores",
            "upward trend",
            "who improved",
            "who got better",
            "who climbed",
            "score gain",
            "marks gain",
            "performance gain",
            "score jump",
            "biggest jump",
            "jumped the most",
            "recovered",
            "recovery",
            "bounced back",
            "getting better",
            "progressing",
            "upward movement",
            "score increase",
            "score increased",
            "improved performance",
            "performance improved",
            "who progressed",
            "progress between attempts",
            "growth in score",
            "score growth",
            "positive movement",
            "better than last time",
            "better than before",
            "increase in score",
            "gained marks",
        ),
    ),
    (
        "Drop",
        "Drop",
        (
            "drop",
            "decline",
            "dropped",
            "score drop",
            "score dropped",
            "score declined",
            "biggest drop",
            "biggest decline",
            "most dropped",
            "dropped between attempts",
            "decline between attempts",
            "downward trend",
            "negative trend",
            "who dropped",
            "who declined",
            "who got worse",
            "who fell",
            "who slipped",
            "who is slipping",
            "performance degraded",
            "performance declined",
            "regression",
            "score regression",
            "regressed",
            "deterioration",
            "score deterioration",
            "deteriorated",
            "score loss",
            "negative movement",
            "getting worse",
            "falling behind",
            "lower than before",
            "lower than last time",
            "downward movement",
            "falling scores",
            "negative progress",
            "went down",
            "score fell",
            "score fallen",
            "marks dropped",
            "decrease in score",
            "score decreased",
            "slipped in performance",
            "performance slip",
            "worsening performance",
            "who is struggling more",
        ),
    ),
    (
        "Grade Distribution",
        "GradeDistribution",
        (
            "grading",
            "grade",
            "filter by grade",
            "filter by grading",
            "show by grade",
            "show by grading",
            "grade wise",
            "grading wise",
            "by grade",
            "by grading",
            "grading category",
            "grade category",
            "grading filter",
            "grade filter",
            "who got excellent",
            "who got good",
            "who got sat",
            "who got fail",
            "who got unsa",
            "who scored excellent",
            "who scored good",
            "who scored sat",
            "show excellents",
            "show goods",
            "show fails",
            "who are in excellent",
            "who are in good",
            "performance by grade",
            "performance by grading",
            "grade breakdown",
            "grading breakdown",
            "exceptionally well",
            "excellent grade",
            "good grade",
            "sat grade",
            "fail grade",
            "unsa grade",
            "categorised by grade",
            "categorized by grade",
        ),
    ),
    (
        "Grading Summary",
        "GradingSummary",
        (
            "gradingsummary",
            "gradesummary",
            "grading summary",
            "grade summary",
            "distribution of grades",
            "grade distribution summary",
            "summary of grading",
            "grading overview",
            "grade overview",
            "how many in each grade",
            "how many per grade",
            "how many got excellent",
            "how many got good",
            "how many got sat",
            "how many got fail",
            "how many got unsa",
            "how many passed grading",
            "how many failed grading",
            "grade count",
            "grading count",
            "grade tally",
            "grading tally",
            "grade wise count",
            "grading wise count",
            "total in each grade",
            "total per grade",
            "grade statistics",
            "grading statistics",
            "breakdown by grade",
            "grade totals",
            "grade distribution summary",
            "grade distribution overview",
        ),
    ),
    (
        "Average Score",
        "AverageScore",
        (
            "average",
            "avg",
            "mean",
            "average score",
            "average marks",
            "average performance",
            "mean score",
            "mean marks",
            "section average",
            "section mean",
            "what is the average",
            "overall average",
            "average analysis",
            "typical score",
            "norm score",
            "standard score",
            "avg score",
            "avg marks",
            "how much on average",
            "average achievement",
            "average result",
            "average per section",
            "section wise average",
            "average subsection score",
            "subsection average",
            "average of all",
            "what do they score on average",
            "typical performance",
        ),
    ),
    (
        "Attempt Wise",
        "AttemptWise",
        (
            "attemptwise",
            "byattempt",
            "attempts",
            "attempt wise",
            "attempt-wise",
            "attempt wise analysis",
            "per attempt",
            "by attempt",
            "each attempt",
            "attempt breakdown",
            "attempts breakdown",
            "attempt 1",
            "attempt 2",
            "attempt 3",
            "first attempt",
            "second attempt",
            "third attempt",
            "attempt number",
            "attempt no",
            "how did each attempt go",
            "score per attempt",
            "attempt by attempt",
            "attempt to attempt",
            "attempt 1 vs attempt 2",
            "compare attempts",
            "all attempts",
            "multiple attempts",
            "how many attempts",
            "attempt history",
            "progression across attempts",
            "attempt progression",
            "scores across attempts",
            "performance per attempt",
            "attempt statistics",
            "attempt data",
            "each attempt score",
            "every attempt",
        ),
    ),
    (
        "Best Attempt",
        "BestAttempt",
        (
            "bestattempt",
            "bestscoreattempt",
            "best attempt",
            "best attempt analysis",
            "best attempt score",
            "best scoring attempt",
            "highest attempt",
            "peak attempt",
            "top attempt",
            "maximum attempt",
            "best they scored",
            "best score attempt",
            "what was their best",
            "best performance attempt",
            "peak performance attempt",
            "personal best attempt",
            "highest single attempt",
            "best attempt result",
            "who had the best attempt",
            "maximum score attempt",
            "best score in any attempt",
            "best try",
            "most they scored in one attempt",
        ),
    ),
    (
        "Comparison",
        "Comparison",
        (
            "compare",
            "comparison",
            " vs ",
            "compare sections",
            "section comparison",
            "section vs section",
            "bpet vs ppt",
            "ppt vs firing",
            "firing vs drill",
            "versus",
            "compared to",
            "compared with",
            "difference between sections",
            "contrast sections",
            "relative performance",
            "how do sections compare",
            "which section is better",
            "which section is worse",
            "cross section comparison",
            "comparative analysis",
            "bpet and ppt comparison",
            "side by side comparison",
            "head to head",
            "compare performance",
            "performance comparison",
            "compare results",
        ),
    ),
    (
        "Section Summary",
        "SectionSummary",
        (
            "summary",
            "sectionsummary",
            "section summary",
            "section overview",
            "section wise summary",
            "section-wise summary",
            "consolidated section report",
            "section snapshot",
            "section brief",
            "all sections overview",
            "section level summary",
            "quick section view",
            "how are sections performing",
            "section performance summary",
            "overall section view",
            "section report",
            "section wise overview",
            "section wise report",
            "overview of all sections",
            "section breakdown summary",
            "all sections at a glance",
        ),
    ),
    (
        "Pass Percentage",
        "PassPercentage",
        (
            "passpercentage",
            "passrate",
            "pass percentage",
            "passing percentage",
            "pass rate",
            "passing rate",
            "how many passed",
            "who passed",
            "clearance rate",
            "success rate",
            "percentage who passed",
            "how many cleared",
            "who cleared the exam",
            "who cleared the test",
            "number of passes",
            "pass count percentage",
            "what percentage passed",
            "passing ratio",
            "qualified percentage",
            "qualified rate",
            "selection rate",
            "cleared percentage",
            "who met the standard",
            "how many met the cutoff",
            "pass statistics",
        ),
    ),
    (
        "Fail Percentage",
        "FailPercentage",
        (
            "failpercentage",
            "failrate",
            "fail percentage",
            "failure percentage",
            "fail rate",
            "failure rate",
            "how many failed",
            "who failed",
            "percentage who failed",
            "failure ratio",
            "failure statistics",
            "fail count percentage",
            "number of failures",
            "what percentage failed",
            "who did not pass",
            "who did not clear",
            "who missed the cutoff",
            "who did not qualify",
            "rejection rate",
            "disqualification rate",
            "failure count",
            "how many did not clear",
            "failed the exam",
            "failed the test",
        ),
    ),
    (
        "Overall Performance",
        "OverallPerformance",
        (
            "overall",
            "composite",
            "allcriteria",
            "overall performers",
            "overall performance",
            "overall performance report",
            "overall score",
            "overall scoring",
            "overall marks",
            "all criteria performance",
            "all criteria",
            "all section performance",
            "multi section performance",
            "combined performance",
            "aggregate performance",
            "total performance",
            "holistic performance",
            "performance across all sections",
            "overall performer",
            "all round performance",
            "all-round performance",
            "combined score",
            "total score across sections",
            "composite score",
            "composite performance",
            "performance in all sections",
            "full performance report",
        ),
    ),
]


_LEAVE_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    (
        "Most Leave Taken",
        "MostLeaveTaken",
        (
            "most",
            "highest",
            "maximum",
            "most leave taken",
            "most leaves taken",
            "highest leave taken",
            "maximum leave taken",
            "most leave",
            "most leaves",
            "highest leave",
            "maximum leave",
            "most absent",
            "most absentee",
            "highest absentee",
            "taken most leave",
            "taken maximum leave",
            "who has the most leave",
            "who took maximum leave",
            "who is absent the most",
            "maximum absentee",
            "most days absent",
            "maximum days on leave",
            "most days off",
            "maximum off days",
            "most frequent leave taker",
        ),
    ),
    (
        "Least Leave Taken",
        "LeastLeaveTaken",
        (
            "least",
            "lowest",
            "minimum",
            "least leave taken",
            "fewest leave taken",
            "lowest leave taken",
            "minimum leave taken",
            "least leave",
            "fewest leave",
            "fewest leaves",
            "lowest leave",
            "minimum leave",
            "least absent",
            "least absentee",
            "minimum absentee",
            "taken least leave",
            "taken minimum leave",
            "who has the least leave",
            "who took minimum leave",
            "who is least absent",
            "fewest days absent",
            "minimum days on leave",
            "least days off",
            "most regular",
            "most punctual",
            "best attendance person",
        ),
    ),
    (
        "Current Leave Status",
        "CurrentLeaveStatus",
        (
            "current",
            "today",
            "now",
            "currently on leave",
            "who is on leave",
            "on leave today",
            "current leave status",
            "leave today",
            "leave now",
            "current leave",
            "leave status",
            "on leave now",
            "who is absent today",
            "absent today",
            "how many on leave",
            "who is not here today",
            "who is away",
            "people on leave",
            "persons on leave",
            "leave list today",
            "today leave",
            "currently absent",
            "active leave",
            "who is on leave right now",
        ),
    ),
    (
        "Absconded Person",
        "AbscondedPerson",
        (
            "absconded",
            "abscond",
            "absconded leave records",
            "absconded person",
            "gone missing",
            "missing person",
            "awol",
            "went missing",
            "unauthorised absence",
            "unauthorized absence",
            "went awol",
            "no information",
            "whereabouts unknown",
            "abandoned post",
            "left without permission",
            "did not return",
            "overstayed leave",
            "not reported back",
            "missing from duty",
        ),
    ),
]


_MEDICAL_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    (
        "Active Cases",
        "ActiveCases",
        (
            "active",
            "cases",
            "admitted",
            "active medical cases",
            "active cases",
            "current patients",
            "admitted patients",
            "in ward",
            "hospitalised",
            "hospitalized",
            "medical case",
            "medical cases",
            "currently in hospital",
            "under treatment",
            "ongoing medical cases",
            "how many sick",
            "how many in hospital",
            "how many admitted",
            "medical ward count",
            "ward count",
            "who is admitted",
            "who is in hospital",
        ),
    ),
    (
        "BMI Analysis",
        "BMIAnalysis",
        (
            "bmi",
            "weight",
            "fitness",
            "bmi outliers",
            "bmi analysis",
            "body mass index",
            "weight analysis",
            "fitness analysis",
            "fitness report",
            "overweight",
            "underweight",
            "body weight analysis",
            "bmi statistics",
            "who is obese",
            "obese agniveers",
            "underweight agniveers",
            "weight distribution",
            "physical fitness data",
            "height weight ratio",
            "weight report",
            "fitness data",
        ),
    ),
    (
        "Disease Statistics",
        "DiseaseStatistics",
        (
            "disease",
            "diagnoses",
            "diagnosis",
            "top diagnoses",
            "disease statistics",
            "top disease",
            "common disease",
            "disease analysis",
            "illness statistics",
            "most common disease",
            "frequent disease",
            "common illness",
            "illness breakdown",
            "disease frequency",
            "medical diagnoses breakdown",
            "what diseases are common",
            "health trends",
            "common ailments",
            "disease report",
        ),
    ),
]


_ATTENDANCE_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    (
        "Monthly Attendance",
        "MonthlyAttendance",
        (
            "monthly",
            "month",
            "stats",
            "monthly attendance",
            "monthly attendance statistics",
            "attendance statistics",
            "attendance stats",
            "attendance this month",
            "attendance for the month",
            "attendance for this month",
            "current month attendance",
            "this month attendance",
            "last month attendance",
            "month wise attendance",
            "attendance by month",
            "attendance report",
            "monthly report",
            "monthly stats",
            "overall attendance",
            "overall attendance statistics",
            "attendance record",
            "monthly attendance record",
            "attendance history",
            "attendance percentage",
        ),
    ),
    (
        "Present Today",
        "PresentToday",
        (
            "present",
            "campus",
            "today",
            "present on campus",
            "present today",
            "who is present",
            "how many present",
            "attendance today",
            "today attendance",
            "on campus today",
            "on campus",
            "who came today",
            "who is here today",
            "strength today",
            "today strength",
            "how many are here",
            "headcount today",
            "today headcount",
            "actual strength today",
            "who is on campus",
        ),
    ),
    (
        "Strength Breakdown",
        "StrengthBreakdown",
        (
            "strength",
            "breakdown",
            "strength breakdown",
            "total strength",
            "strength report",
            "headcount breakdown",
            "how many total",
            "overall strength",
            "strength statistics",
            "total headcount",
            "platoon strength",
            "unit strength",
            "platoon wise strength",
            "total platoon count",
            "active inactive breakdown",
            "active count",
            "total active",
            "total inactive",
        ),
    ),
]


_VERIFICATION_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    (
        "Pending Verification",
        "PendingVerification",
        (
            "pending",
            "sent",
            "noresponse",
            "pending verifications",
            "sent but no response",
            "no response verification",
            "awaiting verification",
            "verification pending",
            "not verified",
            "verification not done",
            "documents not verified",
            "waiting for verification",
            "verification in progress",
            "incomplete verification",
            "unverified documents",
            "verification outstanding",
            "yet to be verified",
            "how many pending",
            "verification queue",
            "who is pending verification",
        ),
    ),
    (
        "Completed Verification",
        "CompletedVerification",
        (
            "completed",
            "verified",
            "done",
            "completed verifications",
            "verification completed",
            "verification done",
            "verified documents",
            "verification finished",
            "documents verified",
            "cleared verification",
            "verification successful",
            "who got verified",
            "verification received",
            "police verification done",
            "verification cleared",
            "how many verified",
            "total verified",
        ),
    ),
]


_EQUIPMENT_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    (
        "Equipment Summary",
        "EquipmentSummary",
        (
            "stats",
            "summary",
            "overview",
            "equipment stats",
            "equipment summary",
            "equipment overview",
            "equipment report",
            "gear summary",
            "kit summary",
            "inventory summary",
            "equipment status",
            "all equipment",
            "equipment inventory",
            "total equipment",
            "how much equipment",
            "equipment count",
            "inventory overview",
        ),
    ),
    (
        "Overdue Equipment",
        "OverdueEquipment",
        (
            "overdue",
            "late",
            "overdue equipment",
            "equipment overdue",
            "overdue returns",
            "not returned equipment",
            "late equipment",
            "equipment not returned",
            "overdue gear",
            "equipment past due",
            "equipment still out",
            "who has not returned",
            "pending equipment return",
            "unreturned equipment",
            "equipment outstanding",
            "equipment not submitted",
            "not submitted equipment",
        ),
    ),
    (
        "Returned Poor Condition",
        "PoorConditionEquipment",
        (
            "returned",
            "poor",
            "condition",
            "returned poor condition",
            "poor condition equipment",
            "equipment returned poor",
            "damaged equipment",
            "bad condition equipment",
            "equipment damaged",
            "equipment in bad shape",
            "equipment degraded",
            "broken equipment",
            "worn out equipment",
            "equipment wear",
            "equipment tear",
            "unserviceable equipment",
            "defective equipment",
            "poor condition returned",
        ),
    ),
    (
        "Issued Items",
        "IssuedItems",
        (
            "issued items",
            "items issued",
            "all issued items",
            "list of issued items",
            "issued equipment list",
            "kit issued",
            "gear issued",
            "uniform issued",
            "what was issued",
            "what is issued to agniveer",
            "issued clothing",
            "issued kit list",
            "dms boot",
            "gp boot",
            "pt shoes brown",
            "cap fs",
            "mug steel",
            "blanket",
            "terry towel",
            "under pant woollen",
            "vest woollen og",
            "vest cotton h/s",
            "ground shed",
            "kit bag",
            "combat t shirt",
            "net mosquito",
            "pagari 5.5m",
            "combat coat",
            "belt ick",
            "combat dress",
            "line bedding",
            "ffd",
            "cover water proof og",
            "cover water pro sikh",
            "haver shack",
            "boot high ankle dvs",
            "spoon desert",
            "frog bayonet ick",
            "net camouflage h/d",
            "lases nylon",
            "pouches amn ick",
            "pack with allmn frame",
            "cord disk identity",
            "disk identity oval",
            "disk identity round",
            "water bottle plastic",
            "vest cotton white s4",
            "drawers cotton white",
            "jersey v neck",
            "short kd",
            "knee elbow",
            "socks woollen og",
            "trouser drill khaki",
            "shirt man khaki",
            "short kd light green",
            "jersey man woollen",
            "shirt angola drive",
            "trouser bd serge",
            "issued",
        ),
    ),
    (
        "Procured Items",
        "ProcuredItems",
        (
            "procured items",
            "items procured",
            "all procured items",
            "list of procured items",
            "procured equipment list",
            "kit procured",
            "gear procured",
            "self purchased",
            "what was procured",
            "what is procured by agniveer",
            "bought items",
            "purchase list",
            "items to buy",
            "procured clothing",
            "procured kit list",
            "rect bag khaki",
            "mufti shoes",
            "black socks",
            "black pagri with fifty",
            "drill shoes",
            "pt dress complete",
            "games dress",
            "mufti dress white",
            "regt tie with clip",
            "mufti blazer",
            "khaki half sleeves",
            "khaki full sleeves",
            "socks og",
            "green pagri with fifty",
            "og dress",
            "combat cap",
            "leather belt with crest",
            "bed sheet",
            "rect shoulder",
            "name plate black",
            "name plate khaki",
            "belt black",
            "water bottle 02 ltr",
            "mug plastic",
            "box steel with paint",
            "hanger",
            "health card",
            "bed card",
            "locker cloth",
            "firing data card",
            "track suit with name",
            "clip board",
            "256 pages copy",
            "white hanky",
            "steel thali",
            "glass steel",
            "spoon steel",
            "soap case",
            "indls photograph",
            "small bucket",
            "progress card",
            "mattress",
            "angola shirt with og pant",
            "big bucket",
            "pt vest with chest no",
            "underwear",
            "swimming costumes",
            "swimming cover",
            "jungle shoes",
            "barret cap",
            "rifle sling",
            "procured",
        ),
    ),
]


_DISTRIBUTION_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    (
        "Latest Distribution",
        "LatestDistribution",
        (
            "latest",
            "recent",
            "last",
            "latest distribution",
            "recent distribution",
            "last distribution",
            "newest distribution",
            "most recent distribution",
            "current distribution",
            "today distribution",
            "fresh distribution",
            "last issued distribution",
        ),
    ),
    (
        "Distribution By Unit",
        "DistributionByUnit",
        (
            "byunit",
            "unit",
            "inunit",
            "distribution by unit",
            "agniveers in unit",
            "how many agniveers in",
            "agniveers in the unit",
            "by unit distribution",
            "unit wise distribution",
            "per unit distribution",
            "in unit distribution",
            "by unit",
            "in unit",
            "unit distribution",
            "unit wise",
            "regiment wise distribution",
            "company wise distribution",
            "which unit has how many",
        ),
    ),
    (
        "Unassigned Items",
        "UnassignedItems",
        (
            "unassigned",
            "notassigned",
            "nounit",
            "not assigned to unit",
            "unassigned agniveers",
            "no unit assigned",
            "items without unit",
            "not assigned",
            "no unit",
            "who has no unit",
            "who is not assigned",
            "pending unit assignment",
            "without unit",
            "not yet assigned",
            "unit not assigned",
            "awaiting unit assignment",
        ),
    ),
    (
        "Top Unit",
        "TopUnit",
        (
            "topunit",
            "highestunit",
            "top unit",
            "highest unit",
            "unit with most agniveers",
            "most agniveers in unit",
            "unit with highest",
            "highest distribution unit",
            "which unit has the highest",
            "which unit has most",
            "biggest unit",
            "largest unit",
            "unit with maximum agniveers",
            "which unit received most",
        ),
    ),
]


_SKILLS_INTENTS: List[Tuple[str, str, Tuple[str, ...]]] = [
    (
        "By Sport",
        "BySport",
        (
            "bysport",
            "sport",
            "sports",
            "best performers in sport",
            "best in sport",
            "by sport",
            "sport wise",
            "roster by sport",
            "skills by sport",
            "cricket players",
            "football players",
            "who plays cricket",
            "who plays football",
            "sports roster",
            "sport category",
            "sport wise roster",
            "athletes",
            "who are the sportsmen",
            "sportsperson roster",
            "sport filter",
        ),
    ),
    (
        "By Class",
        "ByClass",
        (
            "byclass",
            "class",
            "agniveers by class",
            "by class",
            "class wise",
            "roster by class",
            "skills by class",
            "sikh class",
            "dogra class",
            "jat class",
            "class wise roster",
            "class distribution",
            "which class has how many",
            "class breakdown",
            "roster by community",
            "community wise roster",
            "class filter",
        ),
    ),
    (
        "Blood Group",
        "BloodGroup",
        (
            "bloodgroup",
            "blood",
            "blood group statistics",
            "blood group distribution",
            "blood group",
            "blood type",
            "a positive",
            "b positive",
            "o positive",
            "ab positive",
            "ab negative",
            "blood group breakdown",
            "how many with blood group",
            "blood type distribution",
            "blood group count",
            "who has which blood group",
            "blood group roster",
            "blood group filter",
        ),
    ),
]


ADMIN_FUZZY_VOCAB: Dict[str, str] = {
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
    "attnc": "attnc",
    "attenc": "attnc",
    "attn c": "attnc",
    "att nc": "attnc",
    "attnce": "attnc",
    "atncl": "attnc",
    "exppg": "exppg",
    "ex ppg": "exppg",
    "ex-ppg": "exppg",
    "expg": "exppg",
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

_ADMIN_CANONICAL_CASE: Dict[str, str] = {
    "bpet": "BPET",
    "ppt": "PPT",
    "firing": "Firing",
    "drill": "Drill",
}


def admin_normalize_query(query: str) -> str:
    """Fix misspellings and restore canonical section casing."""
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


_SECTION_MAP: Dict[str, str] = {
    "bpet": "BPET",
    "bept": "BPET",
    "ppt": "PPT",
    "firing": "Firing",
    "drill": "Drill",
}

_SUBSECTION_PATTERNS: Dict[str, str] = {
    "5km": "5km",
    "5 km": "5km",
    "chin up": "Chin Ups",
    "chin-up": "Chin Ups",
    "chinup": "Chin Ups",
    "h rope": "H Rope",
    "h-rope": "H Rope",
    "hrope": "H Rope",
}

_GRADING_MAP: Dict[str, str] = {
    "exceptionally well": "ExceptionallyWell",
    "excellent": "Excellent",
    "good": "Good",
    "sat": "SAT",
    "fail": "Fail",
    "unsa": "UNSA",
}

_LEAVE_TYPE_MAP: Dict[str, str] = {
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
}

_SPORT_MAP: Dict[str, str] = {
    "cricket": "Cricket",
    "football": "Football",
    "running": "Running",
    "basketball": "Basketball",
    "volleyball": "Volleyball",
    "kabaddi": "Kabaddi",
    "hockey": "Hockey",
}

_CLASS_MAP: Dict[str, str] = {
    "sikh": "Sikh",
    "oic": "OIC",
    "gurkha": "Gurkha",
    "gorkha": "Gurkha",
    "dogra": "Dogra",
    "jat": "Jat",
    "rajput": "Rajput",
    "punjabi": "Punjabi",
}


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

_ITEM_LOOKUP: Dict[str, Tuple[str, str]] = {}
for _item in ISSUED_ITEMS:
    _ITEM_LOOKUP[_item.lower()] = (_item, "IssuedItems")
for _item in PROCURED_ITEMS:
    _ITEM_LOOKUP[_item.lower()] = (_item, "ProcuredItems")

_GENERIC_WORDS = {
    "by",
    "in",
    "for",
    "per",
    "top",
    "distribution",
    "show",
    "of",
    "the",
    "a",
    "an",
    "latest",
    "recent",
    "last",
    "get",
    "give",
    "list",
}


_MODULES: Dict[str, Tuple[Tuple[str, ...], List[Tuple[str, str, Tuple[str, ...]]]]] = {
    "Attendance": (
        (
            "monthly attendance",
            "attendance statistics",
            "attendance stats",
            "overall attendance",
            "attendance this month",
            "attendance report",
            "present on campus",
            "on campus today",
            "who came today",
            "strength breakdown",
            "total strength",
            "headcount breakdown",
            "attendance",
            "present",
            "campus",
            "strength",
            "headcount",
            "muster",
            "monthly",
        ),
        _ATTENDANCE_INTENTS,
    ),
    "Performance": (
        (
            "top performer",
            "top performers",
            "top scorer",
            "highest performer",
            "highest performers",
            "highest scorer",
            "best performer",
            "best performers",
            "best scorer",
            "topper",
            "toppers",
            "leading performer",
            "ace performer",
            "outstanding performer",
            "who topped",
            "who scored highest",
            "bottom performer",
            "bottom performers",
            "bottom scorer",
            "lowest performer",
            "lowest performers",
            "lowest scorer",
            "worst performer",
            "worst performers",
            "worst scorer",
            "poor performer",
            "weakest performer",
            "last rank",
            "improvement",
            "improve",
            "improved",
            "score improvement",
            "score improved",
            "score gain",
            "most improved",
            "score went up",
            "positive progress",
            "drop",
            "decline",
            "dropped",
            "score drop",
            "score dropped",
            "regression",
            "score fell",
            "grading",
            "grade",
            "filter by grade",
            "grade wise",
            "by grading",
            "grade distribution",
            "who got excellent",
            "who got sat",
            "who got fail",
            "gradingsummary",
            "gradesummary",
            "grading summary",
            "grade summary",
            "how many in each grade",
            "average score",
            "average marks",
            "mean score",
            "avg score",
            "section average",
            "attemptwise",
            "byattempt",
            "attempts",
            "attempt wise",
            "per attempt",
            "by attempt",
            "each attempt",
            "attempt 1",
            "attempt 2",
            "score per attempt",
            "bestattempt",
            "bestscoreattempt",
            "best attempt",
            "highest attempt",
            "peak attempt",
            "compare",
            "comparison",
            " vs ",
            "compare sections",
            "section comparison",
            "bpet vs ppt",
            "section summary",
            "section overview",
            "sectionsummary",
            "pass percentage",
            "passpercentage",
            "passrate",
            "fail percentage",
            "failpercentage",
            "failrate",
            "how many passed",
            "how many failed",
            "pass rate",
            "fail rate",
            "overall",
            "composite",
            "allcriteria",
            "overall performance",
            "composite performance",
            "all criteria",
            "performance",
            "score",
            "marks",
            "exam",
            "bpet",
            "ppt",
            "firing",
            "drill",
            "performer",
            "performers",
            "attempt",
            "section",
            "who scored",
            "who passed",
            "who failed",
        ),
        _PERFORMANCE_INTENTS,
    ),
    "Leave": (
        (
            "most leave",
            "most leave taken",
            "highest leave",
            "maximum leave",
            "least leave",
            "least leave taken",
            "lowest leave",
            "minimum leave",
            "currently on leave",
            "who is on leave",
            "on leave today",
            "absconded",
            "abscond",
            "awol",
            "went missing",
            "leave today",
            "leave status",
            "leave taken",
            "leave",
            "absent",
            "absentee",
            "sick leave",
            "annual leave",
            "medical leave",
            "attnc",
            "attenc",
            "att nc",
            "exppg",
            "ex ppg",
            "ex-ppg",
        ),
        _LEAVE_INTENTS,
    ),
    "Medical": (
        (
            "active medical cases",
            "medical cases",
            "bmi",
            "bmi analysis",
            "bmi outliers",
            "disease",
            "diagnoses",
            "diagnosis",
            "top diagnoses",
            "body mass index",
            "fitness analysis",
            "medical",
            "health",
            "hospital",
            "patient",
            "ward",
            "admitted",
            "ailment",
            "illness",
            "how many sick",
            "common disease",
        ),
        _MEDICAL_INTENTS,
    ),
    "Verification": (
        (
            "pending verifications",
            "sent but no response",
            "awaiting verification",
            "verification pending",
            "not verified",
            "completed verifications",
            "verification completed",
            "verification done",
            "verified documents",
            "verification",
            "verify",
            "verified",
            "document verification",
            "police verification",
        ),
        _VERIFICATION_INTENTS,
    ),
    "Equipment": (
        (
            "equipment stats",
            "equipment summary",
            "equipment overview",
            "overdue equipment",
            "equipment overdue",
            "overdue returns",
            "returned poor condition",
            "poor condition equipment",
            "damaged equipment",
            "equipment",
            "gear",
            "overdue",
            "inventory",
            "damaged",
            "issued items",
            "procured items",
            "items issued",
            "items procured",
            "kit issued",
            "kit procured",
            "self purchased",
            "bought items",
            "dms boot",
            "cap fs",
            "mug steel",
            "terry towel",
            "combat coat",
            "belt ick",
            "haver shack",
            "net camouflage",
            "pagari 5.5m",
            "lases nylon",
            "rect bag",
            "mufti shoes",
            "black pagri",
            "drill shoes",
            "games dress",
            "mufti dress",
            "regt tie",
            "mufti blazer",
            "socks og",
            "green pagri",
            "og dress",
            "combat cap",
            "leather belt with crest",
            "rifle sling",
            "barret cap",
            "jungle shoes",
            "health card",
            "health card details",
        ),
        _EQUIPMENT_INTENTS,
    ),
    "Distribution": (
        (
            "latest distribution",
            "recent distribution",
            "distribution by unit",
            "agniveers in unit",
            "by unit",
            "unassigned agniveers",
            "not assigned",
            "no unit",
            "top unit",
            "highest unit",
            "unit with most",
            "distribution",
            "distributed",
        ),
        _DISTRIBUTION_INTENTS,
    ),
    "Skills": (
        (
            "by sport",
            "sport wise",
            "roster by sport",
            "bysport",
            "by class",
            "class wise",
            "byclass",
            "blood group",
            "blood type",
            "bloodgroup",
            "skill",
            "sport",
            "sports",
            "roster",
            "blood",
        ),
        _SKILLS_INTENTS,
    ),
}


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


def _extract_leave_type(
    text_lower: str, category: Optional[str] = None
) -> Optional[str]:
    if category and category not in ("Leave", None):
        new_type_phrases = {
            "attnc",
            "attenc",
            "attn c",
            "att nc",
            "ex ppg",
            "exppg",
            "ex-ppg",
        }
        found = None
        best_len = 0
        for phrase, code in _LEAVE_TYPE_MAP.items():
            if phrase in text_lower and len(phrase) > best_len:
                if phrase in new_type_phrases or category == "Leave":
                    found = code
                    best_len = len(phrase)
        return found

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
    best_key: Optional[str] = None
    best_len = 0
    for key in _ITEM_LOOKUP:
        if key in text_lower and len(key) > best_len:
            best_key = key
            best_len = len(key)
    if best_key:
        name, cat = _ITEM_LOOKUP[best_key]
        return name, cat

    _STOP = {
        "a",
        "an",
        "the",
        "of",
        "for",
        "in",
        "with",
        "and",
        "or",
        "is",
        "are",
        "was",
        "were",
        "show",
        "list",
        "get",
        "give",
        "tell",
        "me",
        "its",
        "item",
        "items",
        "all",
        "any",
        "which",
        "what",
        "who",
        "how",
        "where",
        "do",
        "does",
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

    return best_match if best_match else (None, None)


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
    match = re.search(r"\bto\s*attempt\s*(\d+)\b", text_lower) or re.search(
        r"\battempt\s*\d+\s+to\s+(\d+)\b", text_lower
    )
    return int(match.group(1)) if match else None


def _extract_date(text: str) -> Optional[str]:
    patterns = [
        r"\b(\d{4}-\d{2}-\d{2})\b",
        r"\b(\d{2}/\d{2}/\d{4})\b",
        r"\b((?:January|February|March|April|May|June|July|August|September|"
        r"October|November|December)\s+\d{4})\b",
        r"\b((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{4})\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1)
    return None


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
        intent_score = sum(_score_intent(query_lower, kws) for _, _, kws in intent_list)
        if intent_score > best_intent_score:
            best_intent_score = intent_score
            best_module = module

    return best_module or next((m for m in _MODULES if m in tied), tied[0])


def _match_intent(
    query_lower: str,
    intent_list: List[Tuple[str, str, Tuple[str, ...]]],
) -> Optional[Tuple[str, str]]:
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


def classify_admin_intent(query: str) -> Dict[str, Any]:
    raw_query = (query or "").strip()
    q = _normalise(raw_query)

    result: Dict[str, Any] = {
        "category": None,
        "subcategory": None,
        "number": None,
        "section": None,
        "sub_section": None,
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
        "raw_query": raw_query,
        "confidence": "low",
    }

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

    intent_match = _match_intent(q, intent_list)
    if intent_match:
        _, intent_code = intent_match
        result["subcategory"] = intent_code
        result["confidence"] = "high"
    else:
        result["subcategory"] = intent_list[0][1]
        result["confidence"] = "medium"

    result["number"] = _extract_number(q)
    result["section"] = _extract_section(q)
    result["sub_section"] = _extract_subsection(q)
    result["grading"] = _extract_grading(q)
    result["leave_type"] = _extract_leave_type(q, category=module)
    result["sport"] = _extract_sport(q)
    result["class"] = _extract_class(q)
    result["unit_name"] = _extract_unit_name(raw_query)
    result["attempt_no"] = _extract_attempt_no(q)
    result["from_attempt"] = _extract_from_attempt(q)
    result["to_attempt"] = _extract_to_attempt(q)
    result["date"] = _extract_date(raw_query)

    item_name, item_cat = _extract_item_query(q)
    result["item_name"] = item_name
    result["item_category"] = item_cat

    if item_cat and result.get("subcategory") in (
        "EquipmentStats",
        "OverdueEquipment",
        "ReturnedEquipment",
        None,
    ):
        result["subcategory"] = item_cat
        if result.get("confidence") != "high":
            result["confidence"] = "medium"

    if result["confidence"] == "high":
        if result["subcategory"] in ("TopPerformers", "LowestPerformers"):
            if result["number"] is None:
                result["confidence"] = "medium"

    return result


def format_admin_payload(intent_result: Dict[str, Any]) -> Dict[str, Any]:
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
