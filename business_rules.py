"""
business_rules.py
=================
Centralized business rules and constants for the AgniAI SQL execution engine.
This module ensures that logic like "Active Filters" or "Leave Definitions"
is never scattered as magic strings throughout the codebase or manually
hard-coded into prompts.
"""

from typing import Set

# ── Status Constants ────────────────────────────────────────────────────────
VERIFICATION_STATUSES: Set[str] = {
    "Pending",
    "Verified",
    "Rejected",
    "Sent",
    "NotResponded",
}

LEAVE_TYPES: Set[str] = {
    "Annual",
    "Medical",
    "Sick",
    "Hospitalized",
    "Absconded",
    "EX PPG",
    "ATTN'C",
}

GRADING_CATEGORIES: Set[str] = {
    "Fail",
    "SAT",
    "Good",
    "Excellent",
    "Exceptionally Well",
}

SECTIONS: Set[str] = {
    "BPET",
    "PPT",
    "Firing",
    "Drill",
    "Theory",
}

SUBSECTIONS_BY_SECTION: dict[str, list[str]] = {
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

# ── SQL Logic Constants ─────────────────────────────────────────────────────
# A reusable SQL string snippet that guarantees only active agniveers are processed.
ACTIVE_AGNIVEER_FILTER = "(a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)"

# ── LLM Hard Rules ──────────────────────────────────────────────────────────
# These rules are injected into the dynamic schema card to prevent the LLM from
# hallucinating arithmetic or re-inventing business rules from scratch.
LLM_HARD_RULES = """
HARD RULES — these are NON-NEGOTIABLE. Break one and the query is rejected.

R0. RAW TABLES ONLY: There are NO views in this database. You MUST NOT use or assume the existence of views like vw_AgniveerSectionGrades, vw_AgniveerBmi, etc. You MUST construct all complex logic using raw tables and standard SQL Server constructs (CTEs, subqueries). A query attempting to SELECT from a fabricated view will be rejected.
R1. ACTIVE FILTER: Every query touching AgniveerMaster MUST include the condition: `(a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)`. Exception: an explicit "disqualified agniveers" request inverts it to `(a.IsDisqualified = 1)`.
R2. SPECIAL COLUMNS: These two AgniveerLeaveMaster columns contain a space/apostrophe and MUST be written bracket-quoted EXACTLY: `[OnATTN'C']` and `[OnEX PPG]`. Never rename, alias-away, or unquote them.
R3. EXCEPTIONAL MARKS: When a section may be exceptional use:
    CASE WHEN s.IsExceptional = 1
         THEN COALESCE(r.ExceptionalMarks, 0.0)
         ELSE r.SubItemTotalMarks END
R4. CROSS-FILTER SHAPE: Each independent condition becomes a CTE that outputs DISTINCT AgniveerId; the final SELECT INNER JOINs every CTE back to AgniveerMaster on a.Id (intersection), displaying a.AgniveerNo.
R5. PROHIBITED DATA: NEVER select UserMaster.Password, LoginToken.*, or DefaultLog.
R6. AGNIVEER ID JOIN RULE: The primary key of AgniveerMaster is `a.Id`. You MUST write `ON <othertable>.AgniveerId = a.Id`. NEVER write `a.AgniveerId` - that column does not exist on AgniveerMaster!
R7. BEST ATTEMPT: You MUST use the RAW table `AgniveerScoreAttempt` and filter by `IsBestAttempt = 1` for best attempts. For a specific attempt, filter by `AttemptNo`.
R8. CURRENT LEAVE: Current leave must strictly be calculated using: `CAST(GETDATE() AS DATE) BETWEEN CAST(l.FromDate AS DATE) AND CAST(l.ToDate AS DATE)` on `AgniveerLeaveMaster`.
R9. ABSCONDED LEAVE: Absconded is detected via `IsAbscondedLeave = 1` directly in `AgniveerLeaveMaster`. Valid Leave Types are: Annual, Medical, Sick, Hospitalized, Absconded, EX PPG, ATTN'C.
R10. BMI / MEDICAL: BMI must be manually calculated by finding the average Height and Weight in `MedicalRecordMaster`, falling back to `AgniveerMaster.Height/Weight`, and applying the standard BMI formula `Weight / POWER(Height / 100.0, 2)`.
R11. VERIFICATION: You MUST use a Window Function (`ROW_NUMBER() OVER (PARTITION BY AgniveerId ORDER BY SentDate DESC)`) on `PoliceVerificationMaster` to find the latest verification status. `Pending` is when there is no record, `NotResponded` is when `Status = 'Sent'` and `ReceivedDate IS NULL`. Valid Statuses are: Pending, Verified, Rejected, Sent, NotResponded.
R12. EQUIPMENT: For currently held equipment, check `ReturnDateTime IS NULL` on `AgniveerEquipment`. For degraded/damaged equipment, manually compare `GivenCondition` against `ReturnCondition` integer ranks (Good=4, Fair=3, Poor=2, Damaged=1).
R13. ATTENDANCE: When querying `AgniveerAttendanceMaster`, you MUST manually override `IsPresent` to 0 if an overlapping leave record exists in `AgniveerLeaveMaster` for that date, using an `EXISTS` subquery.
R14. DISTRIBUTION: The CURRENT distribution is strictly stored in `AgniveerRelationMaster`. Do NOT use `DistributionHistoryMaster` or window functions to find the "latest" distribution. `AgniveerRelationMaster` is the source of truth. Unassigned Agniveers MUST be found via a LEFT JOIN on `AgniveerRelationMaster` where `DistributionId IS NULL`.
R15. ROSTER & SKILLS: `Skill`, `Sports`, `Class`, and `Qualification` are simple string columns on `AgniveerMaster`. Do NOT invent mapping tables like `AgniveerSkillMapping` or perform joins to find skills.
R16. HINT FILTERS: If the CLASSIFIER HINT contains 'batch_id' or 'batchId', you MUST filter `a.BatchId = <value>`. If it contains 'platoon_id' or 'platoonId', filter `a.PlatoonId = <value>`. If it contains 'company_id' or 'companyId', you MUST join `PlatoonMaster p ON a.PlatoonId = p.Id` (if not already joined) and filter `p.CompanyId = <value>`.
R17. GRADES: The only valid Grades are: Fail, SAT, Good, Excellent, Exceptionally Well.
R18. SECTIONS AND SUBSECTIONS: The valid sections are BPET, PPT, Firing, Drill, Theory. Do not hallucinate section names. BPET includes 5km, 60m Sprint, 9' Ditch, H Rope, V Rope. PPT includes 2.4km, 100m Sprint, Chin Ups, Sit Ups, Toe Touch, 5m Shuttle. Firing includes 300m 5 FTS, 200m 5 FTS, 100m LU (R), 100m LU (5/5), 50m SU TRU, 100m NI SFTS SS, 15m (Bal) BC. Drill includes Turn Out and Bearing, Word of Command, Marching, Slow March, Saluting Salami Shastra, Turning.
""".strip()
