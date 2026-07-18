"""
sql_executor.py
===============
Text-to-SQL execution backend for AgniAI (DB_Agni).

This is the *swappable middle* of the pipeline. It occupies the same slot as
`dotnet_executor._call_dotnet` and returns the SAME contract:

    execute_sql_query(...) -> Tuple[data, error]
        (rows, None)  on success
        (None, msg)   on failure

so it drops into the existing call sites in admin_pipeline.py. Everything
downstream (result_combiner, report_generator grounding, widget_engine,
suggested_question_engine) is reused unchanged — this module only replaces
"how the data is fetched", not "how the answer is built".

Flow (as actually wired in execute_sql_query today):
    intent -> [golden fast-path if GOLDEN_QUERIES has this (category,
    operation) AND every filter in the intent is one the golden template
    can express — see _golden_query_can_satisfy] -> otherwise
    query_planner_v2.plan_query() builds a deterministic AST from the
    intent's filters, which sql_builder.build() compiles to SQL
    -> sql_validator.validate_sql() (defense-in-depth gate)
    -> run read-only -> rows.

`generate_sql()` below (free-text LLM SQL generation carrying
business_rules.LLM_HARD_RULES) is NOT part of this flow — execute_sql_query
never calls it, and nothing else in this codebase does either. It predates
the query_planner_v2 AST path and is currently reachable only from tests
that patch it directly. Treat it as available-but-unwired scaffolding, not
a live fallback, until something actually calls it.

SAFETY MODEL (do not weaken any of these):
  1. Connect with a READ-ONLY SQL login (db_datareader only, DENY on
     UserMaster.Password / LoginToken). The login is the real wall; the
     validator below is defense-in-depth, not the primary control.
  2. Only a single SELECT / WITH...SELECT statement is ever executed.
  3. Column/table allowlist derived from the schema card. Sensitive columns
     are hard-denied even if a grant slips — enforced in sql_validator.py's
     validate_sql(), the validator execute_sql_query actually calls (this
     module's own validate_sql() below is not on the live path — see its
     docstring).
  4. Row cap (SET ROWCOUNT) + command timeout on every execution.
  5. Every number in the final narrative is grounded downstream by
     grounding_utils.ground_and_sanitize against these rows.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Config (env-overridable, production-safe defaults) ──────────────────────
SQL_SERVER_2008_COMPAT = os.getenv("SQL_2008_COMPAT", "1").strip() == "1"
SQL_MAX_ROWS = int(os.getenv("SQL_MAX_ROWS", "500"))
SQL_COMMAND_TIMEOUT_S = int(os.getenv("SQL_COMMAND_TIMEOUT_S", "15"))
# Read-only connection string. MUST point at a db_datareader-only login.
SQL_READONLY_CONN = os.getenv("SQL_READONLY_CONN", "")

# ── Hard denylist: columns the SQL layer must NEVER return ─────────────────
DENIED_COLUMNS = {
    "usermaster.password",
    "logintoken.token",
    "logintoken.refreshtoken",
}
DENIED_TABLES = {"logintoken", "defaultlog"}



# Golden fast-path: (category, operation) -> parameterized SQL. `subcategory`
# is deliberately not part of the key — it's fully derived from (category,
# operation) via CATEGORY_OPERATION_TO_SUBCATEGORY in intent_schema.py, so
# keying on it too was redundant and, in practice, error-prone (a typo'd
# subcategory string silently made an entry unreachable with no error).
# Fill this by porting AiCommandService.cs handlers over time. Anything found
# here SKIPS the LLM entirely (deterministic, cheap, testable).
#
# Templates may contain the single placeholder {top_n} — the only thing ever
# substituted into a golden query (see _render_golden_query below), and only
# with a validated integer, so this can never introduce SQL injection.
GOLDEN_QUERIES: Dict[Tuple[str, str], str] = {
    ("Performance", "Top"): """
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, SUM(sa.MarksObtained) AS TotalMarks
FROM AgniveerMaster a
INNER JOIN AgniveerScoreAttempt sa ON sa.AgniveerId = a.Id
INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId
INNER JOIN ScoreSectionMaster sec ON sec.Id = si.SectionId
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
  AND sa.IsBestAttempt = 1
  AND (sec.IsExceptional <> 1 OR sec.IsExceptional IS NULL)
  {section_clause}
GROUP BY a.AgniveerNo, a.FullName
ORDER BY SUM(sa.MarksObtained) DESC, a.AgniveerNo ASC
""".strip(),
    ("Performance", "Bottom"): """
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, SUM(sa.MarksObtained) AS TotalMarks
FROM AgniveerMaster a
INNER JOIN AgniveerScoreAttempt sa ON sa.AgniveerId = a.Id
INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId
INNER JOIN ScoreSectionMaster sec ON sec.Id = si.SectionId
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
  AND sa.IsBestAttempt = 1
  AND (sec.IsExceptional <> 1 OR sec.IsExceptional IS NULL)
  {section_clause}
GROUP BY a.AgniveerNo, a.FullName
ORDER BY SUM(sa.MarksObtained) ASC, a.AgniveerNo ASC
""".strip(),
    ("Performance", "Average"): """
SELECT sec.SectionName, AVG(bt.BestTotal) AS AverageMarks, COUNT(DISTINCT a.Id) AS AgniveerCount
FROM (
    SELECT sa.AgniveerId, si.SectionId, SUM(sa.MarksObtained) AS BestTotal
    FROM AgniveerScoreAttempt sa
    INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId
    WHERE sa.IsBestAttempt = 1
    GROUP BY sa.AgniveerId, si.SectionId
) bt
    INNER JOIN ScoreSectionMaster sec ON sec.Id = bt.SectionId
    INNER JOIN AgniveerMaster a ON a.Id = bt.AgniveerId
    WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
      AND a.IsActive = 1
      AND (sec.IsExceptional <> 1 OR sec.IsExceptional IS NULL)
      {section_clause}
GROUP BY sec.SectionName
ORDER BY AVG(bt.BestTotal) DESC, sec.SectionName ASC
""".strip(),
    ("Attendance", "Summary"): """
SELECT a.AgniveerNo, a.FullName,
       SUM(CASE
           WHEN EXISTS (
               SELECT 1 FROM AgniveerLeaveMaster l
               WHERE l.AgniveerId = att.AgniveerId
                 AND l.FromDate IS NOT NULL
                 AND CAST(att.AttendanceDateTime AS DATE) >= CAST(l.FromDate AS DATE)
                 AND (l.ToDate IS NULL OR CAST(att.AttendanceDateTime AS DATE) <= CAST(l.ToDate AS DATE))
           ) THEN 0
           WHEN att.IsPresent = 1 THEN 1
           ELSE 0
       END) AS PresentDays,
       COUNT(att.AttendanceDateTime) AS TotalDays
FROM AgniveerMaster a
INNER JOIN AgniveerAttendanceMaster att ON att.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
GROUP BY a.AgniveerNo, a.FullName
ORDER BY a.AgniveerNo
""".strip(),
    ("Leave", "Current"): """
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, l.FromDate, l.ToDate,
    CASE WHEN l.[OnEX PPG] = 1 THEN 'EX PPG' WHEN l.[OnATTN''C'] = 1 THEN 'ATTN''C' WHEN l.OnAnnualLeave = 1 THEN 'Annual' WHEN l.OnMedicalLeave = 1 THEN 'Medical' WHEN l.OnSickLeave = 1 THEN 'Sick' WHEN l.IsHospitalized = 1 THEN 'Hospitalized' WHEN l.IsAbscondedLeave = 1 THEN 'Absconded' ELSE 'Other' END AS LeaveType
FROM AgniveerMaster a
INNER JOIN AgniveerLeaveMaster l ON l.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
  AND l.FromDate IS NOT NULL AND l.ToDate IS NOT NULL
  AND CAST(GETDATE() AS DATE) BETWEEN CAST(l.FromDate AS DATE) AND CAST(l.ToDate AS DATE)
ORDER BY l.FromDate DESC, a.AgniveerNo ASC
""".strip(),
    ("Leave", "Most"): """
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, SUM(CASE WHEN l.[OnEX PPG] = 1 THEN (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) / 4 ELSE (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) END) AS TotalLeaveDays
FROM AgniveerMaster a
INNER JOIN AgniveerLeaveMaster l ON l.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
  AND l.FromDate IS NOT NULL AND l.ToDate IS NOT NULL
GROUP BY a.AgniveerNo, a.FullName
ORDER BY SUM(CASE WHEN l.[OnEX PPG] = 1 THEN (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) / 4 ELSE (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) END) DESC, a.AgniveerNo ASC
""".strip(),
    ("Leave", "Least"): """
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, SUM(CASE WHEN l.[OnEX PPG] = 1 THEN (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) / 4 ELSE (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) END) AS TotalLeaveDays
FROM AgniveerMaster a
INNER JOIN AgniveerLeaveMaster l ON l.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
  AND l.FromDate IS NOT NULL AND l.ToDate IS NOT NULL
GROUP BY a.AgniveerNo, a.FullName
ORDER BY SUM(CASE WHEN l.[OnEX PPG] = 1 THEN (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) / 4 ELSE (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) END) ASC, a.AgniveerNo ASC
""".strip(),
    ("Leave", "Absconded"): """
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, l.FromDate, l.ToDate,
    CASE WHEN l.[OnEX PPG] = 1 THEN (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) / 4 ELSE (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) END AS LeaveCount
FROM AgniveerMaster a
INNER JOIN AgniveerLeaveMaster l ON l.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
  AND l.FromDate IS NOT NULL AND l.ToDate IS NOT NULL
  AND l.IsAbscondedLeave = 1
ORDER BY l.FromDate DESC
""".strip(),
    ("Medical", "Disease"): """
SELECT TOP ({top_n}) LTRIM(RTRIM(m.Diagnosis)) AS Diagnosis, COUNT(DISTINCT m.AgniveerId) AS AffectedCount
FROM MedicalRecordMaster m
INNER JOIN AgniveerMaster a ON a.Id = m.AgniveerId
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
  AND m.Diagnosis IS NOT NULL AND LTRIM(RTRIM(m.Diagnosis)) <> ''
GROUP BY LTRIM(RTRIM(m.Diagnosis))
ORDER BY COUNT(DISTINCT m.AgniveerId) DESC, LTRIM(RTRIM(m.Diagnosis)) ASC
""".strip(),
    ("Medical", "BMI"): """
WITH MedicalAvg AS (
    SELECT AgniveerId, AVG(Height) AS AvgHeight, AVG(Weight) AS AvgWeight
    FROM MedicalRecordMaster
    WHERE Height IS NOT NULL AND Weight IS NOT NULL
    GROUP BY AgniveerId
),
Resolved AS (
    SELECT
        a.Id AS AgniveerId,
        COALESCE(m.AvgHeight, a.Height) AS AvgHeight,
        COALESCE(m.AvgWeight, a.Weight) AS AvgWeight
    FROM AgniveerMaster a
    LEFT JOIN MedicalAvg m ON m.AgniveerId = a.Id
    WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
      AND a.IsActive = 1
),
BmiData AS (
    SELECT AgniveerId,
        CASE
            WHEN AvgHeight IS NULL OR AvgWeight IS NULL OR AvgHeight <= 0 THEN NULL
            WHEN AvgWeight / POWER(AvgHeight / 100.0, 2) < 18.5 THEN 'Underweight'
            WHEN AvgWeight / POWER(AvgHeight / 100.0, 2) < 25   THEN 'Normal'
            WHEN AvgWeight / POWER(AvgHeight / 100.0, 2) < 30   THEN 'Overweight'
            ELSE 'Obese'
        END AS BmiCategory
    FROM Resolved
)
SELECT BmiCategory, COUNT(DISTINCT AgniveerId) AS AgniveerCount
FROM BmiData
WHERE BmiCategory IS NOT NULL
GROUP BY BmiCategory
ORDER BY COUNT(DISTINCT AgniveerId) DESC, BmiCategory ASC
""".strip(),
    ("Medical", "BloodGroup"): """
SELECT a.BloodGroup, COUNT(DISTINCT a.Id) AS AgniveerCount
FROM AgniveerMaster a
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
  AND a.BloodGroup IS NOT NULL
GROUP BY a.BloodGroup
ORDER BY COUNT(DISTINCT a.Id) DESC, a.BloodGroup ASC
""".strip(),
    ("Verification", "Pending"): """
WITH Latest AS (
    SELECT AgniveerId, PoliceStation, SentDate, ReceivedDate, Status,
        ROW_NUMBER() OVER (PARTITION BY AgniveerId ORDER BY SentDate DESC, Id DESC) AS rn
    FROM PoliceVerificationMaster
)
SELECT a.AgniveerNo, a.FullName, l.PoliceStation, l.SentDate,
    CASE WHEN l.AgniveerId IS NULL THEN 'Pending'
         WHEN l.Status = 'Sent' AND l.ReceivedDate IS NULL THEN 'NotResponded'
         ELSE l.Status END AS Status
FROM AgniveerMaster a
LEFT JOIN Latest l ON l.AgniveerId = a.Id AND l.rn = 1
WHERE (l.AgniveerId IS NULL OR (CASE WHEN l.Status = 'Sent' AND l.ReceivedDate IS NULL THEN 'NotResponded' ELSE l.Status END) = 'Pending')
ORDER BY l.SentDate ASC, a.AgniveerNo ASC
""".strip(),
    ("Verification", "Verified"): """
WITH Latest AS (
    SELECT AgniveerId, PoliceStation, SentDate, ReceivedDate, Status,
        ROW_NUMBER() OVER (PARTITION BY AgniveerId ORDER BY SentDate DESC, Id DESC) AS rn
    FROM PoliceVerificationMaster
)
SELECT a.AgniveerNo, a.FullName, l.PoliceStation, l.SentDate, l.ReceivedDate, l.Status
FROM AgniveerMaster a
INNER JOIN Latest l ON l.AgniveerId = a.Id AND l.rn = 1
WHERE l.Status = 'Verified'
ORDER BY l.ReceivedDate DESC, a.AgniveerNo ASC
""".strip(),
    ("Verification", "Rejected"): """
WITH Latest AS (
    SELECT AgniveerId, PoliceStation, SentDate, ReceivedDate, Status,
        ROW_NUMBER() OVER (PARTITION BY AgniveerId ORDER BY SentDate DESC, Id DESC) AS rn
    FROM PoliceVerificationMaster
)
SELECT a.AgniveerNo, a.FullName, l.PoliceStation, l.SentDate, l.ReceivedDate, l.Status
FROM AgniveerMaster a
INNER JOIN Latest l ON l.AgniveerId = a.Id AND l.rn = 1
WHERE l.Status = 'Rejected'
ORDER BY l.ReceivedDate DESC, a.AgniveerNo ASC
""".strip(),
    ("Verification", "Sent"): """
WITH Latest AS (
    SELECT AgniveerId, PoliceStation, SentDate, ReceivedDate, Status,
        ROW_NUMBER() OVER (PARTITION BY AgniveerId ORDER BY SentDate DESC, Id DESC) AS rn
    FROM PoliceVerificationMaster
)
SELECT a.AgniveerNo, a.FullName, l.PoliceStation, l.SentDate, l.Status
FROM AgniveerMaster a
INNER JOIN Latest l ON l.AgniveerId = a.Id AND l.rn = 1
WHERE l.Status = 'Sent'
ORDER BY l.SentDate DESC, a.AgniveerNo ASC
""".strip(),
    ("Verification", "NotResponded"): """
WITH Latest AS (
    SELECT AgniveerId, PoliceStation, SentDate, ReceivedDate, Status,
        ROW_NUMBER() OVER (PARTITION BY AgniveerId ORDER BY SentDate DESC, Id DESC) AS rn
    FROM PoliceVerificationMaster
)
SELECT a.AgniveerNo, a.FullName, l.PoliceStation, l.SentDate,
    CASE WHEN l.SentDate IS NOT NULL THEN DATEDIFF(DAY, l.SentDate, GETDATE()) ELSE NULL END AS DaysSinceSent,
    'NotResponded' AS Status
FROM AgniveerMaster a
INNER JOIN Latest l ON l.AgniveerId = a.Id AND l.rn = 1
WHERE l.Status = 'Sent' AND l.ReceivedDate IS NULL
ORDER BY CASE WHEN l.SentDate IS NOT NULL THEN DATEDIFF(DAY, l.SentDate, GETDATE()) ELSE NULL END DESC, a.AgniveerNo ASC
""".strip(),
    ("Equipment", "Holding"): """
SELECT a.AgniveerNo, a.FullName, e.Name AS EquipmentName, ae.GivenDateTime, ae.GivenCondition
FROM AgniveerMaster a
INNER JOIN AgniveerEquipment ae ON ae.AgniveerId = a.Id
INNER JOIN EquipmentMaster e ON e.Id = ae.EquipmentId
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
  AND ae.ReturnDateTime IS NULL
ORDER BY ae.GivenDateTime DESC, a.AgniveerNo ASC
""".strip(),
    ("Equipment", "Returned"): """
WITH Degraded AS (
    SELECT ae.AgniveerId, e.Name AS EquipmentName
    FROM AgniveerEquipment ae
    INNER JOIN EquipmentMaster e ON e.Id = ae.EquipmentId
    WHERE ae.ReturnDateTime IS NOT NULL
      AND (CASE UPPER(ae.GivenCondition) WHEN 'GOOD' THEN 4 WHEN 'FAIR' THEN 3 WHEN 'POOR' THEN 2 WHEN 'DAMAGED' THEN 1 ELSE NULL END) >
          (CASE UPPER(ae.ReturnCondition) WHEN 'GOOD' THEN 4 WHEN 'FAIR' THEN 3 WHEN 'POOR' THEN 2 WHEN 'DAMAGED' THEN 1 ELSE NULL END)
)
SELECT a.AgniveerNo, a.FullName, v.EquipmentName
FROM AgniveerMaster a
INNER JOIN Degraded v ON v.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
ORDER BY a.AgniveerNo
""".strip(),
    ("Attendance", "Present"): """
SELECT a.AgniveerNo, a.FullName, p.Name AS Platoon, c.Name AS Company
FROM AgniveerMaster a
INNER JOIN (
    SELECT att.AgniveerId, CAST(att.AttendanceDateTime AS DATE) AS [Date],
        CASE WHEN EXISTS (SELECT 1 FROM AgniveerLeaveMaster l WHERE l.AgniveerId = att.AgniveerId AND l.FromDate IS NOT NULL AND CAST(att.AttendanceDateTime AS DATE) >= CAST(l.FromDate AS DATE) AND (l.ToDate IS NULL OR CAST(att.AttendanceDateTime AS DATE) <= CAST(l.ToDate AS DATE))) THEN 0 ELSE att.IsPresent END AS IsPresent
    FROM AgniveerAttendanceMaster att
) att ON att.AgniveerId = a.Id
LEFT JOIN PlatoonMaster p ON a.PlatoonId = p.Id
LEFT JOIN CompanyMaster c ON p.CompanyId = c.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
  AND att.IsPresent = 1
  AND att.Date = CAST(GETDATE() AS DATE)
ORDER BY a.AgniveerNo ASC
""".strip(),
    ("Attendance", "Monthly"): """
SELECT CAST(YEAR(att.Date) AS VARCHAR) + '-' + RIGHT('0' + CAST(MONTH(att.Date) AS VARCHAR), 2) AS MonthYear,
       SUM(CASE WHEN att.IsPresent = 1 THEN 1 ELSE 0 END) AS TotalPresent,
       COUNT(att.Date) AS TotalDays
FROM (
    SELECT att.AgniveerId, CAST(att.AttendanceDateTime AS DATE) AS [Date],
        CASE WHEN EXISTS (SELECT 1 FROM AgniveerLeaveMaster l WHERE l.AgniveerId = att.AgniveerId AND l.FromDate IS NOT NULL AND CAST(att.AttendanceDateTime AS DATE) >= CAST(l.FromDate AS DATE) AND (l.ToDate IS NULL OR CAST(att.AttendanceDateTime AS DATE) <= CAST(l.ToDate AS DATE))) THEN 0 ELSE att.IsPresent END AS IsPresent
    FROM AgniveerAttendanceMaster att
) att
INNER JOIN AgniveerMaster a ON a.Id = att.AgniveerId
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
GROUP BY YEAR(att.Date), MONTH(att.Date)
ORDER BY YEAR(att.Date) DESC, MONTH(att.Date) DESC
""".strip(),
    ("Attendance", "Weekly"): """
SELECT YEAR(att.Date) AS Year, DATEPART(week, att.Date) AS WeekNumber,
       SUM(CASE WHEN att.IsPresent = 1 THEN 1 ELSE 0 END) AS TotalPresent,
       COUNT(att.Date) AS TotalDays
FROM (
    SELECT att.AgniveerId, CAST(att.AttendanceDateTime AS DATE) AS [Date],
        CASE WHEN EXISTS (SELECT 1 FROM AgniveerLeaveMaster l WHERE l.AgniveerId = att.AgniveerId AND l.FromDate IS NOT NULL AND CAST(att.AttendanceDateTime AS DATE) >= CAST(l.FromDate AS DATE) AND (l.ToDate IS NULL OR CAST(att.AttendanceDateTime AS DATE) <= CAST(l.ToDate AS DATE))) THEN 0 ELSE att.IsPresent END AS IsPresent
    FROM AgniveerAttendanceMaster att
) att
INNER JOIN AgniveerMaster a ON a.Id = att.AgniveerId
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
GROUP BY YEAR(att.Date), DATEPART(week, att.Date)
ORDER BY YEAR(att.Date) DESC, DATEPART(week, att.Date) DESC
""".strip(),
    ("Attendance", "Daily"): """
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, att.Date, att.IsPresent
FROM AgniveerMaster a
INNER JOIN (
    SELECT att.AgniveerId, CAST(att.AttendanceDateTime AS DATE) AS [Date],
        CASE WHEN EXISTS (SELECT 1 FROM AgniveerLeaveMaster l WHERE l.AgniveerId = att.AgniveerId AND l.FromDate IS NOT NULL AND CAST(att.AttendanceDateTime AS DATE) >= CAST(l.FromDate AS DATE) AND (l.ToDate IS NULL OR CAST(att.AttendanceDateTime AS DATE) <= CAST(l.ToDate AS DATE))) THEN 0 ELSE att.IsPresent END AS IsPresent
    FROM AgniveerAttendanceMaster att
) att ON att.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
  AND att.Date = CAST(GETDATE() AS DATE)
ORDER BY a.AgniveerNo ASC
""".strip(),
    ("Distribution", "Latest"): """
SELECT d.Name AS DistributionName, COUNT(DISTINCT r.AgniveerId) AS AgniveerCount
FROM DistributionMaster d
INNER JOIN AgniveerRelationMaster r ON r.DistributionId = d.Id
INNER JOIN AgniveerMaster a ON a.Id = r.AgniveerId
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
GROUP BY d.Name
ORDER BY COUNT(DISTINCT r.AgniveerId) DESC, d.Name ASC
""".strip(),
    ("Distribution", "ByUnit"): """
SELECT a.AgniveerNo, a.FullName, d.Name AS DistributionName
FROM AgniveerMaster a
INNER JOIN AgniveerRelationMaster r ON a.Id = r.AgniveerId
INNER JOIN DistributionMaster d ON r.DistributionId = d.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
ORDER BY d.Name ASC, a.AgniveerNo ASC
""".strip(),
    ("Distribution", "Unassigned"): """
SELECT a.AgniveerNo, a.FullName, p.Name AS Platoon, c.Name AS Company
FROM AgniveerMaster a
LEFT JOIN AgniveerRelationMaster r ON a.Id = r.AgniveerId
LEFT JOIN PlatoonMaster p ON a.PlatoonId = p.Id
LEFT JOIN CompanyMaster c ON p.CompanyId = c.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND r.AgniveerId IS NULL
ORDER BY a.AgniveerNo ASC
""".strip(),
    ("Skills", "BySport"): """
SELECT a.AgniveerNo, a.FullName, p.Name AS Platoon, c.Name AS Company, a.Sports
FROM AgniveerMaster a
LEFT JOIN PlatoonMaster p ON a.PlatoonId = p.Id
LEFT JOIN CompanyMaster c ON p.CompanyId = c.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
  AND a.Sports IS NOT NULL AND a.Sports <> ''
ORDER BY a.Sports ASC, a.AgniveerNo ASC
""".strip(),
    ("Skills", "ByClass"): """
SELECT a.AgniveerNo, a.FullName, p.Name AS Platoon, c.Name AS Company, a.Class
FROM AgniveerMaster a
LEFT JOIN PlatoonMaster p ON a.PlatoonId = p.Id
LEFT JOIN CompanyMaster c ON p.CompanyId = c.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
  AND a.Class IS NOT NULL AND a.Class <> ''
ORDER BY a.Class ASC, a.AgniveerNo ASC
""".strip(),
    ("Performance", "BestAttempt"): """
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, sec.SectionName AS Section, SUM(sa.MarksObtained) AS BestTotal
FROM AgniveerMaster a
INNER JOIN AgniveerScoreAttempt sa ON sa.AgniveerId = a.Id
INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId
INNER JOIN ScoreSectionMaster sec ON sec.Id = si.SectionId
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
  AND sa.IsBestAttempt = 1
  AND (sec.IsExceptional <> 1 OR sec.IsExceptional IS NULL)
GROUP BY a.AgniveerNo, a.FullName, sec.SectionName
ORDER BY SUM(sa.MarksObtained) DESC, a.AgniveerNo ASC
""".strip(),
    ("Performance", "Grading"): """
WITH AttemptedMax AS (
    SELECT DISTINCT sa.AgniveerId, si.SectionId, si.Id AS SubItemId, si.MaxMarks
    FROM AgniveerScoreAttempt sa
    INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId
),
DynamicMax AS (
    SELECT AgniveerId, SectionId, SUM(MaxMarks) AS DynamicMax
    FROM AttemptedMax
    GROUP BY AgniveerId, SectionId
),
BestTotals AS (
    SELECT sa.AgniveerId, si.SectionId, SUM(sa.MarksObtained) AS BestTotal
    FROM AgniveerScoreAttempt sa
    INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId
    WHERE sa.IsBestAttempt = 1
    GROUP BY sa.AgniveerId, si.SectionId
),
Scored AS (
    SELECT bt.AgniveerId, sec.SectionName, bt.BestTotal,
        CASE WHEN dm.DynamicMax > 0 THEN 100.0 * bt.BestTotal / dm.DynamicMax ELSE NULL END AS Percentage,
        CASE WHEN dm.DynamicMax IS NULL OR dm.DynamicMax = 0 THEN NULL
             WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 90 THEN 'Exceptionally Well'
             WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 75 THEN 'Excellent'
             WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 60 THEN 'Good'
             WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 45 THEN 'SAT'
             ELSE 'Fail' END AS Grade
    FROM BestTotals bt
    INNER JOIN ScoreSectionMaster sec ON sec.Id = bt.SectionId
    LEFT JOIN DynamicMax dm ON dm.AgniveerId = bt.AgniveerId AND dm.SectionId = bt.SectionId
    WHERE (sec.IsExceptional <> 1 OR sec.IsExceptional IS NULL)
)
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, sg.SectionName, sg.Grade, sg.Percentage, sg.BestTotal
FROM AgniveerMaster a
INNER JOIN Scored sg ON sg.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
ORDER BY sg.Percentage DESC, a.AgniveerNo ASC
""".strip(),
    ("Performance", "GradingSummary"): """
WITH AttemptedMax AS (
    SELECT DISTINCT sa.AgniveerId, si.SectionId, si.Id AS SubItemId, si.MaxMarks
    FROM AgniveerScoreAttempt sa
    INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId
),
DynamicMax AS (
    SELECT AgniveerId, SectionId, SUM(MaxMarks) AS DynamicMax
    FROM AttemptedMax
    GROUP BY AgniveerId, SectionId
),
BestTotals AS (
    SELECT sa.AgniveerId, si.SectionId, SUM(sa.MarksObtained) AS BestTotal
    FROM AgniveerScoreAttempt sa
    INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId
    WHERE sa.IsBestAttempt = 1
    GROUP BY sa.AgniveerId, si.SectionId
),
Scored AS (
    SELECT bt.AgniveerId, sec.SectionName,
        CASE WHEN dm.DynamicMax IS NULL OR dm.DynamicMax = 0 THEN NULL
             WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 90 THEN 'Exceptionally Well'
             WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 75 THEN 'Excellent'
             WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 60 THEN 'Good'
             WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 45 THEN 'SAT'
             ELSE 'Fail' END AS Grade
    FROM BestTotals bt
    INNER JOIN ScoreSectionMaster sec ON sec.Id = bt.SectionId
    LEFT JOIN DynamicMax dm ON dm.AgniveerId = bt.AgniveerId AND dm.SectionId = bt.SectionId
    WHERE (sec.IsExceptional <> 1 OR sec.IsExceptional IS NULL)
)
SELECT sg.SectionName, sg.Grade, COUNT(DISTINCT a.Id) AS AgniveerCount
FROM AgniveerMaster a
INNER JOIN Scored sg ON sg.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
GROUP BY sg.SectionName, sg.Grade
ORDER BY sg.SectionName, sg.Grade
""".strip(),
    ("Performance", "Improvement"): """
WITH AttemptTotals AS (
    SELECT sa.AgniveerId, sa.AttemptNo, SUM(sa.MarksObtained) AS TotalMarks
    FROM AgniveerScoreAttempt sa
    INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId
    INNER JOIN ScoreSectionMaster sec ON sec.Id = si.SectionId
    WHERE (sec.IsExceptional <> 1 OR sec.IsExceptional IS NULL)
    GROUP BY sa.AgniveerId, sa.AttemptNo
),
MinMaxAttempts AS (
    SELECT AgniveerId, MIN(AttemptNo) as MinAttempt, MAX(AttemptNo) as MaxAttempt
    FROM AttemptTotals
    GROUP BY AgniveerId
    HAVING MIN(AttemptNo) < MAX(AttemptNo)
),
ImprovementData AS (
    SELECT m.AgniveerId, t1.TotalMarks AS FromTotal, t2.TotalMarks AS ToTotal,
           (t2.TotalMarks - t1.TotalMarks) AS Improvement
    FROM MinMaxAttempts m
    INNER JOIN AttemptTotals t1 ON t1.AgniveerId = m.AgniveerId AND t1.AttemptNo = m.MinAttempt
    INNER JOIN AttemptTotals t2 ON t2.AgniveerId = m.AgniveerId AND t2.AttemptNo = m.MaxAttempt
)
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, i.FromTotal, i.ToTotal, i.Improvement,
       CASE WHEN i.FromTotal > 0 THEN (i.Improvement * 100.0) / i.FromTotal ELSE NULL END AS ImprovementPercentage
FROM AgniveerMaster a
INNER JOIN ImprovementData i ON i.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
  AND i.Improvement > 0
ORDER BY i.Improvement DESC, a.AgniveerNo ASC
""".strip(),
    ("Performance", "Drop"): """
WITH AttemptTotals AS (
    SELECT sa.AgniveerId, sa.AttemptNo, SUM(sa.MarksObtained) AS TotalMarks
    FROM AgniveerScoreAttempt sa
    INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId
    INNER JOIN ScoreSectionMaster sec ON sec.Id = si.SectionId
    WHERE (sec.IsExceptional <> 1 OR sec.IsExceptional IS NULL)
    GROUP BY sa.AgniveerId, sa.AttemptNo
),
MinMaxAttempts AS (
    SELECT AgniveerId, MIN(AttemptNo) as MinAttempt, MAX(AttemptNo) as MaxAttempt
    FROM AttemptTotals
    GROUP BY AgniveerId
    HAVING MIN(AttemptNo) < MAX(AttemptNo)
),
DropData AS (
    SELECT m.AgniveerId, t1.TotalMarks AS FromTotal, t2.TotalMarks AS ToTotal,
           (t1.TotalMarks - t2.TotalMarks) AS ScoreDrop
    FROM MinMaxAttempts m
    INNER JOIN AttemptTotals t1 ON t1.AgniveerId = m.AgniveerId AND t1.AttemptNo = m.MinAttempt
    INNER JOIN AttemptTotals t2 ON t2.AgniveerId = m.AgniveerId AND t2.AttemptNo = m.MaxAttempt
)
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, d.FromTotal, d.ToTotal, d.ScoreDrop
FROM AgniveerMaster a
INNER JOIN DropData d ON d.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
  AND d.ScoreDrop > 0
ORDER BY d.ScoreDrop DESC, a.AgniveerNo ASC
""".strip(),
    ("Performance", "AttemptWise"): """
WITH AttemptTotals AS (
    SELECT sa.AgniveerId, sa.AttemptNo, SUM(sa.MarksObtained) AS TotalMarks
    FROM AgniveerScoreAttempt sa
    INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId
    INNER JOIN ScoreSectionMaster sec ON sec.Id = si.SectionId
    WHERE (sec.IsExceptional <> 1 OR sec.IsExceptional IS NULL)
    GROUP BY sa.AgniveerId, sa.AttemptNo
),
MaxTotals AS (
    SELECT AgniveerId, MAX(TotalMarks) AS MaxTotal
    FROM AttemptTotals
    GROUP BY AgniveerId
)
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, t.AttemptNo, t.TotalMarks
FROM AgniveerMaster a
INNER JOIN AttemptTotals t ON t.AgniveerId = a.Id
INNER JOIN MaxTotals mt ON mt.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
ORDER BY mt.MaxTotal DESC, a.AgniveerNo ASC, t.AttemptNo ASC
""".strip(),
    ("Performance", "Trend"): """
WITH AttemptedMax AS (
    SELECT DISTINCT sa.AgniveerId, sa.AttemptNo, si.Id AS SubItemId, si.MaxMarks
    FROM AgniveerScoreAttempt sa
    INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId
    INNER JOIN ScoreSectionMaster sec ON sec.Id = si.SectionId
    WHERE (sec.IsExceptional <> 1 OR sec.IsExceptional IS NULL)
),
DynamicMaxPerAttempt AS (
    SELECT AgniveerId, AttemptNo, SUM(MaxMarks) AS DynamicMax
    FROM AttemptedMax
    GROUP BY AgniveerId, AttemptNo
),
TotalsPerAttempt AS (
    SELECT sa.AgniveerId, sa.AttemptNo, SUM(sa.MarksObtained) AS TotalObtained
    FROM AgniveerScoreAttempt sa
    INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId
    INNER JOIN ScoreSectionMaster sec ON sec.Id = si.SectionId
    WHERE (sec.IsExceptional <> 1 OR sec.IsExceptional IS NULL)
    GROUP BY sa.AgniveerId, sa.AttemptNo
),
Percentages AS (
    SELECT t.AgniveerId, t.AttemptNo, t.TotalObtained, d.DynamicMax,
           CASE WHEN d.DynamicMax > 0 THEN (t.TotalObtained * 100.0) / d.DynamicMax ELSE NULL END AS Pct
    FROM TotalsPerAttempt t
    INNER JOIN DynamicMaxPerAttempt d ON d.AgniveerId = t.AgniveerId AND d.AttemptNo = t.AttemptNo
)
SELECT p.AttemptNo, AVG(p.TotalObtained) AS AverageMarks, AVG(p.Pct) AS AveragePercentage, COUNT(DISTINCT p.AgniveerId) AS AgniveerCount
FROM Percentages p
INNER JOIN AgniveerMaster a ON a.Id = p.AgniveerId
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
GROUP BY p.AttemptNo
ORDER BY p.AttemptNo ASC
""".strip(),
    ("Overall", "OverallPerformance"): """
WITH BestTotals AS (
    SELECT sa.AgniveerId, SUM(sa.MarksObtained) AS BestTotal
    FROM AgniveerScoreAttempt sa
    INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId
    INNER JOIN ScoreSectionMaster sec ON sec.Id = si.SectionId
    WHERE sa.IsBestAttempt = 1
      AND (sec.IsExceptional <> 1 OR sec.IsExceptional IS NULL)
    GROUP BY sa.AgniveerId
),
MedicalVisits AS (
    SELECT AgniveerId, COUNT(*) AS VisitCount
    FROM MedicalRecordMaster
    GROUP BY AgniveerId
),
LeaveDays AS (
    SELECT AgniveerId,
           SUM(CASE WHEN [OnEX PPG] = 1 THEN (DATEDIFF(DAY, FromDate, ToDate) + 1) / 4
                    ELSE (DATEDIFF(DAY, FromDate, ToDate) + 1) END) AS TotalLeaveDays
    FROM AgniveerLeaveMaster
    WHERE FromDate IS NOT NULL AND ToDate IS NOT NULL
    GROUP BY AgniveerId
),
Bounds AS (
    SELECT MAX(bt.BestTotal) AS MaxScore,
           MAX(ISNULL(mv.VisitCount, 0)) AS MaxVisits,
           MAX(ISNULL(ld.TotalLeaveDays, 0)) AS MaxLeaveDays
    FROM BestTotals bt
    LEFT JOIN MedicalVisits mv ON mv.AgniveerId = bt.AgniveerId
    LEFT JOIN LeaveDays ld ON ld.AgniveerId = bt.AgniveerId
)
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, bt.BestTotal AS OverallMarks,
    ISNULL(mv.VisitCount, 0) AS MedicalVisits,
    ISNULL(ld.TotalLeaveDays, 0) AS TotalLeaveDays,
    (
        (CASE WHEN b.MaxScore > 0 THEN (bt.BestTotal * 1.0 / b.MaxScore) * 100.0 ELSE 0.0 END)
        - (CASE WHEN b.MaxVisits > 0 THEN (ISNULL(mv.VisitCount, 0) * 1.0 / b.MaxVisits) * 20.0 ELSE 0.0 END)
        - (CASE WHEN b.MaxLeaveDays > 0 THEN (ISNULL(ld.TotalLeaveDays, 0) * 1.0 / b.MaxLeaveDays) * 10.0 ELSE 0.0 END)
    ) AS CompositeScore
FROM AgniveerMaster a
INNER JOIN BestTotals bt ON bt.AgniveerId = a.Id
LEFT JOIN MedicalVisits mv ON mv.AgniveerId = a.Id
LEFT JOIN LeaveDays ld ON ld.AgniveerId = a.Id
CROSS JOIN Bounds b
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND a.IsActive = 1
ORDER BY CompositeScore DESC, a.AgniveerNo ASC
""".strip(),
    ("Equipment", "AgniveerWise"): """
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, COUNT(ae.Id) AS ItemsIssued, COUNT(CASE WHEN ae.ReturnDateTime IS NULL THEN 1 END) AS ItemsCurrentlyHeld
FROM AgniveerMaster a
INNER JOIN AgniveerEquipment ae ON ae.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
GROUP BY a.AgniveerNo, a.FullName
ORDER BY a.AgniveerNo ASC
""".strip(),
    ("disqualified", "removed"): """
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, a.DisqualifiedDate, a.Remarks, p.Name AS PlatoonName, c.Name AS CompanyName
FROM AgniveerMaster a
LEFT JOIN PlatoonMaster p ON a.PlatoonId = p.Id
LEFT JOIN CompanyMaster c ON p.CompanyId = c.Id
WHERE a.IsDisqualified = 1
ORDER BY a.DisqualifiedDate DESC, a.AgniveerNo ASC
""".strip(),
    ("personaldetail", "info"): """
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, a.DateOfBirth, a.DateOfJoining, a.MobileNo, a.Email, a.Address, a.BloodGroup, a.Class, p.Name AS PlatoonName, c.Name AS CompanyName
FROM AgniveerMaster a
LEFT JOIN PlatoonMaster p ON a.PlatoonId = p.Id
LEFT JOIN CompanyMaster c ON p.CompanyId = c.Id
ORDER BY a.AgniveerNo ASC
""".strip(),
    ("Medical", "Individual"): """
WITH MedicalAvg AS (
    SELECT AgniveerId, AVG(Height) AS AvgHeight, AVG(Weight) AS AvgWeight
    FROM MedicalRecordMaster
    WHERE Height IS NOT NULL AND Weight IS NOT NULL
    GROUP BY AgniveerId
),
Resolved AS (
    SELECT a.Id AS AgniveerId, COALESCE(m.AvgHeight, a.Height) AS AvgHeight, COALESCE(m.AvgWeight, a.Weight) AS AvgWeight
    FROM AgniveerMaster a
    LEFT JOIN MedicalAvg m ON m.AgniveerId = a.Id
),
BmiData AS (
    SELECT AgniveerId, CASE WHEN AvgHeight IS NULL OR AvgWeight IS NULL OR AvgHeight <= 0 THEN NULL WHEN AvgWeight / POWER(AvgHeight / 100.0, 2) < 18.5 THEN 'Underweight' WHEN AvgWeight / POWER(AvgHeight / 100.0, 2) < 25 THEN 'Normal' WHEN AvgWeight / POWER(AvgHeight / 100.0, 2) < 30 THEN 'Overweight' ELSE 'Obese' END AS BmiCategory
    FROM Resolved
)
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, a.BloodGroup, b.BmiCategory, m.Diagnosis, m.Status, m.VisitDate
FROM AgniveerMaster a
LEFT JOIN BmiData b ON b.AgniveerId = a.Id
LEFT JOIN MedicalRecordMaster m ON m.AgniveerId = a.Id
ORDER BY a.AgniveerNo ASC, m.VisitDate DESC
""".strip(),
    ("Strength", ""): """
SELECT TOP ({top_n}) c.Name AS CompanyName, p.Name AS PlatoonName,
    COUNT(a.Id) AS TotalStrength,
    SUM(CASE WHEN a.IsActive = 1 THEN 1 ELSE 0 END) AS ActiveCount,
    SUM(CASE WHEN a.IsActive = 0 OR a.IsActive IS NULL THEN 1 ELSE 0 END) AS InactiveCount,
    SUM(CASE WHEN a.IsDisqualified = 1 THEN 1 ELSE 0 END) AS DisqualifiedCount
FROM AgniveerMaster a
LEFT JOIN PlatoonMaster p ON a.PlatoonId = p.Id
LEFT JOIN CompanyMaster c ON p.CompanyId = c.Id
GROUP BY c.Name, p.Name
ORDER BY c.Name ASC, p.Name ASC
""".strip(),
    ("Schedule", "bytoday"): """
SELECT TOP ({top_n}) c.Name AS CompanyName, s.ScheduleDate, s.Pd, s.TimeRange, s.Code, s.Type, s.Details, s.Location, s.Resp
FROM CompanySchedule s
LEFT JOIN CompanyMaster c ON s.CompanyId = c.Id
WHERE CAST(s.ScheduleDate AS DATE) = CAST(GETDATE() AS DATE)
ORDER BY c.Name ASC, s.ScheduleDate ASC, s.Pd ASC
""".strip(),
    ("Schedule", "bycompany"): """
SELECT TOP ({top_n}) c.Name AS CompanyName, s.ScheduleDate, s.Pd, s.TimeRange, s.Code, s.Type, s.Details, s.Location, s.Resp
FROM CompanySchedule s
LEFT JOIN CompanyMaster c ON s.CompanyId = c.Id
ORDER BY s.ScheduleDate DESC, c.Name ASC, s.Pd ASC
""".strip(),
    ("Schedule", "bydate"): """
SELECT TOP ({top_n}) c.Name AS CompanyName, s.ScheduleDate, s.Pd, s.TimeRange, s.Code, s.Type, s.Details, s.Location, s.Resp
FROM CompanySchedule s
LEFT JOIN CompanyMaster c ON s.CompanyId = c.Id
ORDER BY s.ScheduleDate DESC, c.Name ASC, s.Pd ASC
""".strip(),
    ("Schedule", "byagniveer"): """
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, c.Name AS CompanyName, s.ScheduleDate, s.Pd, s.TimeRange, s.Code, s.Type, s.Details, s.Location
FROM CompanySchedule s
INNER JOIN CompanyMaster c ON s.CompanyId = c.Id
INNER JOIN PlatoonMaster p ON p.CompanyId = c.Id
INNER JOIN AgniveerMaster a ON a.PlatoonId = p.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
ORDER BY s.ScheduleDate DESC, a.AgniveerNo ASC, s.Pd ASC
""".strip(),
}


# Golden templates are static SQL that only ever substitute {top_n} and,
# for Performance, an exact-match {section_clause} against this fixed set —
# every other filter (batch/platoon/company/class/bloodGroup/leaveType/...)
# is silently ignored whenever a golden template is used, regardless of
# whether the user's question asked for it. `_golden_query_can_satisfy`
# below refuses the golden path whenever a filter is present that it can't
# actually express, so those questions fall through to the AST path instead
# of silently answering an unscoped/broader question than was asked.
_GOLDEN_SAFE_SECTIONS = {"BPET", "PPT", "Firing", "Drill", "Theory"}

# Every filter dimension the intent pipeline can extract (see
# intent_engine/entity_extractor.py / admin_intent.py) that no golden
# template has a placeholder for. If any of these carry a real value, the
# golden path cannot honor them and must not be used.
_GOLDEN_UNSUPPORTED_FILTER_KEYS = (
    "companyId", "company_id", "Company", "company",
    "platoonId", "platoon_id",
    "batchId", "batch_id",
    "agniveerNo", "agniveer_no",
    "class",
    "bloodGroup", "blood_group",
    "sport",
    "grading",
    "leaveType", "leave_type",
    "unitName", "unit_name",
    "equipmentName", "item_name",
    "equipmentType", "equipment_type",
    "diagnose",
    "givenCondition", "given_condition",
    "returnCondition", "return_condition",
    "attemptNo", "attempt_no",
    "fromAttempt", "from_attempt",
    "toAttempt", "to_attempt",
    "bmiCategory", "bmi_category",
    "days",
    "fromDate", "from_date",
    "toDate", "to_date",
    "date",
)


def _golden_query_can_satisfy(intent: Optional[Dict]) -> bool:
    intent = intent or {}
    for key in _GOLDEN_UNSUPPORTED_FILTER_KEYS:
        if intent.get(key) not in (None, "", [], {}):
            return False
    section = intent.get("section") or intent.get("sub_section")
    if section and str(section).strip() not in _GOLDEN_SAFE_SECTIONS:
        return False
    # medicalStatus IS a real, meaningful filter no golden template applies —
    # checked separately since it's read via _pick_legacy_value's two spellings.
    if intent.get("medicalStatus") not in (None, "", [], {}) or intent.get("medical_status") not in (None, "", [], {}):
        return False
    return True


def _render_golden_query(template: str, intent: Optional[Dict]) -> str:
    """Fill the {top_n} placeholder a golden template may contain from the
    classifier intent. The only substitution ever performed on a golden
    query is this validated integer — never raw user text — so this path
    can never introduce SQL injection."""
    intent = intent or {}
    raw_n = intent.get("number") or intent.get("top_n")
    section = str(intent.get("section") or intent.get("sub_section") or "").strip()
    try:
        top_n = int(raw_n) if raw_n is not None else SQL_MAX_ROWS
    except (TypeError, ValueError):
        top_n = SQL_MAX_ROWS
    top_n = max(1, min(top_n, SQL_MAX_ROWS))
    section_clause = ""
    if section in {"BPET", "PPT", "Firing", "Drill"}:
        section_clause = f"AND sec.SectionName = '{section}'"
    try:
        return template.format(top_n=top_n, section_clause=section_clause)
    except (KeyError, IndexError):
        return template


_GENERATION_SYSTEM = (
    "You are a T-SQL generator for a read-only reporting API. "
    "Given the schema and a question, output ONE SQL Server SELECT statement "
    "and NOTHING else — no prose, no markdown, no comments, no semicolons. "
    "Obey every HARD RULE. If the question cannot be answered from the schema, "
    "output exactly: CANNOT_ANSWER.\n\n"
    "Example 1:\n"
    "QUESTION: Top 10 performers in BPET\n"
    "SQL: SELECT TOP 10 a.AgniveerNo, a.FullName, SUM(sa.MarksObtained) AS BestTotal FROM AgniveerMaster a INNER JOIN AgniveerScoreAttempt sa ON a.Id = sa.AgniveerId INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId INNER JOIN ScoreSectionMaster sec ON sec.Id = si.SectionId WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL) AND sa.IsBestAttempt = 1 AND sec.SectionName = 'BPET' GROUP BY a.AgniveerNo, a.FullName ORDER BY BestTotal DESC\n\n"
    "Example 2:\n"
    "QUESTION: Which Agniveers play volleyball?\n"
    "SQL: SELECT a.AgniveerNo, a.FullName FROM AgniveerMaster a WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL) AND a.Sports = 'Volleyball'\n\n"
    "Example 3:\n"
    "QUESTION: PPT grades for Alpha company\n"
    "SQL: WITH AttemptedMax AS (SELECT DISTINCT sa.AgniveerId, si.SectionId, SUM(si.MaxMarks) AS DynamicMax FROM AgniveerScoreAttempt sa INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId GROUP BY sa.AgniveerId, si.SectionId), BestTotals AS (SELECT sa.AgniveerId, si.SectionId, SUM(sa.MarksObtained) AS BestTotal FROM AgniveerScoreAttempt sa INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId WHERE sa.IsBestAttempt = 1 GROUP BY sa.AgniveerId, si.SectionId), Scored AS (SELECT bt.AgniveerId, sec.SectionName, CASE WHEN dm.DynamicMax IS NULL OR dm.DynamicMax = 0 THEN NULL WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 90 THEN 'Exceptionally Well' WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 75 THEN 'Excellent' WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 60 THEN 'Good' WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 45 THEN 'SAT' ELSE 'Fail' END AS Grade FROM BestTotals bt INNER JOIN ScoreSectionMaster sec ON sec.Id = bt.SectionId LEFT JOIN AttemptedMax dm ON dm.AgniveerId = bt.AgniveerId AND dm.SectionId = bt.SectionId) SELECT a.AgniveerNo, a.FullName, sg.Grade FROM AgniveerMaster a INNER JOIN PlatoonMaster p ON a.PlatoonId = p.Id INNER JOIN CompanyMaster c ON p.CompanyId = c.Id INNER JOIN Scored sg ON a.Id = sg.AgniveerId WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL) AND sg.SectionName = 'PPT' AND c.Name = 'Alpha'\n\n"
    "Example 4:\n"
    "QUESTION: failed firing and still have issued equipment\n"
    "SQL: WITH AttemptedMax AS (SELECT DISTINCT sa.AgniveerId, si.SectionId, SUM(si.MaxMarks) AS DynamicMax FROM AgniveerScoreAttempt sa INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId GROUP BY sa.AgniveerId, si.SectionId), BestTotals AS (SELECT sa.AgniveerId, si.SectionId, SUM(sa.MarksObtained) AS BestTotal FROM AgniveerScoreAttempt sa INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId WHERE sa.IsBestAttempt = 1 GROUP BY sa.AgniveerId, si.SectionId), Scored AS (SELECT bt.AgniveerId, sec.SectionName, CASE WHEN dm.DynamicMax IS NULL OR dm.DynamicMax = 0 THEN NULL WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 90 THEN 'Exceptionally Well' WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 75 THEN 'Excellent' WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 60 THEN 'Good' WHEN 100.0 * bt.BestTotal / dm.DynamicMax >= 45 THEN 'SAT' ELSE 'Fail' END AS Grade FROM BestTotals bt INNER JOIN ScoreSectionMaster sec ON sec.Id = bt.SectionId LEFT JOIN AttemptedMax dm ON dm.AgniveerId = bt.AgniveerId AND dm.SectionId = bt.SectionId), FiringFail AS (SELECT AgniveerId FROM Scored WHERE SectionName = 'Firing' AND Grade = 'Fail'), HasEq AS (SELECT AgniveerId FROM AgniveerEquipment WHERE ReturnDateTime IS NULL) SELECT a.AgniveerNo, a.FullName FROM AgniveerMaster a INNER JOIN FiringFail f ON a.Id = f.AgniveerId INNER JOIN HasEq e ON a.Id = e.AgniveerId WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)\n\n"
    "Example 5:\n"
    "QUESTION: Rank volleyball players by total marks\n"
    "SQL: SELECT a.AgniveerNo, a.FullName, SUM(sa.MarksObtained) AS BestTotal FROM AgniveerMaster a INNER JOIN AgniveerScoreAttempt sa ON a.Id = sa.AgniveerId WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL) AND sa.IsBestAttempt = 1 AND a.Sports = 'Volleyball' GROUP BY a.AgniveerNo, a.FullName ORDER BY BestTotal DESC"
)

# ── Safety validator ───────────────────────────────────────────────────────
_FORBIDDEN = re.compile(
    r"\b(insert|update|delete|merge|drop|alter|truncate|create|grant|revoke|"
    r"exec|execute|sp_|xp_|openrowset|openquery|bulk|shutdown|reconfigure|"
    r"waitfor)\b",
    re.IGNORECASE,
)
_MULTI_STATEMENT = re.compile(r";\s*\S")  # a ';' followed by more content
_COMMENT = re.compile(r"(--|/\*|\*/)")

# R7 (business_rules.LLM_HARD_RULES): any query that touches
# AgniveerScoreAttempt.MarksObtained MUST scope to a single attempt per
# Agniveer/sub-item — either via `IsBestAttempt = 1` or by keying off
# `AttemptNo` directly (e.g. a per-attempt trend/comparison). Without one of
# these, summing/aggregating MarksObtained silently double/triple-counts
# every retake, since AgniveerScoreAttempt has one row per attempt.
_SCORE_ATTEMPT_TABLE = re.compile(r"\bagniveerscoreattempt\b", re.IGNORECASE)
_MARKS_COLUMN = re.compile(r"\bmarksobtained\b", re.IGNORECASE)
_BEST_ATTEMPT_GUARD = re.compile(r"\bisbestattempt\b", re.IGNORECASE)
_ATTEMPT_NO_GUARD = re.compile(r"\battemptno\b", re.IGNORECASE)


def validate_sql(sql: str) -> Optional[str]:
    """Return an error string if the SQL is unsafe, else None.

    NOT on the live request path: execute_sql_query calls
    sql_validator.validate_sql() (sql_validator.py) instead, which has its
    own, now-equivalent copy of this rule set (R7 + DENIED_TABLES/COLUMNS +
    comment/multi-statement checks — ported there since this function's
    checks were silently never running against real traffic). Kept here for
    the tests that exercise it directly against GOLDEN_QUERIES templates.
    """
    if not sql or not sql.strip():
        return "Empty SQL."
    s = sql.strip().rstrip(";").strip()

    low = s.lower()
    if not (low.startswith("select") or low.startswith("with")):
        return "Only single SELECT / WITH...SELECT statements are allowed."
    if _MULTI_STATEMENT.search(s):
        return "Multiple statements are not allowed."
    if _COMMENT.search(s):
        return "Comments are not allowed."
    if _FORBIDDEN.search(low):
        return "Statement contains a forbidden keyword."

    if (
        _SCORE_ATTEMPT_TABLE.search(low)
        and _MARKS_COLUMN.search(low)
        and not _BEST_ATTEMPT_GUARD.search(low)
        and not _ATTEMPT_NO_GUARD.search(low)
    ):
        return (
            "Query uses AgniveerScoreAttempt.MarksObtained without scoping to a "
            "single attempt (IsBestAttempt = 1 or AttemptNo) — would double-count "
            "marks across retakes (R7)."
        )

    for denied in DENIED_TABLES:
        if re.search(rf"\b{denied}\b", low):
            return f"Access to table '{denied}' is denied."
    for denied in DENIED_COLUMNS:
        tbl, col = denied.split(".")
        if col in low and tbl in low:
            return f"Access to column '{denied}' is denied."
    return None


# ── SQL generation (LLM) ───────────────────────────────────────────────────
# NOT called by execute_sql_query, or by anything else in this codebase —
# see the module docstring. Left in place as scaffolding for a future
# LLM-generation fallback (for filters/categories the golden path and
# query_planner_v2's AST path can't express), but currently dead code
# outside of tests that patch it directly.
def generate_sql(
    question: str, intent: Optional[Dict] = None
) -> Tuple[Optional[str], Optional[str]]:
    """Generate a single SELECT for `question`. Returns (sql, error)."""
    try:
        from config import DEFAULT_MODEL, ollama_session
        from ollama_cpu_chat import chat_with_fallback
        from sql_schema_guard import generate_dynamic_schema_card
        from business_rules import LLM_HARD_RULES
    except Exception as exc:  # pragma: no cover
        return None, f"LLM unavailable: {exc}"

    hint = ""
    if intent:
        hint = f"\nCLASSIFIER HINT (may be partial): {intent}\n"

    dynamic_schema = generate_dynamic_schema_card()
    
    if SQL_SERVER_2008_COMPAT:
        dialect_hint = "\nDIALECT: SQL Server 2008 target. DO NOT use STRING_AGG, OFFSET/FETCH, IIF, CONCAT, or LAG/LEAD. Use TOP (n), STUFF+FOR XML PATH for aggregation, ROW_NUMBER() for paging.\n"
    else:
        dialect_hint = ""

    user = f"{dynamic_schema}\n{LLM_HARD_RULES}\n{dialect_hint}\n{hint}\nQUESTION: {question}\nSQL:"
    try:
        result = chat_with_fallback(
            ollama_session,
            DEFAULT_MODEL,
            [
                {"role": "system", "content": _GENERATION_SYSTEM},
                {"role": "user", "content": user},
            ],
            stream_tokens=False,
        )
        text = str(getattr(result, "text", "") or "")
    except Exception as exc:
        return None, f"SQL generation failed: {exc}"

    sql = _extract_sql(text)
    if not sql or sql.strip().upper() == "CANNOT_ANSWER":
        return None, "CANNOT_ANSWER"
    return sql, None


def _extract_sql(text: str) -> str:
    """Strip markdown fences / stray prose, keep the SQL body."""
    t = text.strip()
    t = re.sub(r"```(?:sql)?", "", t, flags=re.IGNORECASE).strip()
    # Keep from the first SELECT/WITH onward.
    m = re.search(r"\b(with|select)\b", t, re.IGNORECASE)
    sql = t[m.start() :].strip() if m else t
    # Fix common LLM hallucination where it thinks AgniveerMaster has AgniveerId
    sql = re.sub(r'\ba\.AgniveerId\b', 'a.Id', sql, flags=re.IGNORECASE)
    return sql


def _to_camel_case(name: str) -> str:
    """PascalCase SQL column name -> camelCase, matching System.Text.Json's
    default camelCase naming policy — the same simple "lowercase the first
    character only" rule .NET's JSON serialization uses, and the convention
    universal_normalizer.py / cross_filter_engine.py / compare_engine.py
    (and every test fixture for them) are built against (e.g. "agniveerNo",
    "fullName"). SQL Server's cursor.description returns column names
    verbatim ("AgniveerNo", "FullName"); without this conversion those keys
    never match _ID_FIELD_PRIORITY in either module, so no SQL-backend row
    is recognized as an Agniveer record and cross-filter intersection
    silently finds nothing to match on."""
    if not name:
        return name
    return name[0].lower() + name[1:]


def _camel_case_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {_to_camel_case(k): v for k, v in row.items()}


def _to_section(
    rows: List[Dict[str, Any]], intent: Optional[Dict] = None, sql: Optional[str] = None
) -> Dict[str, Any]:
    """Wrap flat rows in the same envelope shape the .NET path produces, so
    `universal_normalizer.normalize_response()` / `result_combiner._extract_records()`
    resolve the rows directly instead of falling back to raw-row scanning.

    Column names are camelCased here (see `_to_camel_case`) so the row shape
    actually matches that .NET envelope's key casing, not just its nesting.
    """
    camel_rows = [_camel_case_row(r) for r in rows]
    res = {
        "success": True,
        "records": camel_rows,
        "data": camel_rows,
        "count": len(camel_rows),
    }
    if sql:
        res["sql"] = sql
    return res


# ── Read-only execution ────────────────────────────────────────────────────
def run_readonly(sql: str, params: Optional[List[Any]] = None) -> Tuple[Optional[List[Dict[str, Any]]], Optional[str]]:
    """Execute a validated SELECT against the READ-ONLY login. (rows, error)."""
    if not SQL_READONLY_CONN:
        return None, "SQL_READONLY_CONN is not configured."
    try:
        import pyodbc  # imported lazily so the rest of AgniAI runs without it
        pyodbc.pooling = True
    except Exception as exc:  # pragma: no cover
        return None, f"pyodbc not installed: {exc}"

    try:
        conn = pyodbc.connect(
            SQL_READONLY_CONN, timeout=SQL_COMMAND_TIMEOUT_S, autocommit=True
        )
        conn.timeout = SQL_COMMAND_TIMEOUT_S
        cur = conn.cursor()
        # Belt-and-suspenders row cap regardless of query shape (works on 2008).
        cur.execute(f"SET ROWCOUNT {SQL_MAX_ROWS}")
        if params:
            cur.execute(sql, tuple(params))
        else:
            cur.execute(sql)
        cols = [d[0] for d in cur.description]
        
        rows = []
        while True:
            chunk = cur.fetchmany(100)
            if not chunk:
                break
            for r in chunk:
                rows.append(dict(zip(cols, r)))
                
        cur.execute("SET ROWCOUNT 0")
        conn.close()
        return rows, None
    except Exception as exc:
        # Never leak the raw SQL / connection details to the caller.
        logger.warning("SQL execution error: %s | %s\nQuery: %s", type(exc).__name__, str(exc), sql)
        return None, "The generated query could not be executed against the database."


# ── Public entrypoint — AST Pipeline ──────
def execute_sql_query(
    payload: Optional[Dict] = None,
    *,
    question: str = "",
    intent: Optional[Dict] = None,
    trace_id: Optional[str] = None,
    **_ignored: Any,
) -> Tuple[Any, Optional[str]]:
    """
    Executes a query via the AST semantic compilation pipeline.
    """
    logger.warning(f"[DEBUG SQL EXECUTOR] question: {question!r}")
    if not intent:
        return None, "No intent provided to query planner."

    from query_planner_v2 import query_planner_v2
    from sql_builder import sql_builder
    from sql_validator import sql_validator

    import time
    from explainability_engine import explainability_engine
    
    t0 = time.time()
    # 1. AST Generation
    try:
        # Convert legacy intent to v2 intent format
        def _pick_legacy_value(*keys: str) -> Any:
            for key in keys:
                value = intent.get(key)
                if value not in (None, "", [], {}):
                    return value
            return None

        filters: Dict[str, Any] = {}
        if isinstance(intent.get("filters"), dict):
            filters.update(
                {
                    key: value
                    for key, value in intent["filters"].items()
                    if value not in (None, "", [], {})
                }
            )

        company_id = _pick_legacy_value("companyId", "company_id")
        platoon_id = _pick_legacy_value("platoonId", "platoon_id")
        batch_id = _pick_legacy_value("batchId", "batch_id")
        agniveer_no = _pick_legacy_value("agniveerNo", "agniveer_no")
        medical_status = _pick_legacy_value("medicalStatus", "medical_status")
        company_name = _pick_legacy_value("Company", "company")
        class_ = _pick_legacy_value("class", "class_")
        blood_group = _pick_legacy_value("bloodGroup", "blood_group")
        sport = _pick_legacy_value("sport")
        diagnose = _pick_legacy_value("diagnose")
        unit_name = _pick_legacy_value("unitName", "unit_name")

        if company_id is not None:
            filters.setdefault("Company.Id", company_id)
        elif company_name is not None:
            filters.setdefault("Company.Name", company_name)

        if platoon_id is not None:
            filters.setdefault("Agniveer.PlatoonId", platoon_id)
        if batch_id is not None:
            filters.setdefault("Agniveer.BatchId", batch_id)
        if agniveer_no is not None:
            filters.setdefault("Agniveer.AgniveerNo", agniveer_no)
        if medical_status is not None:
            filters.setdefault("Medical.Status", medical_status)
        # These map onto a single real column via an existing, correctly
        # concept-mapped table (business_ontology.json), so promoting them
        # is safe. Deliberately NOT extended to section/subSection/grading/
        # leaveType/attemptNo-ranges/bmiCategory/equipmentName/equipmentType/
        # givenCondition/returnCondition/days — those need either a
        # multi-hop join through a table business_ontology.json doesn't map
        # 1:1 (e.g. "equipmentName" needs AgniveerEquipment -> EquipmentMaster,
        # but the "Equipment" concept points at AgniveerEquipment, which has
        # no Name column) or non-column semantics (leaveType is several
        # boolean flags, not one column; bmiCategory/days are computed, not
        # stored) that a flat WhereNode can't express correctly without
        # dedicated handling this pass doesn't attempt.
        if class_ is not None:
            filters.setdefault("Agniveer.Class", class_)
        if blood_group is not None:
            filters.setdefault("Agniveer.BloodGroup", blood_group)
        if sport is not None:
            filters.setdefault("Agniveer.Sports", {"operator": "LIKE", "value": f"%{sport}%"})
        if diagnose is not None:
            filters.setdefault("Medical.Diagnosis", {"operator": "LIKE", "value": f"%{diagnose}%"})
        if unit_name is not None:
            filters.setdefault("Distribution.Name", {"operator": "LIKE", "value": f"%{unit_name}%"})

        v2_intent = {
            "base_concept": intent.get("category", "Agniveer"),
            "filters": filters,
            "limit": intent.get("number") or intent.get("top_n") or SQL_MAX_ROWS
        }
        golden_template = GOLDEN_QUERIES.get(
            (intent.get("category"), intent.get("operation"))
        )
        if golden_template and _golden_query_can_satisfy(intent):
            ast = None
            sql = _render_golden_query(golden_template, intent)
            params: List[Any] = []
        else:
            ast = query_planner_v2.plan_query(v2_intent)
            sql = ""
            params = []
    except Exception as e:
        logger.error(f"AST generation failed: {e}")
        return None, f"Could not construct semantic query plan: {e}"
        
    t1 = time.time()

    # 2. AST Validation
    if ast is not None:
        is_valid, val_err = sql_validator.validate_ast(ast)
        if not is_valid:
            logger.info("AST rejected by validator: %s", val_err)
            metrics_hook("validator_rejected")
            return None, val_err

        # 3. Compilation
        sql, params = sql_builder.build(ast)
        logger.warning(f"[DEBUG SQL EXECUTOR] Compiled SQL:\n{sql}\nParams: {params}")
    else:
        logger.warning(f"[DEBUG SQL EXECUTOR] Golden SQL:\n{sql}\nParams: {params}")

    t2 = time.time()

    # 4. Final SQL Validation
    is_sql_valid, sql_val_err = sql_validator.validate_sql(sql)
    if not is_sql_valid:
        metrics_hook("validator_rejected")
        return None, sql_val_err

    if ast is None:
        metrics_hook("golden_hit")
    metrics_hook("generated")

    # 5. Execute read-only
    rows, run_err = run_readonly(sql, params)
    if run_err:
        metrics_hook("exec_error")
        return None, run_err
        
    t3 = time.time()
    
    # 6. Metadata & Explainability
    explanation = explainability_engine.explain(ast) if ast is not None else {
        "intent": "Database Query",
        "base_table": intent.get("category", "Agniveer"),
        "joins": [],
        "filters": [],
        "groupings": [],
        "having": [],
        "aggregations": [],
        "sorting": [],
        "limit": intent.get("number") or intent.get("top_n") or SQL_MAX_ROWS,
    }
    execution_metadata = {
        "planning_duration_ms": int((t1 - t0) * 1000),
        "compilation_duration_ms": int((t2 - t1) * 1000),
        "execution_duration_ms": int((t3 - t2) * 1000),
        "rows_returned": len(rows) if rows else 0,
        "explanation": explanation
    }
    
    res = _to_section(rows or [], intent, sql=sql)
    res["execution_metadata"] = execution_metadata
    return res, None


def metrics_hook(event: str) -> None:
    """Best-effort metrics increment — never lets an observability failure
    affect the query result. Imported lazily to avoid a hard dependency."""
    try:
        from metrics import metrics_collector

        {
            "generated": metrics_collector.inc_sql_generated,
            "golden_hit": metrics_collector.inc_sql_golden_hit,
            "validator_rejected": metrics_collector.inc_sql_validator_rejected,
            "cannot_answer": metrics_collector.inc_sql_cannot_answer,
            "exec_error": metrics_collector.inc_sql_exec_error,
        }[event]()
    except Exception:
        pass
