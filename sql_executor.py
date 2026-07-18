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

Flow:  intent/question -> [golden fast-path?] -> generate SQL (LLM)
       -> validate (hard safety gate) -> run read-only -> rows

SAFETY MODEL (do not weaken any of these):
  1. Connect with a READ-ONLY SQL login (db_datareader only, DENY on
     UserMaster.Password / LoginToken). The login is the real wall; the
     validator below is defense-in-depth, not the primary control.
  2. Only a single SELECT / WITH...SELECT statement is ever executed.
  3. Column/table allowlist derived from the schema card. Sensitive columns
     are hard-denied even if a grant slips.
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
  AND sa.IsBestAttempt = 1
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
  AND sa.IsBestAttempt = 1
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
GROUP BY a.AgniveerNo, a.FullName
ORDER BY a.AgniveerNo
""".strip(),
    ("Leave", "Current"): """
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, l.FromDate, l.ToDate, 
    CASE WHEN l.[OnEX PPG] = 1 THEN 'EX PPG' WHEN l.[OnATTN''C'] = 1 THEN 'ATTN''C' WHEN l.OnAnnualLeave = 1 THEN 'Annual' WHEN l.OnMedicalLeave = 1 THEN 'Medical' WHEN l.OnSickLeave = 1 THEN 'Sick' WHEN l.IsHospitalized = 1 THEN 'Hospitalized' WHEN l.IsAbscondedLeave = 1 THEN 'Absconded' ELSE 'Other' END AS LeaveType
FROM AgniveerMaster a
INNER JOIN AgniveerLeaveMaster l ON l.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND l.FromDate IS NOT NULL AND l.ToDate IS NOT NULL
  AND CAST(GETDATE() AS DATE) BETWEEN CAST(l.FromDate AS DATE) AND CAST(l.ToDate AS DATE)
ORDER BY l.FromDate DESC, a.AgniveerNo ASC
""".strip(),
    ("Leave", "Most"): """
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, SUM(CASE WHEN l.[OnEX PPG] = 1 THEN (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) / 4 ELSE (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) END) AS TotalLeaveDays
FROM AgniveerMaster a
INNER JOIN AgniveerLeaveMaster l ON l.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND l.FromDate IS NOT NULL AND l.ToDate IS NOT NULL
GROUP BY a.AgniveerNo, a.FullName
ORDER BY SUM(CASE WHEN l.[OnEX PPG] = 1 THEN (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) / 4 ELSE (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) END) DESC, a.AgniveerNo ASC
""".strip(),
    ("Leave", "Least"): """
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, SUM(CASE WHEN l.[OnEX PPG] = 1 THEN (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) / 4 ELSE (DATEDIFF(DAY, l.FromDate, l.ToDate) + 1) END) AS TotalLeaveDays
FROM AgniveerMaster a
INNER JOIN AgniveerLeaveMaster l ON l.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
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
  AND l.FromDate IS NOT NULL AND l.ToDate IS NOT NULL
  AND l.IsAbscondedLeave = 1
ORDER BY l.FromDate DESC
""".strip(),
    ("Medical", "Disease"): """
SELECT TOP ({top_n}) m.Diagnosis, COUNT(DISTINCT m.AgniveerId) AS AffectedCount
FROM MedicalRecordMaster m
INNER JOIN AgniveerMaster a ON a.Id = m.AgniveerId
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND m.Diagnosis IS NOT NULL
GROUP BY m.Diagnosis
ORDER BY COUNT(DISTINCT m.AgniveerId) DESC, m.Diagnosis ASC
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
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND (l.AgniveerId IS NULL OR (CASE WHEN l.Status = 'Sent' AND l.ReceivedDate IS NULL THEN 'NotResponded' ELSE l.Status END) = 'Pending')
ORDER BY l.SentDate ASC, a.AgniveerNo ASC
""".strip(),
    ("Verification", "Completed"): """
WITH Latest AS (
    SELECT AgniveerId, PoliceStation, SentDate, ReceivedDate, Status,
        ROW_NUMBER() OVER (PARTITION BY AgniveerId ORDER BY SentDate DESC, Id DESC) AS rn
    FROM PoliceVerificationMaster
)
SELECT a.AgniveerNo, a.FullName, l.PoliceStation, l.SentDate, l.ReceivedDate, l.Status
FROM AgniveerMaster a
INNER JOIN Latest l ON l.AgniveerId = a.Id AND l.rn = 1
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND l.Status = 'Completed'
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
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND l.Status = 'Rejected'
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
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND l.Status = 'Sent'
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
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND l.Status = 'Sent' AND l.ReceivedDate IS NULL
ORDER BY CASE WHEN l.SentDate IS NOT NULL THEN DATEDIFF(DAY, l.SentDate, GETDATE()) ELSE NULL END DESC, a.AgniveerNo ASC
""".strip(),
    ("Equipment", "Holding"): """
SELECT a.AgniveerNo, a.FullName, e.Name AS EquipmentName, ae.GivenDateTime, ae.GivenCondition
FROM AgniveerMaster a
INNER JOIN AgniveerEquipment ae ON ae.AgniveerId = a.Id
INNER JOIN EquipmentMaster e ON e.Id = ae.EquipmentId
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
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
  AND a.Sports IS NOT NULL AND a.Sports <> ''
ORDER BY a.Sports ASC, a.AgniveerNo ASC
""".strip(),
    ("Skills", "ByClass"): """
SELECT a.AgniveerNo, a.FullName, p.Name AS Platoon, c.Name AS Company, a.Class
FROM AgniveerMaster a
LEFT JOIN PlatoonMaster p ON a.PlatoonId = p.Id
LEFT JOIN CompanyMaster c ON p.CompanyId = c.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
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
  AND sa.IsBestAttempt = 1
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
)
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, sg.SectionName, sg.Grade, sg.Percentage, sg.BestTotal
FROM AgniveerMaster a
INNER JOIN Scored sg ON sg.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
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
)
SELECT sg.SectionName, sg.Grade, COUNT(DISTINCT a.Id) AS AgniveerCount
FROM AgniveerMaster a
INNER JOIN Scored sg ON sg.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
GROUP BY sg.SectionName, sg.Grade
ORDER BY sg.SectionName, sg.Grade
""".strip(),
    ("Performance", "Improvement"): """
WITH AttemptTotals AS (
    SELECT sa.AgniveerId, sa.AttemptNo, SUM(sa.MarksObtained) AS TotalMarks
    FROM AgniveerScoreAttempt sa
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
  AND i.Improvement > 0
ORDER BY i.Improvement DESC, a.AgniveerNo ASC
""".strip(),
    ("Performance", "Drop"): """
WITH AttemptTotals AS (
    SELECT sa.AgniveerId, sa.AttemptNo, SUM(sa.MarksObtained) AS TotalMarks
    FROM AgniveerScoreAttempt sa
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
  AND d.ScoreDrop > 0
ORDER BY d.ScoreDrop DESC, a.AgniveerNo ASC
""".strip(),
    ("Performance", "AttemptWise"): """
WITH AttemptTotals AS (
    SELECT sa.AgniveerId, sa.AttemptNo, SUM(sa.MarksObtained) AS TotalMarks
    FROM AgniveerScoreAttempt sa
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
ORDER BY mt.MaxTotal DESC, a.AgniveerNo ASC, t.AttemptNo ASC
""".strip(),
    ("Performance", "Trend"): """
WITH AttemptedMax AS (
    SELECT DISTINCT sa.AgniveerId, sa.AttemptNo, si.Id AS SubItemId, si.MaxMarks
    FROM AgniveerScoreAttempt sa
    INNER JOIN ScoreSubItemMaster si ON si.Id = sa.SubItemId
),
DynamicMaxPerAttempt AS (
    SELECT AgniveerId, AttemptNo, SUM(MaxMarks) AS DynamicMax
    FROM AttemptedMax
    GROUP BY AgniveerId, AttemptNo
),
TotalsPerAttempt AS (
    SELECT AgniveerId, AttemptNo, SUM(MarksObtained) AS TotalObtained
    FROM AgniveerScoreAttempt
    GROUP BY AgniveerId, AttemptNo
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
GROUP BY p.AttemptNo
ORDER BY p.AttemptNo ASC
""".strip(),
    ("Overall", "OverallPerformance"): """
SELECT TOP ({top_n}) a.AgniveerNo, a.FullName, SUM(sa.MarksObtained) AS OverallMarks
FROM AgniveerMaster a
INNER JOIN AgniveerScoreAttempt sa ON sa.AgniveerId = a.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
  AND sa.IsBestAttempt = 1
GROUP BY a.AgniveerNo, a.FullName
ORDER BY SUM(sa.MarksObtained) DESC, a.AgniveerNo ASC
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
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
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
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
ORDER BY a.AgniveerNo ASC, m.VisitDate DESC
""".strip(),
    ("Strength", ""): """
SELECT TOP ({top_n}) c.Name AS CompanyName, p.Name AS PlatoonName, COUNT(a.Id) AS TotalStrength
FROM AgniveerMaster a
LEFT JOIN PlatoonMaster p ON a.PlatoonId = p.Id
LEFT JOIN CompanyMaster c ON p.CompanyId = c.Id
WHERE (a.IsDisqualified <> 1 OR a.IsDisqualified IS NULL)
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
    """Return an error string if the SQL is unsafe, else None."""
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


def _to_section(
    rows: List[Dict[str, Any]], intent: Optional[Dict] = None, sql: Optional[str] = None
) -> Dict[str, Any]:
    """Wrap flat rows in the same envelope shape the .NET path produces, so
    `universal_normalizer.normalize_response()` / `result_combiner._extract_records()`
    resolve the rows directly instead of falling back to raw-row scanning."""
    res = {
        "success": True,
        "records": rows,
        "data": rows,
        "count": len(rows),
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

        v2_intent = {
            "base_concept": intent.get("category", "Agniveer"),
            "filters": filters,
            "limit": intent.get("number") or intent.get("top_n") or SQL_MAX_ROWS
        }
        golden_template = GOLDEN_QUERIES.get(
            (intent.get("category"), intent.get("operation"))
        )
        if golden_template:
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
