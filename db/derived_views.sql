-- ============================================================================
-- derived_views.sql
-- ============================================================================
-- Derived-views layer for AgniAI's text-to-SQL backend (sql_executor.py).
-- Each view re-expresses one piece of .NET business logic (from
-- AiCommandService.cs / PerformanceDomainHandler.cs) as a SQL view, so the
-- LLM SQL generator can filter a view column instead of re-deriving scoring,
-- grading, leave-day, BMI, or equipment-condition arithmetic per query.
--
-- Column types below are taken directly from the DB_Agni DDL provided
-- 2026-07-17 (CREATE TABLE scripts for AgniveerMaster, AgniveerScoreAttempt,
-- AgniveerSectionResult, AgniveerLeaveMaster, MedicalRecordMaster,
-- AgniveerEquipment, PoliceVerificationMaster, etc.) — not guessed.
--
-- Run once, by a DBA, against DB_Agni. Idempotent: safe to re-run.
-- The reporting login (db/readonly_login.sql) is explicitly denied
-- CREATE VIEW — these views must be created by a higher-privileged login,
-- then the reporting login (which only has db_datareader) can SELECT from
-- them like any other table.
--
-- ⚠ The original C# handler source (AiCommandService.cs /
-- PerformanceDomainHandler.cs) is not present in this repo, so business-rule
-- ambiguities were resolved by explicit confirmation rather than guessed —
-- see the comment above vw_AgniveerLeaveThreshold (all-time total /
-- single-row continuous) and vw_AgniveerBmi (height unit assumption still
-- open) before relying on those in production.
-- ============================================================================

USE DB_Agni;
GO

-- ── 1. vw_AgniveerBestAttemptTotals ─────────────────────────────────────────
-- PerformanceDomainHandler.GetBestTotalsAsync — sum of MarksObtained where
-- IsBestAttempt = 1, grouped by section.
IF OBJECT_ID (
    'dbo.vw_AgniveerBestAttemptTotals',
    'V'
) IS NOT NULL
DROP VIEW dbo.vw_AgniveerBestAttemptTotals;
GO
CREATE VIEW dbo.vw_AgniveerBestAttemptTotals AS
SELECT sa.AgniveerId, sec.SectionName AS Section, SUM(sa.MarksObtained) AS BestTotal
FROM dbo.AgniveerScoreAttempt sa
    INNER JOIN dbo.ScoreSubItemMaster si ON si.Id = sa.SubItemId
    INNER JOIN dbo.ScoreSectionMaster sec ON sec.Id = si.SectionId
WHERE
    sa.IsBestAttempt = 1
GROUP BY
    sa.AgniveerId,
    sec.SectionName;
GO

-- ── 2. vw_AgniveerSectionGrades ──────────────────────────────────────────────
-- PerformanceDomainHandler.ByGrading / GetGrade() — percentage against a
-- DYNAMIC max (sum of MaxMarks only for subitems the agniveer actually has an
-- AgniveerScoreAttempt row for, not every subitem in the section), bucketed:
-- >=90 Exceptionally Well / >=75 Excellent / >=60 Good / >=45 SAT / else Fail.
-- BUG FIX: BestTotal is now sourced from SUM(AgniveerScoreAttempt.MarksObtained
-- WHERE IsBestAttempt = 1) — matching the C# ByGrading numerator and
-- vw_AgniveerBestAttemptTotals's own source — instead of the earlier draft's
-- AgniveerSectionResult.SubItemTotalMarks/ExceptionalMarks, which is a
-- different table with no confirmed guarantee it equals this sum. Before
-- trusting this in production, spot-check a few real agniveers: does
-- AgniveerSectionResult.SubItemTotalMarks already equal
-- SUM(MarksObtained WHERE IsBestAttempt=1) today? If yes, the two sources
-- were equivalent and this is just a correctness-by-construction fix; if no,
-- earlier answers from this view (before this fix) would have diverged from
-- .NET's ByGrading output for the same question.
IF OBJECT_ID (
    'dbo.vw_AgniveerSectionGrades',
    'V'
) IS NOT NULL
DROP VIEW dbo.vw_AgniveerSectionGrades;
GO
CREATE VIEW dbo.vw_AgniveerSectionGrades AS
WITH
    AttemptedMax AS (
        SELECT DISTINCT
            sa.AgniveerId,
            si.SectionId,
            si.Id AS SubItemId,
            si.MaxMarks
        FROM dbo.AgniveerScoreAttempt sa
            INNER JOIN dbo.ScoreSubItemMaster si ON si.Id = sa.SubItemId
    ),
    DynamicMax AS (
        SELECT
            AgniveerId,
            SectionId,
            SUM(MaxMarks) AS DynamicMax
        FROM AttemptedMax
        GROUP BY
            AgniveerId,
            SectionId
    ),
    BestTotals AS (
        SELECT sa.AgniveerId, si.SectionId, SUM(sa.MarksObtained) AS BestTotal
        FROM dbo.AgniveerScoreAttempt sa
            INNER JOIN dbo.ScoreSubItemMaster si ON si.Id = sa.SubItemId
        WHERE
            sa.IsBestAttempt = 1
        GROUP BY
            sa.AgniveerId,
            si.SectionId
    ),
    Scored AS (
        SELECT bt.AgniveerId, bt.SectionId, sec.SectionName, bt.BestTotal, dm.DynamicMax
        FROM
            BestTotals bt
            INNER JOIN dbo.ScoreSectionMaster sec ON sec.Id = bt.SectionId
            LEFT JOIN DynamicMax dm ON dm.AgniveerId = bt.AgniveerId
            AND dm.SectionId = bt.SectionId
    )
SELECT
    AgniveerId,
    SectionId,
    SectionName,
    BestTotal,
    DynamicMax,
    CASE
        WHEN DynamicMax > 0 THEN 100.0 * BestTotal / DynamicMax
        ELSE NULL
    END AS Percentage,
    CASE
        WHEN DynamicMax IS NULL
        OR DynamicMax = 0 THEN NULL
        WHEN 100.0 * BestTotal / DynamicMax >= 90 THEN 'Exceptionally Well'
        WHEN 100.0 * BestTotal / DynamicMax >= 75 THEN 'Excellent'
        WHEN 100.0 * BestTotal / DynamicMax >= 60 THEN 'Good'
        WHEN 100.0 * BestTotal / DynamicMax >= 45 THEN 'SAT'
        ELSE 'Fail'
    END AS Grade
FROM Scored;
GO

-- ── 3. vw_AgniveerLeaveDayCounts ─────────────────────────────────────────────
-- AiCommandService.CalcLeaveCount — normal leave = inclusive day span
-- (ToDate - FromDate + 1); EX PPG = that span / 4, integer division (T-SQL
-- int/int already truncates, matching the C# integer division).
-- AgniveerLeaveMaster has no single LeaveType column — it's decomposed into
-- bit flags, so LeaveType here is derived. Rows with more than one flag set
-- true take the first match in the CASE (EX PPG checked first since it has
-- its own count formula) — confirm against real data that flags are
-- mutually exclusive per row before trusting multi-flag edge cases.
IF OBJECT_ID (
    'dbo.vw_AgniveerLeaveDayCounts',
    'V'
) IS NOT NULL
DROP VIEW dbo.vw_AgniveerLeaveDayCounts;
GO
CREATE VIEW dbo.vw_AgniveerLeaveDayCounts AS
SELECT
    l.AgniveerId,
    CASE
        WHEN l.[OnEX PPG] = 1 THEN 'EX PPG'
        WHEN l.[OnATTN'C'] = 1 THEN 'ATTN''C'
        WHEN l.OnAnnualLeave = 1 THEN 'Annual'
        WHEN l.OnMedicalLeave = 1 THEN 'Medical'
        WHEN l.OnSickLeave = 1 THEN 'Sick'
        WHEN l.IsHospitalized = 1 THEN 'Hospitalized'
        WHEN l.IsAbscondedLeave = 1 THEN 'Absconded'
        ELSE 'Other'
    END AS LeaveType,
    l.FromDate,
    l.ToDate,
    CASE
        WHEN l.[OnEX PPG] = 1 THEN (
            DATEDIFF (DAY, l.FromDate, l.ToDate) + 1
        ) / 4
        ELSE (
            DATEDIFF (DAY, l.FromDate, l.ToDate) + 1
        )
    END AS LeaveCount
FROM dbo.AgniveerLeaveMaster l
WHERE
    l.FromDate IS NOT NULL
    AND l.ToDate IS NOT NULL;
GO

-- ── 4. vw_AgniveerLeaveThreshold ─────────────────────────────────────────────
-- Cmd07/Cmd09 threshold rule — continuous 40-44 days OR total 55-59 days.
-- Confirmed: "total" sums ALL leave records ever for the agniveer (no
-- per-batch/cycle scoping), and "continuous" is a single AgniveerLeaveMaster
-- row's own span (adjacent rows are NOT merged into one run before
-- measuring).
IF OBJECT_ID (
    'dbo.vw_AgniveerLeaveThreshold',
    'V'
) IS NOT NULL
DROP VIEW dbo.vw_AgniveerLeaveThreshold;
GO
CREATE VIEW dbo.vw_AgniveerLeaveThreshold AS
SELECT AgniveerId, IsThreshold, Reason
FROM (
        SELECT
            AgniveerId, CAST(1 AS BIT) AS IsThreshold, 'Continuous 40-44 days' AS Reason
        FROM dbo.vw_AgniveerLeaveDayCounts
        WHERE
            LeaveCount BETWEEN 40 AND 44
        UNION
        SELECT
            AgniveerId, CAST(1 AS BIT) AS IsThreshold, 'Total 55-59 days' AS Reason
        FROM dbo.vw_AgniveerLeaveDayCounts
        GROUP BY
            AgniveerId
        HAVING
            SUM(LeaveCount) BETWEEN 55 AND 59
    ) t;
GO

-- ── 5. vw_AgniveerBmi ─────────────────────────────────────────────────────
-- Cmd12_BmiOutliers — avg height/weight from MedicalRecordMaster, falling
-- back to AgniveerMaster's single stored Height/Weight when no medical
-- record exists. Buckets: <18.5 Underweight / <25 Normal / <30 Overweight /
-- else Obese.
-- ⚠ ASSUMES Height is stored in centimetres (both tables: decimal(18,2), no
-- unit column) — confirm this against the .NET input form / validation
-- range before trusting BMI output; if it's actually metres this view's BMI
-- values will be off by a factor of 10,000.
IF OBJECT_ID ('dbo.vw_AgniveerBmi', 'V') IS NOT NULL
DROP VIEW dbo.vw_AgniveerBmi;
GO
CREATE VIEW dbo.vw_AgniveerBmi AS
WITH
    MedicalAvg AS (
        SELECT
            AgniveerId,
            AVG(Height) AS AvgHeight,
            AVG(Weight) AS AvgWeight
        FROM dbo.MedicalRecordMaster
        WHERE
            Height IS NOT NULL
            AND Weight IS NOT NULL
        GROUP BY
            AgniveerId
    ),
    Resolved AS (
        SELECT
            a.Id AS AgniveerId,
            COALESCE(m.AvgHeight, a.Height) AS AvgHeight,
            COALESCE(m.AvgWeight, a.Weight) AS AvgWeight
        FROM dbo.AgniveerMaster a
            LEFT JOIN MedicalAvg m ON m.AgniveerId = a.Id
    )
SELECT
    AgniveerId,
    AvgHeight,
    AvgWeight,
    CASE
        WHEN AvgHeight > 0 THEN AvgWeight / POWER(AvgHeight / 100.0, 2)
        ELSE NULL
    END AS Bmi,
    CASE
        WHEN AvgHeight IS NULL
        OR AvgWeight IS NULL
        OR AvgHeight <= 0 THEN NULL
        WHEN AvgWeight / POWER(AvgHeight / 100.0, 2) < 18.5 THEN 'Underweight'
        WHEN AvgWeight / POWER(AvgHeight / 100.0, 2) < 25 THEN 'Normal'
        WHEN AvgWeight / POWER(AvgHeight / 100.0, 2) < 30 THEN 'Overweight'
        ELSE 'Obese'
    END AS BmiCategory
FROM Resolved;
GO

-- ── 6. vw_EquipmentDegraded ─────────────────────────────────────────────────
-- Cmd16 condition-rank comparison (Good=4 > Fair=3 > Poor=2 > Damaged=1) —
-- degraded when GivenCondition rank > ReturnCondition rank. Only meaningful
-- once the item has actually been returned.
-- ⚠ ASSUMES GivenCondition/ReturnCondition store exactly one of
-- Good/Fair/Poor/Damaged (case-insensitive) — confirm against the UI's
-- dropdown values; any other stored string ranks as NULL (excluded, not
-- miscounted as degraded).
IF OBJECT_ID (
    'dbo.vw_EquipmentDegraded',
    'V'
) IS NOT NULL
DROP VIEW dbo.vw_EquipmentDegraded;
GO
CREATE VIEW dbo.vw_EquipmentDegraded AS
SELECT
    ae.Id AS AssignmentId,
    ae.AgniveerId,
    e.Name AS EquipmentName,
    CASE
        WHEN (
            CASE UPPER(ae.GivenCondition)
                WHEN 'GOOD' THEN 4
                WHEN 'FAIR' THEN 3
                WHEN 'POOR' THEN 2
                WHEN 'DAMAGED' THEN 1
                ELSE NULL
            END
        ) > (
            CASE UPPER(ae.ReturnCondition)
                WHEN 'GOOD' THEN 4
                WHEN 'FAIR' THEN 3
                WHEN 'POOR' THEN 2
                WHEN 'DAMAGED' THEN 1
                ELSE NULL
            END
        ) THEN CAST(1 AS BIT)
        ELSE CAST(0 AS BIT)
    END AS IsDegraded
FROM dbo.AgniveerEquipment ae
    INNER JOIN dbo.EquipmentMaster e ON e.Id = ae.EquipmentId
WHERE
    ae.ReturnDateTime IS NOT NULL;
GO

-- ── 7. vw_AgniveerAttendanceStatus ───────────────────────────────────────────
-- Cmd26 / Cmd_AttendanceSummary — per-day historical row, one per
-- AgniveerAttendanceMaster record. Base IsPresent comes from the attendance
-- record itself; it's overridden to 0 when ANY leave record spans that date.
-- BUG FIX: earlier draft excluded IsAbscondedLeave from this override
-- (matching Cmd07_CurrentlyOnLeave's narrower rule), but this view backs
-- Cmd26/Cmd_AttendanceSummary, whose todaysLeaves query has no absconded
-- filter — confirmed to use Cmd26's broader rule, so absconded now counts
-- as on-leave/not-present here same as every other leave type.
IF OBJECT_ID (
    'dbo.vw_AgniveerAttendanceStatus',
    'V'
) IS NOT NULL
DROP VIEW dbo.vw_AgniveerAttendanceStatus;
GO
CREATE VIEW dbo.vw_AgniveerAttendanceStatus AS
SELECT
    att.AgniveerId,
    CAST(
        att.AttendanceDateTime AS DATE
    ) AS [Date],
    CASE
        WHEN EXISTS (
            SELECT 1
            FROM dbo.AgniveerLeaveMaster l
            WHERE
                l.AgniveerId = att.AgniveerId
                AND l.FromDate IS NOT NULL
                AND CAST(
                    att.AttendanceDateTime AS DATE
                ) >= CAST(l.FromDate AS DATE)
                AND (
                    l.ToDate IS NULL
                    OR CAST(
                        att.AttendanceDateTime AS DATE
                    ) <= CAST(l.ToDate AS DATE)
                )
        ) THEN CAST(0 AS BIT)
        ELSE att.IsPresent
    END AS IsPresent
FROM dbo.AgniveerAttendanceMaster att;
GO

-- ── 8. vw_AgniveerVerificationStatus ─────────────────────────────────────────
-- Cmd_VerificationByStatus — latest PoliceVerificationMaster record per
-- agniveer; Sent + NULL ReceivedDate -> NotResponded; agniveers with no
-- verification record at all -> Pending (LEFT JOIN from AgniveerMaster so
-- they still appear).
IF OBJECT_ID (
    'dbo.vw_AgniveerVerificationStatus',
    'V'
) IS NOT NULL
DROP VIEW dbo.vw_AgniveerVerificationStatus;
GO
CREATE VIEW dbo.vw_AgniveerVerificationStatus AS
WITH
    Latest AS (
        SELECT v.AgniveerId, v.PoliceStation, v.SentDate, v.ReceivedDate, v.Status, ROW_NUMBER() OVER (
                PARTITION BY
                    v.AgniveerId
                ORDER BY v.SentDate DESC, v.Id DESC
            ) AS rn
        FROM dbo.PoliceVerificationMaster v
    )
SELECT
    a.Id AS AgniveerId,
    CASE
        WHEN l.AgniveerId IS NULL THEN 'Pending'
        WHEN l.Status = 'Sent'
        AND l.ReceivedDate IS NULL THEN 'NotResponded'
        ELSE l.Status
    END AS Status,
    l.PoliceStation,
    l.SentDate,
    l.ReceivedDate,
    CASE
        WHEN l.SentDate IS NOT NULL THEN DATEDIFF (DAY, l.SentDate, GETDATE ())
        ELSE NULL
    END AS DaysSinceSent
FROM dbo.AgniveerMaster a
    LEFT JOIN Latest l ON l.AgniveerId = a.Id
    AND l.rn = 1;
GO